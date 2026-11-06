"""
training/trainer.py
====================
WHAT THIS FILE DOES:
    Full training loop with:
      - Thermal management  : GPU power cap + sleep between batches
      - Checkpoint saving   : saves every epoch (model + optimizer + scheduler)
      - Resume from checkpoint: picks up exactly where it left off
      - Mixed precision (fp16): faster training, less VRAM
      - Cosine LR + warmup  : stable convergence
      - Gradient clipping   : prevents exploding gradients
      - Training curves     : saved to results/

RESUME LOGIC:
    Every epoch end → saves results/checkpoints/checkpoint_epoch_N.pt
    Also saves results/checkpoints/best_model.pt (lowest val SMAPE)
    On restart → finds latest checkpoint → resumes from next epoch
    Scheduler + optimizer state fully restored → training identical to uninterrupted

THERMAL MANAGEMENT:
    - GPU memory fraction capped at 0.85 → leaves headroom, less heat
    - BATCH_SLEEP_MS delay between batches → GPU breathes between steps
    - torch.backends.cudnn.benchmark=False → no auto-tuning heat spikes
    - Temp logged every 50 batches → early warning if overheating
"""

import os
import time
import json
import torch
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.model import AmazonPricePredictor, HuberLoss, create_optimizer, compute_smape
from training.dataset import create_dataloaders





# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    "train_path"       : "data/train_augmented_v2.csv",
    "checkpoint_dir"   : "results/checkpoints",
    "results_dir"      : "results",

    # Model
    "model_name"       : "microsoft/deberta-v3-base",
    "max_seq_len"      : 160,
    "dropout"          : 0.25,    # 0.2 → 0.3: stronger regularization
    "hidden_dim"       : 512,
    "hidden_dim2"      : 256,

    # Training
    "epochs"           : 10,
    "batch_size"       : 16,
    "grad_accum_steps" : 1,
    "encoder_lr"       : 3e-5,   # 2e-5 → 1e-5: slower encoder fine-tuning
    "head_lr"          : 1e-3,   # 1e-3 → 5e-4: slower head learning
    "weight_decay"     : 0.01,   # 0.01 → 0.05: stronger L2 penalty
    "huber_delta"      : 0.5,
    "max_grad_norm"    : 1.0,
    "warmup_ratio"     : 0.2,   # 0.1 → 0.15: longer warmup, more stable
    "val_split"        : 0.1,
    "seed"             : 42,

    # Thermal management
    "gpu_memory_fraction" : 0.85,   # cap VRAM usage → less heat
    "batch_sleep_ms"      : 20,     # ms sleep between batches → GPU breathes
    "temp_warn_celsius"   : 78,     # log warning above this temp
    "temp_check_every"    : 50,     # check GPU temp every N batches
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Full reproducibility across Python/NumPy/PyTorch/CUDA."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # no auto-tune → less heat


def get_gpu_temp() -> int:
    """
    Reads GPU temperature via nvidia-smi.
    Returns integer celsius, or -1 if unavailable.

    WHY MONITOR TEMP:
        Laptop GPUs throttle at ~80°C → slower training
        Above 90°C → risk of damage
        We log a warning so you can pause if needed
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


def setup_thermal_management(config: dict, device: torch.device):
    """
    Applies GPU settings to reduce heat on laptop.

    MEMORY FRACTION (0.85):
        PyTorch pre-allocates a memory pool up to this fraction.
        Leaving 15% free reduces memory pressure → less heat.
        Formula: 8.5 GB × 0.85 = 7.2 GB max allocated.

    cudnn.benchmark = False:
        When True, CUDA runs multiple kernel implementations per op
        and picks the fastest → uses extra GPU cycles → more heat.
        We disable it — small speed loss, meaningful thermal benefit.
    """
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(
            config['gpu_memory_fraction']
        )
        torch.backends.cudnn.benchmark = False
        print(f"  Thermal: GPU memory capped at "
              f"{config['gpu_memory_fraction']*100:.0f}%")
        print(f"  Thermal: batch sleep = {config['batch_sleep_ms']}ms")
        print(f"  Thermal: temp warning at {config['temp_warn_celsius']}°C")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CHECKPOINT SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    history: dict,
    val_smape: float,
    config: dict,
    is_best: bool = False
):
    """
    Saves full training state so training can resume exactly.

    WHAT WE SAVE:
        model_state     : all 184M parameter weights
        optimizer_state : Adam momentum buffers (m, v for each param)
        scheduler_state : current LR position in cosine curve
        scaler_state    : GradScaler scale factor for fp16
        epoch           : which epoch just finished
        history         : all loss/SMAPE values so far
        val_smape       : this epoch's validation SMAPE
        config          : full hyperparameter dict

    WHY SAVE OPTIMIZER STATE:
        Adam keeps running averages (m, v) per parameter.
        Without them, resuming = cold start → first few batches
        have wrong gradient estimates → unstable loss spike.
        With them, resume is mathematically identical to uninterrupted.

    WHY SAVE SCHEDULER STATE:
        Scheduler tracks current step in cosine curve.
        Without it, LR resets to peak → overshoots on resume.

    NAMING CONVENTION:
        checkpoint_epoch_N.pt → one per epoch (can resume from any)
        best_model.pt         → lowest val SMAPE so far (for inference)
    """
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    state = {
        'epoch'          : epoch,
        'model_state'    : model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state'   : scaler.state_dict(),
        'history'        : history,
        'val_smape'      : val_smape,
        'config'         : config,
    }

    # Save per-epoch checkpoint
    epoch_path = Path(config['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pt"
    torch.save(state, epoch_path)
    print(f"  💾 Checkpoint saved → {epoch_path}")

    # Save best model separately (for inference)
    if is_best:
        best_path = Path(config['checkpoint_dir']) / "best_model.pt"
        torch.save(state, best_path)
        print(f"  ⭐ Best model saved → {best_path}")

    # Keep only last 2 epoch checkpoints to save disk space
    # (best_model.pt is always kept separately)
    checkpoints = sorted(
        Path(config['checkpoint_dir']).glob("checkpoint_epoch_*.pt")
    )
    if len(checkpoints) > 2:
        for old_ckpt in checkpoints[:-2]:
            old_ckpt.unlink()
            print(f"  🗑️  Removed old checkpoint: {old_ckpt.name}")


def find_latest_checkpoint(checkpoint_dir: str):
    """
    Finds the most recent epoch checkpoint to resume from.

    Returns:
        path to latest checkpoint, or None if no checkpoint exists
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None

    latest = checkpoints[-1]
    print(f"  Found checkpoint: {latest}")
    return str(latest)


def load_checkpoint(path: str, model, optimizer, scheduler, scaler, device):
    """
    Loads full training state for resuming.

    Returns:
        start_epoch : epoch to resume FROM (last completed + 1)
        history     : loss/SMAPE history so far
        best_smape  : best val SMAPE so far
    """
    print(f"\n  Loading checkpoint: {path}")
    state = torch.load(path, map_location=device)

    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    scaler.load_state_dict(state['scaler_state'])

    start_epoch = state['epoch'] + 1  # resume from NEXT epoch
    history     = state['history']
    best_smape  = min(history['val_smape']) if history['val_smape'] else float('inf')

    print(f"  Resuming from epoch {start_epoch}")
    print(f"  Best val SMAPE so far: {best_smape:.2f}%")

    return start_epoch, history, best_smape


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: TRAINING EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler,
    loss_fn, scaler, device, epoch, total_epochs, config
):
    """
    One full pass over training data with thermal management.

    THERMAL ADDITIONS vs basic trainer:
        1. batch_sleep_ms: time.sleep() between batches
           GPU runs → sleep → GPU runs → sleep (reduces sustained load)
        2. Temp check every 50 batches via nvidia-smi
           Logs warning if above temp_warn_celsius
        3. cudnn.benchmark=False set in setup_thermal_management
    """
    model.train()
    total_loss   = 0.0
    total_smape  = 0.0
    n_batches    = len(loader)
    grad_accum   = config['grad_accum_steps']
    sleep_sec    = config['batch_sleep_ms'] / 1000.0
    start_time   = time.time()

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):

        # ── Move to GPU ────────────────────────────────────────────────────
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        # ── Forward pass (fp16) ────────────────────────────────────────────
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = loss_fn(predictions, labels) / grad_accum

        # ── Backward pass ──────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step (every grad_accum batches) ──────────────────────
        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['max_grad_norm']
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # ── Thermal sleep ──────────────────────────────────────────────────
        # Small pause lets GPU cool between batches
        # 20ms per batch × 4219 batches = ~84s extra per epoch
        # Worth it to stay under 75°C
        time.sleep(sleep_sec)

        # ── Track metrics ──────────────────────────────────────────────────
        total_loss += loss.item() * grad_accum
        with torch.no_grad():
            batch_smape = compute_smape(predictions.float(), labels.float())
        total_smape += batch_smape

        # ── Logging + temp check ───────────────────────────────────────────
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
            elapsed   = time.time() - start_time
            remaining = elapsed / (batch_idx + 1) * (n_batches - batch_idx - 1)
            current_lr = scheduler.get_last_lr()[0]

            # Check GPU temp every temp_check_every batches
            temp_str = ""
            if (batch_idx + 1) % config['temp_check_every'] == 0:
                temp = get_gpu_temp()
                if temp > 0:
                    temp_str = f" 🌡️{temp}°C"
                    if temp >= config['temp_warn_celsius']:
                        print(f"  ⚠️  GPU temp {temp}°C — consider pausing!")

            print(f"  Epoch {epoch}/{total_epochs} "
                  f"[{batch_idx+1:4d}/{n_batches}] "
                  f"loss={loss.item()*grad_accum:.4f} "
                  f"smape={batch_smape:.2f}% "
                  f"lr={current_lr:.2e} "
                  f"eta={remaining/60:.1f}m"
                  f"{temp_str}")

    return total_loss / n_batches, total_smape / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, loader, loss_fn, device):
    """Evaluation on validation set — no gradients, no thermal sleep needed."""
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = loss_fn(predictions, labels)
            total_loss += loss.item()

            all_preds.append(predictions.cpu())
            all_targets.append(labels.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    val_smape   = compute_smape(all_preds, all_targets)
    val_loss    = total_loss / len(loader)

    return val_loss, val_smape


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str):
    """Saves loss + SMAPE curves. Goes in GitHub README."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val',   linewidth=2)
    ax1.set_title('Huber Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_smape'], 'b-o', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_smape'],   'r-o', label='Val',   linewidth=2)
    ax2.set_title('SMAPE (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('SMAPE (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle('DeBERTa-v3-base — Training Curves', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict = CONFIG):
    """
    Full training pipeline with resume support.

    RESUME FLOW:
        1. Check checkpoint_dir for latest checkpoint_epoch_N.pt
        2. If found → load model + optimizer + scheduler + scaler + history
        3. Set start_epoch = last_epoch + 1
        4. Continue training from there — identical to uninterrupted run
        5. If not found → fresh training from epoch 1
    """

    print(f"\n{'='*60}")
    print(f"  AMAZON PRICE PREDICTOR — TRAINING")
    print(f"{'='*60}")

    set_seed(config['seed'])
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        temp = get_gpu_temp()
        if temp > 0:
            print(f"  Temp   : {temp}°C (starting)")

    # Apply thermal settings
    setup_thermal_management(config, device)

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, tokenizer = create_dataloaders(
        train_path=config['train_path'],
        tokenizer_name=config['model_name'],
        max_length=config['max_seq_len'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        random_seed=config['seed']
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"\n  Building model: {config['model_name']}")
    model = AmazonPricePredictor(
        model_name=config['model_name'],
        dropout_rate=config['dropout'],
        hidden_dim=config['hidden_dim'],
        hidden_dim2=config['hidden_dim2']
    ).to(device).float()

    # ── Loss, optimizer, scheduler, scaler ────────────────────────────────
    loss_fn   = HuberLoss(delta=config['huber_delta'])
    optimizer = create_optimizer(
        model,
        encoder_lr=config['encoder_lr'],
        head_lr=config['head_lr'],
        weight_decay=config['weight_decay']
    )

    total_steps  = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # GradScaler: manages fp16 loss scaling
    scaler = GradScaler()

    # ── Resume or fresh start ──────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_smape': [], 'val_smape': []
    }
    best_val_smape = float('inf')
    best_epoch     = 0
    start_epoch    = 1

    latest_ckpt = find_latest_checkpoint(config['checkpoint_dir'])
    if latest_ckpt:
        print(f"\n  ✅ Resuming from checkpoint...")
        start_epoch, history, best_val_smape = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler, device
        )
        best_epoch = len(history['val_smape'])
    else:
        print(f"\n  🆕 No checkpoint found — starting fresh training")

    print(f"\n  Total steps  : {total_steps:,}")
    print(f"  Warmup steps : {warmup_steps:,}")
    print(f"  Start epoch  : {start_epoch}")
    print(f"  End epoch    : {config['epochs']}")

    if start_epoch > config['epochs']:
        print(f"\n  ✅ Training already complete ({config['epochs']} epochs done)!")
        return model, history, best_val_smape

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING (epochs {start_epoch}–{config['epochs']})")
    print(f"  Thermal sleep: {config['batch_sleep_ms']}ms/batch")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start = time.time()

        temp = get_gpu_temp()
        print(f"\n── Epoch {epoch}/{config['epochs']} "
              f"{'(GPU: ' + str(temp) + '°C)' if temp > 0 else ''} "
              f"─────────────────────")

        # Train
        train_loss, train_smape = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, scaler, device, epoch, config['epochs'], config
        )

        # Validate
        print(f"  Validating...")
        val_loss, val_smape = validate(model, val_loader, loss_fn, device)

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_smape'].append(train_smape)
        history['val_smape'].append(val_smape)

        epoch_time = time.time() - epoch_start
        is_best    = val_smape < best_val_smape

        if is_best:
            best_val_smape = val_smape
            best_epoch     = epoch

        print(f"\n  ── Epoch {epoch} Summary ──────────────────────────")
        print(f"  Train Loss  : {train_loss:.4f}  |  Val Loss  : {val_loss:.4f}")
        print(f"  Train SMAPE : {train_smape:.2f}%  |  Val SMAPE : {val_smape:.2f}%")
        print(f"  Time        : {epoch_time/60:.1f} min")
        print(f"  Best so far : epoch {best_epoch}, SMAPE={best_val_smape:.2f}%")

        # Save checkpoint every epoch
        save_checkpoint(
            epoch, model, optimizer, scheduler, scaler,
            history, val_smape, config, is_best=is_best
        )

        # Save history JSON after every epoch
        history_path = Path(config['results_dir']) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Update plot after every epoch
        if epoch >= 2:
            plot_training_curves(history, config['results_dir'])

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Best val SMAPE : {best_val_smape:.2f}%")
    print(f"  Model saved at : {config['checkpoint_dir']}/best_model.pt")
    print(f"{'='*60}\n")

    plot_training_curves(history, config['results_dir'])
    return model, history, best_val_smape


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, history, best_smape = train(CONFIG)