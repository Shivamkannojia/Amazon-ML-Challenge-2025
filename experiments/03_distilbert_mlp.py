"""
experiments/03_distilbert_mlp.py
==================================
DistilBERT encoder + MLP regression head (fine-tuned end-to-end)

Step up from frozen SentenceTransformer:
    Experiment 02: frozen encoder → XGBoost (no task-specific learning)
    Experiment 03: fine-tuned encoder → trained MLP (learns price signals)

DistilBERT:
    Distilled (compressed) version of BERT
    6 transformer layers vs BERT's 12
    66M params vs BERT's 110M
    40% smaller, 60% faster, retains 97% of BERT's quality
    Good for quick experiments when DeBERTa is too slow

Same architecture as our main model (model.py) but smaller encoder.
Same training loop as trainer.py but simplified — no checkpoint resume,
no thermal management, just clean training for comparison.

Expected SMAPE: ~44-46%
    Better than frozen embeddings because encoder adapts to price prediction.
    Worse than DeBERTa because smaller model, less pretraining quality.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.dataset import AmazonPriceDataset


def compute_smape(actual, predicted):
    numerator   = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100


class DistilBERTPricePredictor(nn.Module):
    """
    Same architecture as our main AmazonPricePredictor (model.py)
    but using DistilBERT instead of DeBERTa-v3-base.

    DistilBERT hidden size = 768 (same as DeBERTa-base).
    So MLP head dimensions are identical.
    Only difference: encoder quality and pretraining.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size  = self.encoder.config.hidden_size   # 768

        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        # Mean pooling — same as our Run 3 model
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self.regression_head(pooled).squeeze(-1)


def run_distilbert():

    print("=" * 60)
    print("  Experiment 03 — DistilBERT + MLP Head")
    print("=" * 60)

    EPOCHS     = 5
    BATCH_SIZE = 32
    ENCODER_LR = 2e-5
    HEAD_LR    = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv("data/train_augmented.csv")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    # Tokenizer + datasets
    print("\n[2/4] Building datasets...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = AmazonPriceDataset(train_df, tokenizer, max_length=160, is_test=False)
    val_dataset   = AmazonPriceDataset(val_df,   tokenizer, max_length=160, is_test=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\n[3/4] Building model...")
    model = DistilBERTPricePredictor(dropout=0.2).to(device).float()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Dual learning rates — same as main model
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),          "lr": ENCODER_LR},
        {"params": model.regression_head.parameters(),  "lr": HEAD_LR}
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.HuberLoss(delta=0.5)
    scaler       = torch.amp.GradScaler()

    # Training loop
    print(f"\n[4/4] Training {EPOCHS} epochs...")
    best_smape = float("inf")

    start = time.time()
    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss  = loss_fn(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(loss.item())

        elapsed_mins = round((time.time() - start) / 60, 1)

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds   = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                all_preds.extend(torch.expm1(preds).cpu().numpy())
                all_targets.extend(torch.expm1(batch["labels"]).numpy())

        preds_arr   = np.array(all_preds).clip(0.01, 10000)
        targets_arr = np.array(all_targets)
        val_smape   = compute_smape(targets_arr, preds_arr)
        best_smape  = min(best_smape, val_smape)

        print(f"  Epoch {epoch}/{EPOCHS} — train_loss={np.mean(train_losses):.4f}  val_smape={val_smape:.2f}%")

    print(f"\n  Best SMAPE: {best_smape:.2f}%")

    result = {
        "experiment"         : "DistilBERT + MLP Head",
        "description"        : "distilbert-base-uncased (fine-tuned) + LayerNorm + MLP (512→256→1), mean pooling",
        "smape"              : float(round(best_smape, 2)),
        "mae"                : None,
        "epochs"             : EPOCHS,
        "training_time_mins" : elapsed_mins,
        "requires_gpu"       : True
    }
    with open("results/exp03_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved → results/exp03_result.json")
    print(f"\n{'=' * 60}\n")
    return result


if __name__ == "__main__":
    result = run_distilbert()
    print(f"Final SMAPE: {result['smape']}%")