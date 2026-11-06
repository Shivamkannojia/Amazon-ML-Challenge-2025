"""
inference/predict.py
=====================
Loads best_model.pt → runs on test set → saves submission.csv

Usage:
    python inference/predict.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.model import AmazonPricePredictor
from training.dataset import AmazonPriceDataset, build_prompt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match the run you want to use for inference
# ─────────────────────────────────────────────────────────────────────────────
INFER_CONFIG = {
    "model_name"    : "microsoft/deberta-v3-base",
    "checkpoint"    : "results/checkpoints/best_model.pt",
    "test_path"     : "data/test_augmented_v2.csv",    # v1 data (Run 3 was trained on v1)
    "output_path"   : "results/submission.csv",
    "max_seq_len"   : 160,
    "batch_size"    : 32,                           # larger batch fine for inference
    "hidden_dim"    : 512,
    "hidden_dim2"   : 256,
    "dropout"       : 0.25,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(config: dict = INFER_CONFIG):
    """
    Full inference pipeline:
        1. Load test CSV
        2. Build prompts (same as training)
        3. Load best model weights
        4. Run forward pass (no gradients)
        5. Apply expm1 to reverse log1p transform
        6. Save submission.csv

    KEY: must use SAME prompt format and tokenizer as training.
         If training used v1 prompts → inference uses v1 prompts.
         Mismatch = garbage predictions.
    """

    print(f"\n{'='*60}")
    print(f"  INFERENCE PIPELINE")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    # ── Load test data ─────────────────────────────────────────────────────
    print(f"\n  Loading test data: {config['test_path']}")
    test_df = pd.read_csv(config['test_path'])
    print(f"  Test rows: {len(test_df):,}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"  Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # ── Build dataset (is_test=True → no labels) ──────────────────────────
    print(f"  Building test dataset...")
    test_dataset = AmazonPriceDataset(
        test_df,
        tokenizer,
        max_length=config['max_seq_len'],
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,          # NEVER shuffle test data
        num_workers=0,
        pin_memory=False
    )
    print(f"  Test batches: {len(test_loader):,}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n  Loading model from: {config['checkpoint']}")
    model = AmazonPricePredictor(
        model_name=config['model_name'],
        dropout_rate=config['dropout'],
        hidden_dim=config['hidden_dim'],
        hidden_dim2=config['hidden_dim2']
    ).to(device).float()

    # Load checkpoint
    checkpoint = torch.load(config['checkpoint'], map_location=device)

    # Checkpoint might be full state (with optimizer etc.) or just weights
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Val SMAPE at save: {checkpoint.get('val_smape', '?'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded model weights")

    model.eval()    # disable dropout for inference

    # ── Run inference ──────────────────────────────────────────────────────
    print(f"\n  Running inference...")
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass — output is log1p(price)
            log_predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Reverse log1p transform → actual prices
            # expm1(x) = exp(x) - 1 = inverse of log1p(x)
            predictions = torch.expm1(log_predictions)

            # Clamp to valid price range
            # No product should be free ($0) or over $10,000
            predictions = torch.clamp(predictions, min=0.01, max=10000.0)

            all_predictions.extend(predictions.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)}")

    print(f"  Generated {len(all_predictions):,} predictions")

    # ── Build submission ───────────────────────────────────────────────────
    print(f"\n  Building submission...")
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price'    : all_predictions
    })

    # Sanity check predictions
    print(f"  Prediction stats:")
    print(f"    min   : ${submission['price'].min():.2f}")
    print(f"    max   : ${submission['price'].max():.2f}")
    print(f"    mean  : ${submission['price'].mean():.2f}")
    print(f"    median: ${submission['price'].median():.2f}")

    # Check for any NaN or negative predictions
    n_nan = submission['price'].isna().sum()
    n_neg = (submission['price'] < 0).sum()
    if n_nan > 0:
        print(f"  ⚠️  {n_nan} NaN predictions — filling with median")
        submission['price'].fillna(submission['price'].median(), inplace=True)
    if n_neg > 0:
        print(f"  ⚠️  {n_neg} negative predictions — clamping to 0.01")
        submission['price'] = submission['price'].clip(lower=0.01)

    # ── Save ───────────────────────────────────────────────────────────────
    Path(config['output_path']).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config['output_path'], index=False)

    print(f"\n  ✅ Submission saved → {config['output_path']}")
    print(f"  Rows: {len(submission):,}")
    print(f"\n  Sample predictions:")
    print(submission.head(10).to_string(index=False))
    print(f"\n{'='*60}\n")

    return submission


if __name__ == "__main__":
    submission = run_inference(INFER_CONFIG)