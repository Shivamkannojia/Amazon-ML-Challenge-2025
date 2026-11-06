"""
experiments/02_sentence_embeddings_xgboost.py
===============================================
SentenceTransformer embeddings → XGBoost regression

Step up from TF-IDF:
    TF-IDF: sparse word counts, no understanding of meaning
    SentenceTransformer: dense 384-dim semantic vectors
        "ketchup" and "tomato sauce" are close in embedding space
        TF-IDF treats them as completely different words

Model: all-MiniLM-L6-v2
    Small, fast sentence embedding model (22M params)
    Produces 384-dim vectors from text
    Pretrained on large corpus — understands product semantics
    No fine-tuning here — frozen encoder + XGBoost on top

This is what our original pipeline was roughly doing
(Sentence Transformers + XGBoost from the comparison table).

Expected SMAPE: ~49-52%
    Better than TF-IDF because embeddings capture semantics.
    Worse than fine-tuned DeBERTa because encoder is not task-specific.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
from pathlib import Path
import time


def compute_smape(actual, predicted):
    numerator   = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100


def run_sentence_xgboost():

    print("=" * 60)
    print("  Experiment 02 — SentenceTransformer + XGBoost")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv("data/train_augmented.csv")
    print(f"  Rows: {len(df):,}")

    # Use our structured prompt as input
    # Better than raw catalog_content — already cleaned and structured
    # X_text = df["model_input"].fillna("").values
    # y      = np.log1p(df["price"].values)
    X_text = df["catalog_content"].fillna("").tolist()
    y = np.log1p(df["price"].astype(float).to_numpy())

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.1, random_state=42
    )
    print(f"  Train: {len(X_train_text):,}  Val: {len(X_val_text):,}")

    # Load sentence transformer
    # all-MiniLM-L6-v2: good balance of speed and quality
    # 384-dim output, runs fast even on CPU
    print("\n[2/4] Loading SentenceTransformer...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Model loaded")

    # Encode text to dense vectors
    # batch_size=256: encode 256 sentences at once (faster)
    # show_progress_bar=True: shows progress for large datasets
    print("\n[3/4] Encoding text → embeddings...")
    print("  Encoding train set...")
    X_train_emb = encoder.encode(
        X_train_text,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("  Encoding val set...")
    X_val_emb = encoder.encode(
        X_val_text,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    from sklearn.preprocessing import StandardScaler

    # After encoding, before XGBoost:
    scaler_emb = StandardScaler()
    X_train_emb = scaler_emb.fit_transform(X_train_emb)
    X_val_emb   = scaler_emb.transform(X_val_emb)
    print(f"  Embedding shape: {X_train_emb.shape}")

    # XGBoost on embeddings
    # Same config as Experiment 01 for fair comparison
    print("\n[4/4] Training XGBoost on embeddings...")
    model = xgb.XGBRegressor(
    n_estimators=1000,       # more trees for dense features
    max_depth=8,             # deeper trees — 384 dims need more splits
    learning_rate=0.05,      # slower learning — more stable
    subsample=0.8,
    colsample_bytree=1.0,    # use ALL 384 features every tree (was 0.8)
    min_child_weight=3,      # prevents overfit on dense space
    tree_method="hist",
    device="cuda",
    random_state=42,
    verbosity=1
    )
    start = time.time()
    model.fit(
        X_train_emb, y_train,
        eval_set=[(X_val_emb, y_val)],
        verbose=100
    )
    elapsed_mins = round((time.time() - start) / 60, 1)

    # Evaluate
    val_preds_log = model.predict(X_val_emb)
    val_preds     = np.expm1(val_preds_log).clip(0.01, 10000)
    val_actuals   = np.expm1(y_val)

    smape = compute_smape(val_actuals, val_preds)
    mae   = np.mean(np.abs(val_actuals - val_preds))

    print(f"\n  Results:")
    print(f"    SMAPE : {smape:.2f}%")
    print(f"    MAE   : ${mae:.2f}")

    result = {
        "experiment"         : "SentenceTransformer + XGBoost",
        "description"        : "all-MiniLM-L6-v2 (frozen, 384-dim) + XGBoost",
        "smape"              : round(smape, 2),
        "mae"                : round(mae, 2),
        "training_time_mins" : elapsed_mins,
        "requires_gpu"       : False
    }
    with open("results/exp02_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved → results/exp02_result.json")
    print(f"\n{'=' * 60}\n")
    return result


if __name__ == "__main__":
    result = run_sentence_xgboost()
    print(f"Final SMAPE: {result['smape']}%")