"""
experiments/01_xgboost_tfidf_baseline.py
=========================================
Simplest possible baseline:
    TF-IDF vectors from text → XGBoost regression

This is the starting point for the comparison table.
No neural networks, no transformers, no GPU needed.
Runs in under 2 minutes.

TF-IDF: Term Frequency - Inverse Document Frequency
    Converts text to a sparse vector of word importance scores.
    Common words (the, a, is) get low scores.
    Rare but meaningful words (ketchup, organic) get high scores.
    No understanding of meaning — just word frequency statistics.

XGBoost: Extreme Gradient Boosting
    Ensemble of decision trees built sequentially.
    Each tree corrects errors of the previous one.
    Fast, handles sparse features well, good baseline for tabular data.

Expected SMAPE: ~52-55%
    Shows how much signal is in raw word frequencies alone.
    Any neural model beating this is actually learning semantics.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
from pathlib import Path
import time


def compute_smape(actual, predicted):
    """
    SMAPE = mean(|actual - pred| / ((|actual| + |pred|) / 2)) * 100
    Lower is better. Range: 0% to 200%.
    """
    numerator   = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100


def run_xgboost_tfidf():

    print("=" * 60)
    print("  Experiment 01 — XGBoost + TF-IDF Baseline")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv("data/train_augmented.csv")
    print(f"  Rows: {len(df):,}")

    # Use catalog_content as raw text input
    # No preprocessing — pure TF-IDF on raw text
    # X = df["catalog_content"].fillna("").values
    # y = np.log1p(df["price"].values)

    X = df["catalog_content"].fillna("").tolist()
    y = np.log1p(df["price"].astype(float).to_numpy())

    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42 )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")

    # TF-IDF vectorization
    # max_features=50000: keep only top 50k terms by frequency
    # ngram_range=(1,2): include single words AND word pairs
    #   "tomato" and "tomato sauce" are both features
    # sublinear_tf=True: apply log(tf) instead of raw tf
    #   prevents very frequent words from dominating
    print("\n[2/4] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    print(f"  Feature matrix: {X_train_tfidf.shape}")

    # XGBoost regression
    # n_estimators=500: 500 trees
    # max_depth=6: each tree can have up to 6 levels
    # learning_rate=0.1: each tree contributes 10% of its prediction
    # subsample=0.8: each tree trains on 80% of rows (prevents overfit)
    # colsample_bytree=0.8: each tree uses 80% of features
    # tree_method="hist": faster algorithm for sparse data
    print("\n[3/4] Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        # device="cuda",
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    start = time.time()
    model.fit(
        X_train_tfidf, y_train,
        eval_set=[(X_val_tfidf, y_val)],
        verbose=100
    )
    elapsed_mins = round((time.time() - start) / 60, 1)

    # Evaluate
    print("\n[4/4] Evaluating...")
    val_preds_log = model.predict(X_val_tfidf)

    # Reverse log1p transform
    val_preds   = np.expm1(val_preds_log).clip(0.01, 10000)
    val_actuals = np.expm1(y_val)

    smape = compute_smape(val_actuals, val_preds)
    mae   = np.mean(np.abs(val_actuals - val_preds))

    print(f"\n  Results:")
    print(f"    SMAPE : {smape:.2f}%")
    print(f"    MAE   : ${mae:.2f}")

    # Save result for comparison notebook
    result = {
        "experiment"  : "XGBoost + TF-IDF",
        "description" : "TF-IDF (50k features, bigrams) + XGBoost (500 trees)",
        "smape"       : round(smape, 2),
        "mae"         : round(mae, 2),
        "training_time_mins" : elapsed_mins,
        "requires_gpu": False
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/exp01_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved → results/exp01_result.json")
    print(f"\n{'=' * 60}\n")
    return result


if __name__ == "__main__":
    result = run_xgboost_tfidf()
    print(f"Final SMAPE: {result['smape']}%")