"""
experiments/04_model_comparison.py
====================================
Loads results from all experiments and builds:
    - Final comparison table (printed + saved as CSV)
    - SMAPE comparison bar chart (saved as PNG)

Run after all experiment scripts have finished:
    python experiments/01_xgboost_tfidf_baseline.py
    python experiments/02_sentence_embeddings_xgboost.py
    python experiments/03_distilbert_mlp.py
    python experiments/04_model_comparison.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_results():
    """
    Loads experiment JSON files from results/ folder.
    Adds DeBERTa Run 6 (our best) manually.
    """
    results = []

    exp_files = [
        "results/exp01_result.json",
        "results/exp02_result.json",
        "results/exp03_result.json",
    ]

    for path in exp_files:
        if Path(path).exists():
            with open(path) as f:
                results.append(json.load(f))
        else:
            print(f"  Warning: {path} not found — run experiment first")

    # Our best DeBERTa run
    results.append({
        "experiment"         : "DeBERTa-v3-base (Ours)",
        "description"        : "Mean pooling, LayerNorm, v2 data augmentation, encoder_lr=3e-5",
        "smape"              : 45.12,
        "mae"                : None,
        "epochs"             : 5,
        "training_time_mins" : 74,
        "requires_gpu"       : True
    })

    return results


def build_comparison_table(results):
    """Builds comparison DataFrame sorted by SMAPE."""
    df = pd.DataFrame(results)
    df = df.sort_values("smape", ascending=True).reset_index(drop=True)
    df.index += 1

    print("\n" + "=" * 75)
    print("  MODEL COMPARISON — SMAPE (lower is better)")
    print("=" * 75)
    print(f"  {'Rank':<5} {'Model':<38} {'SMAPE':>8}  {'GPU':>5}")
    print("  " + "-" * 60)

    for rank, row in df.iterrows():
        gpu_str = "Yes" if row.get("requires_gpu") else "No"
        print(f"  {rank:<5} {row['experiment']:<38} {row['smape']:>7.2f}%  {gpu_str:>5}")

    print("=" * 75)
    return df


def plot_comparison(df):
    """
    Horizontal bar chart — SMAPE by model.
    Color: green = best, orange = mid, red = worst.
    Reference line at original competition SMAPE (48.71%).
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    models = df["experiment"].values[::-1]
    smapes = df["smape"].values[::-1]

    colors = []
    for s in smapes:
        if s < 47:
            colors.append("#27ae60")
        elif s < 52:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    bars = ax.barh(models, smapes, color=colors, edgecolor="white", height=0.55)

    for bar, smape in zip(bars, smapes):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{smape:.2f}%",
            va="center", ha="left",
            fontsize=10, fontweight="bold"
        )

    # Original competition score reference line
    ax.axvline(x=48.71, color="#7f8c8d", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(48.71 + 0.2, -0.6, "Original\n48.71%", fontsize=8.5, color="#7f8c8d")

    ax.set_xlabel("SMAPE (%) — Lower is Better", fontsize=11)
    ax.set_title(
        "Amazon ML Challenge 2025 — Model Comparison\nPrice Prediction SMAPE by Approach",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlim(38, 68)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    green_patch  = mpatches.Patch(color="#27ae60", label="< 47% (beats original)")
    orange_patch = mpatches.Patch(color="#f39c12", label="47–52%")
    red_patch    = mpatches.Patch(color="#e74c3c", label="> 52%")
    ax.legend(handles=[green_patch, orange_patch, red_patch],
              loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path = "results/model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {save_path}")


def main():
    print("\n" + "=" * 60)
    print("  Experiment 04 — Model Comparison")
    print("=" * 60)

    Path("results").mkdir(exist_ok=True)

    results = load_results()
    print(f"\n  Loaded {len(results)} experiments")

    df = build_comparison_table(results)

    csv_path = "results/model_comparison.csv"
    df.to_csv(csv_path)
    print(f"\n  Table saved → {csv_path}")

    plot_comparison(df)

    print("\n  Files ready:")
    print("    results/model_comparison.csv")
    print("    results/model_comparison.png")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()