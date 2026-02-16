# Amazon ML Challenge 2025: Product Price Prediction

This repository contains the solution code for the Amazon ML Challenge 2025. The objective of this project was to predict product prices (`price`) based on catalog content (text descriptions, bullet points) and product images.

## 📌 Approach Overview

We approached this as a **multimodal regression problem**, utilizing a 2-stage stacking ensemble. The solution combines information from unstructured text, explicit numerical features extracted via Regex, and pre-computed image embeddings.

The evaluation metric for this challenge is **SMAPE** (Symmetric Mean Absolute Percentage Error).

### 1. Data Preprocessing & Feature Engineering
* **Text Cleaning:** Handled missing values in `catalog_content`.
* **Regex Feature Extraction:** Parsed catalog descriptions to extract key physical attributes that correlate with price:
    * `pack_quantity`: Extracted from patterns like "(Pack of 6)".
    * `weight_oz`: Extracted weight information (Ounces/Pounds).
    * `item_count`: Extracted count information (e.g., "12 Count").
    * `total_weight_oz`: Calculated as `weight_oz * pack_quantity`.
* **Target Transformation:** Applied `np.log1p` to the `price` variable to handle the heavy right-skew in price distribution.

### 2. Model Architecture (Stacking Ensemble)

We implemented a stacking pipeline with **5-Fold Cross-Validation** to prevent leakage.

#### Level 1: Base Models
1.  **LightGBM (Text + Numerical):** * **Features:** TF-IDF vectors (ngrams 1-3, max features 20k) concatenated with extracted numerical features.
    * **Objective:** Regression with L1 loss (MAE).
2.  **Transformer Fine-Tuning (DistilBERT):**
    * **Input Formatting:** Structured the input as `PACKS: {qty} | WEIGHT: {wt} [SEP] {description}` to give the LLM context on quantity.
    * **Model:** `distilbert-base-uncased` fine-tuned for regression.
    * **Framework:** HuggingFace Transformers & PyTorch.
3.  **Image Ridge Regression:**
    * **Features:** Pre-computed image embeddings (512-d).
    * **Model:** Ridge Regression on standard-scaled embeddings to capture visual signal.
4.  **Multimodal LightGBM:**
    * **Features:** A massive sparse matrix combining TF-IDF text features, numerical extraction, and image embeddings.

#### Level 2: Meta-Learner
* **Model:** **XGBoost Regressor**.
* **Input:** Out-of-Fold (OOF) predictions from all Level 1 models + original extracted numerical features.
* **Output:** Final log-price prediction (converted back via `expm1`).

## 🛠️ Tech Stack

* **Core:** `Python 3.11`, `Pandas`, `NumPy`
* **ML/DL:** `PyTorch`, `HuggingFace Transformers`, `LightGBM`, `XGBoost`, `Scikit-Learn`
* **Text Processing:** `Regex`, `TfidfVectorizer`

## 📂 Project Structure

```text
├── amazonml2025.ipynb   # Main notebook containing the full pipeline
├── models/              # Directory for saved model checkpoints (LGBM, PyTorch)
├── train_images/        # Training image assets
├── test_images/         # Test image assets
└── submission.csv       # Final predictions
