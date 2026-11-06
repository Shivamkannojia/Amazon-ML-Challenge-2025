"""
augmentation/regex_extractor_v2.py
====================================
WHAT CHANGED FROM v1:
    v1: Used regex to find "32 oz" embedded in raw text
        → noisy, misses explicit fields

    v2: Parses catalog_content LINE BY LINE (like top-5 team)
        → extracts explicit Value: and Unit: fields directly
        → extracts bullet points as rich description
        → matches their augmented_catalog_final_2.csv format exactly

WHY THIS MATTERS:
    catalog_content has STRUCTURED lines like:
        "Item Name: Heinz Ketchup\n"
        "Bullet Point 1: Made with real tomatoes\n"
        "Value: 32\n"
        "Unit: oz\n"
    
    Their SMAPE: 42.22% used these explicit fields.
    Our v1 tried to find "32 oz" in raw text → noisier signal.

OUTPUT COLUMNS (matches their augmented_catalog_final_2.csv):
    product_category  : item name (their column name)
    description       : bullet points + product description joined
    value             : numeric quantity (explicit Value: field)
    unit              : unit string (explicit Unit: field)
    price_per_unit    : price / value (KEY FEATURE we missed!)
    model_input       : structured prompt for DeBERTa

PROMPT FORMAT (matches their create_combined_text):
    "Category: [name] [SEP] description: [desc] [SEP] Amount: [val] [unit]"
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LINE-BY-LINE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_catalog_content(content: str) -> dict:
    """
    Parses catalog_content line by line to extract structured fields.

    CATALOG CONTENT FORMAT (actual Amazon ML Challenge data):
        Item Name: Heinz Tomato Ketchup
        Bullet Point 1: Made with real tomatoes, no artificial...
        Bullet Point 2: 32 oz squeeze bottle
        Product Description: Classic ketchup for everyday use
        Value: 32
        Unit: oz

    PARSING STRATEGY:
        - Split on newlines → process each line
        - Match by line prefix (startswith) → deterministic, no regex needed
        - Collect ALL bullet points into description → rich semantic content
        - Extract explicit Value/Unit → cleaner than embedded text search

    Args:
        content: raw catalog_content string

    Returns:
        dict with keys: product_category, description, value, unit
    """

    result = {
        'product_category': None,  # item name (their column naming)
        'description'     : None,  # bullet points + product desc
        'value'           : None,  # numeric quantity
        'unit'            : None,  # unit string
    }

    if not isinstance(content, str) or not content.strip():
        return result

    item_name_parts  = []
    misc_lines       = []  # bullet points + product description

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        # ── Item Name → product_category ─────────────────────────────────
        if line.startswith('Item Name'):
            parts = line.split(':', 1)
            if len(parts) > 1:
                item_name_parts.append(parts[1].strip())

        # ── Bullet Points → description ───────────────────────────────────
        # Handles: "Bullet Point 1:", "Bullet points I:", "Bullet Point:"
        elif line.startswith('Bullet Point') or line.startswith('Bullet point'):
            parts = line.split(':', 1)
            if len(parts) > 1 and parts[1].strip():
                misc_lines.append(parts[1].strip())

        # ── Product Description → description ─────────────────────────────
        elif line.startswith('Product Description'):
            parts = line.split(':', 1)
            if len(parts) > 1 and parts[1].strip():
                misc_lines.append(parts[1].strip())

        # ── Explicit Value field ───────────────────────────────────────────
        # This is the KEY field we missed in v1!
        elif line.startswith('Value:'):
            value_str = line.replace('Value:', '', 1).strip()
            try:
                result['value'] = float(value_str)
            except (ValueError, TypeError):
                result['value'] = None

        # ── Explicit Unit field ───────────────────────────────────────────
        elif line.startswith('Unit:'):
            unit_str = line.replace('Unit:', '', 1).strip()
            if unit_str:
                result['unit'] = unit_str

    # ── Assemble fields ────────────────────────────────────────────────────
    result['product_category'] = (
        ' '.join(item_name_parts).strip() if item_name_parts else None
    )
    result['description'] = (
        '\n'.join(misc_lines).strip() if misc_lines else None
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PROMPT BUILDER (matches their create_combined_text exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_prompt(row: pd.Series) -> str:
    """
    Builds structured DeBERTa input prompt.

    THEIR FORMAT (from b2_with_data.py):
        "Category: [name] [SEP] description: [desc] [SEP] Amount: [val] [unit]"

    DESIGN:
        - Category = item name (strongest price signal: brand + product)
        - description = bullet points (rich semantic content about features)
        - Amount = explicit value + unit (cleaner than our v1 regex extraction)
        - Skip Amount if unit is 'count' and value is 1 (default imputed value)

    WHY DESCRIPTION MATTERS:
        "32 oz squeeze bottle, made with real tomatoes, no HFCS" →
        DeBERTa learns: squeeze bottle → consumer product → ~$4-6
        vs just "Heinz Ketchup 32 oz" → less context

    Args:
        row: DataFrame row with product_category, description, value, unit

    Returns:
        str: structured prompt
    """

    parts = []

    # ── Category (item name) ──────────────────────────────────────────────
    category = row.get('product_category')
    if pd.notna(category) and str(category).strip():
        parts.append(f"Category: {str(category).strip()}")
    else:
        # Fallback: first 150 chars of raw catalog text
        raw = str(row.get('catalog_content', ''))
        parts.append(f"Category: {raw[:150]}")

    # ── Description (bullet points) ───────────────────────────────────────
    description = row.get('description')
    if pd.notna(description) and str(description).strip():
        # Truncate to 200 chars — most signal is in first few bullet points
        desc_text = str(description).strip()[:200]
        parts.append(f"description: {desc_text}")

    # ── Amount (value + unit) ─────────────────────────────────────────────
    value = row.get('value')
    unit  = row.get('unit')

    # Skip if it's just the default imputed value (1 Count)
    if pd.notna(value) and pd.notna(unit):
        unit_str = str(unit).strip()
        skip = (unit_str.lower() == 'count' and float(value) == 1.0)
        if not skip:
            parts.append(f"Amount: {value} {unit_str.capitalize()}")

    return " [SEP] ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PRICE PER UNIT (KEY FEATURE WE MISSED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_price_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price_per_unit = price / value.

    WHY THIS HELPS:
        Raw price: $7.99 for 64 oz ketchup
        price_per_unit: $0.12/oz

        DeBERTa sees "64 oz" in text and predicts price.
        But if we give price_per_unit as context, the model learns:
        "this product costs $0.12/oz" → much easier regression target.

        Top-5 team included this in their augmented CSV.
        This feature alone likely accounts for 1-2% SMAPE improvement.

    FORMULA:
        if value is valid (not NaN, not 0):
            price_per_unit = price / value
        else:
            price_per_unit = price  (fallback: treat as unit price)

    Note: Only computed for training data (test has no price column)
    """

    if 'price' not in df.columns:
        return df

    is_valid = (df['value'].notna()) & (df['value'] > 0)

    df['price_per_unit'] = np.where(
        is_valid,
        df['price'] / df['value'],
        df['price']
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: IMPUTE MISSING VALUES (replaces LLM imputer for our case)
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing value/unit with sensible defaults.

    THEIR APPROACH:
        - Missing value → 1
        - Missing unit  → 'Count'
        This means "1 unit of the product" — safe default for products
        where quantity isn't specified (e.g. a single jar, single bottle)

    WHY NOT LLM:
        Our regex extracted value/unit from explicit Value:/Unit: fields.
        If those fields exist → extracted.
        If they don't exist → product is likely a single unit → default to 1 Count.
        LLM imputation is expensive and only needed when the field exists
        but has an ambiguous value (their case with embedded quantities).

    MISSING STATS (expected after v2 extraction):
        - value missing: rows without explicit "Value:" field
        - unit missing:  rows without explicit "Unit:" field
    """

    missing_value = df['value'].isna().sum()
    missing_unit  = df['unit'].isna().sum()

    print(f"  Imputing {missing_value:,} missing values → 1")
    print(f"  Imputing {missing_unit:,} missing units  → 'Count'")

    df['value'] = df['value'].fillna(1.0)
    df['unit']  = df['unit'].fillna('Count')

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction_v2(input_path: str, output_path: str, is_test: bool = False):
    """
    Full v2 extraction pipeline.

    Steps:
        1. Load CSV
        2. Parse catalog_content line by line → structured fields
        3. Impute missing value/unit with defaults
        4. Compute price_per_unit (train only)
        5. Build DeBERTa prompts
        6. Save augmented CSV

    Args:
        input_path  : path to train.csv or test.csv
        output_path : where to save augmented CSV
        is_test     : True for test set (no price column)
    """

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION PIPELINE v2 (line-by-line parser)")
    print(f"{'='*60}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Mode  : {'TEST' if is_test else 'TRAIN'}\n")

    # ── Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df):,} rows × {df.shape[1]} columns")

    # ── Parse line by line ─────────────────────────────────────────────────
    print(f"  Parsing catalog_content line by line...")
    extracted = df['catalog_content'].apply(parse_catalog_content)
    extracted_df = pd.DataFrame(extracted.tolist())
    df = pd.concat([df, extracted_df], axis=1)

    # ── Stats ──────────────────────────────────────────────────────────────
    total = len(df)
    print(f"\n  ── Extraction Results ──────────────────────────────")
    print(f"  product_category found : {df['product_category'].notna().sum():,} / {total:,}")
    print(f"  description found      : {df['description'].notna().sum():,} / {total:,}")
    print(f"  value found            : {df['value'].notna().sum():,} / {total:,}")
    print(f"  unit found             : {df['unit'].notna().sum():,} / {total:,}")

    # ── Impute missing ─────────────────────────────────────────────────────
    print(f"\n  ── Imputing Missing Values ─────────────────────────")
    df = impute_missing_values(df)

    # ── Price per unit (train only) ────────────────────────────────────────
    if not is_test:
        print(f"\n  ── Computing price_per_unit ────────────────────────")
        df = compute_price_per_unit(df)
        print(f"  price_per_unit stats:")
        print(f"    mean={df['price_per_unit'].mean():.3f}, "
              f"median={df['price_per_unit'].median():.3f}, "
              f"max={df['price_per_unit'].max():.3f}")

    # ── Build prompts ──────────────────────────────────────────────────────
    print(f"\n  ── Building DeBERTa Prompts ────────────────────────")
    df['model_input'] = df.apply(build_model_prompt, axis=1)

    # Show samples
    print(f"\n  Sample prompts:")
    for i, row in df.head(3).iterrows():
        print(f"  [{i}] {row['model_input'][:120]}...")

    # ── Save ───────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")
    print(f"{'='*60}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Process train
    train_df = run_extraction_v2(
        input_path="data/train.csv",
        output_path="data/train_augmented_v2.csv",
        is_test=False
    )

    # Process test
    test_df = run_extraction_v2(
        input_path="data/test.csv",
        output_path="data/test_augmented_v2.csv",
        is_test=True
    )

    print("✅ v2 augmentation complete.")
    print("   Files: data/train_augmented_v2.csv, data/test_augmented_v2.csv")
    print("   Next: update dataset.py to use train_augmented_v2.csv")
    print("  then run Run 4 with DeBERTa-v3-large")