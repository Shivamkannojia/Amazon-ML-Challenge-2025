"""
augmentation/regex_extractor.py
================================
WHAT THIS FILE DOES:
    Extracts structured fields (item_name, item_quantity, unit) from
    semi-structured Amazon catalog_content strings using regular expressions.

WHY REGEX FIRST (not LLM):
    - 75,000 rows × LLM API call = expensive and slow
    - ~80-85% of rows follow predictable patterns like:
        "Item Name: Heinz Ketchup, 32 oz, Pack of 2 ..."
    - Regex handles these instantly and deterministically
    - LLM is called ONLY for the remaining ~15-20% (see llm_imputer.py)
    This hybrid approach reduces cost a lot.

WHAT WE EXTRACT:
    - item_name  : product name (e.g., "Heinz Ketchup")
    - item_value : numeric quantity (e.g., 32.0)
    - item_unit  : measurement unit (e.g., "oz")
    - category   : product category if present
    - pack_count : number of units in pack (e.g., 2 from "Pack of 2")

OUTPUT:
    data/train_augmented.csv  ← original data + new extracted columns
    data/regex_failures.csv   ← rows where regex failed (fed to LLM next)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: REGEX PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
# re.compile() pre-compiles the pattern → faster when applied to 75k rows
# re.IGNORECASE → matches "oz", "OZ", "Oz" all the same way

# Pattern: "Item Name: <name>," or "Item Name: <name>\n"
# Group 1 captures everything after "Item Name: " until comma or newline
ITEM_NAME_PATTERN = re.compile(
    r'Item\s+Name\s*:\s*([^,\n]+)',
    re.IGNORECASE
)

# Pattern: captures a number (int or decimal) followed by a unit
# Examples: "32 oz", "1.5 kg", "500ml", "2.5 lbs"
# \s* means optional whitespace between number and unit
QUANTITY_UNIT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(oz|fl\.?\s*oz|lb|lbs|kg|g|ml|l|ct|count|pack|piece|pcs|tablet|capsule|gallon|gal|quart|qt|pt|pint|ounce|pound|gram|liter|litre)s?\b',
    re.IGNORECASE
)

# Pattern: "Pack of 2" or "Set of 4" or "Count of 6"
PACK_COUNT_PATTERN = re.compile(
    r'(?:pack|set|count|box|bundle)\s+of\s+(\d+)',
    re.IGNORECASE
)

# Pattern: "Category: <category>" — not always present
CATEGORY_PATTERN = re.compile(
    r'Category\s*:\s*([^\n,|]+)',
    re.IGNORECASE
)

# Pattern: "Item Package Quantity: 2" or "IPQ: 2"
IPQ_PATTERN = re.compile(
    r'(?:Item\s+Package\s+Quantity|IPQ)\s*[:\-]\s*(\d+)',
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: UNIT NORMALIZATION MAP
# ─────────────────────────────────────────────────────────────────────────────
# Standardize unit variants so the model sees consistent tokens
# "fl oz", "fl. oz", "floz" → all become "fl_oz"
UNIT_NORMALIZATION = {
    'oz': 'oz',
    'ounce': 'oz',
    'ounces': 'oz',
    'fl oz': 'fl_oz',
    'fl. oz': 'fl_oz',
    'fl.oz': 'fl_oz',
    'floz': 'fl_oz',
    'lb': 'lb',
    'lbs': 'lb',
    'pound': 'lb',
    'pounds': 'lb',
    'kg': 'kg',
    'kilogram': 'kg',
    'g': 'g',
    'gram': 'g',
    'grams': 'g',
    'ml': 'ml',
    'milliliter': 'ml',
    'l': 'liter',
    'liter': 'liter',
    'litre': 'liter',
    'liters': 'liter',
    'litres': 'liter',
    'ct': 'count',
    'count': 'count',
    'pack': 'pack',
    'pcs': 'pcs',
    'piece': 'pcs',
    'pieces': 'pcs',
    'tablet': 'tablet',
    'capsule': 'capsule',
    'gallon': 'gallon',
    'gal': 'gallon',
    'quart': 'quart',
    'qt': 'quart',
    'pint': 'pint',
    'pt': 'pint',
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SINGLE-ROW EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_fields_from_text(catalog_text: str) -> dict:
    """
    Applies all regex patterns to one catalog_content string.

    Args:
        catalog_text (str): raw catalog_content value from CSV

    Returns:
        dict with keys: item_name, item_value, item_unit, category,
                        pack_count, ipq, regex_success
        regex_success=False means this row needs LLM imputation.

    INTERVIEW EXPLANATION:
        "We extract 6 structured signals from raw text using regex.
         If any critical field (item_name or quantity) is missing,
         we flag the row for LLM imputation. This keeps 80%+ of rows
         out of the expensive LLM path."
    """

    # Default result — all None until we find something
    result = {
        'item_name': None,
        'item_value': None,
        'item_unit': None,
        'category': None,
        'pack_count': None,
        'ipq': None,
        'regex_success': False  # will set True if we find key fields
    }

    # Safety check — skip NaN or empty strings
    if not isinstance(catalog_text, str) or catalog_text.strip() == '':
        return result

    # ── Extract item name ──────────────────────────────────────────────────
    name_match = ITEM_NAME_PATTERN.search(catalog_text)
    if name_match:
        # .group(1) → first capture group = the actual name
        # .strip() → remove leading/trailing whitespace
        result['item_name'] = name_match.group(1).strip()

    # ── Extract quantity + unit ────────────────────────────────────────────
    # .findall() returns ALL matches (list of tuples)
    # We take the FIRST one — usually the primary package size
    qty_matches = QUANTITY_UNIT_PATTERN.findall(catalog_text)
    if qty_matches:
        raw_value, raw_unit = qty_matches[0]
        result['item_value'] = float(raw_value)

        # Normalize unit (e.g., "fl. oz" → "fl_oz")
        normalized_unit = UNIT_NORMALIZATION.get(
            raw_unit.lower().strip(),
            raw_unit.lower().strip()  # fallback: keep as-is
        )
        result['item_unit'] = normalized_unit

    # ── Extract pack count ─────────────────────────────────────────────────
    pack_match = PACK_COUNT_PATTERN.search(catalog_text)
    if pack_match:
        result['pack_count'] = int(pack_match.group(1))

    # ── Extract category ──────────────────────────────────────────────────
    cat_match = CATEGORY_PATTERN.search(catalog_text)
    if cat_match:
        result['category'] = cat_match.group(1).strip()

    # ── Extract Item Package Quantity (IPQ) ───────────────────────────────
    ipq_match = IPQ_PATTERN.search(catalog_text)
    if ipq_match:
        result['ipq'] = int(ipq_match.group(1))

    # ── Mark success ──────────────────────────────────────────────────────
    # We consider regex "successful" if we got at least item_name
    # OR a quantity+unit pair — these are the most price-relevant fields
    if result['item_name'] is not None or result['item_value'] is not None:
        result['regex_success'] = True

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: BUILD STRUCTURED PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_deberta_prompt(row: pd.Series) -> str:
    """
    Converts a row's extracted fields into the structured input string
    that DeBERTa will be fine-tuned on.

    FORMAT (from README):
        "Category: [cat] [SEP] Description: [desc] [SEP] Amount: [val] [unit]"

    WHY THIS FORMAT:
        - Structured prompts help the transformer focus on key signals
        - [SEP] tokens create soft boundaries between field types
        - DeBERTa's tokenizer handles [SEP] natively
        - Giving explicit Amount: signal helps the model learn that
          quantity × price_per_unit ≈ total_price

    INTERVIEW EXPLANATION:
        "We structure the input so the model sees the category,
         a cleaned description, and the extracted quantity in a
         consistent format — this reduces noise vs raw catalog text."
    """

    # Build each part, using 'unknown' as fallback for missing fields
    # category_part = f"Category: {row['category'] or 'unknown'}"
    category_val = row['category']
    if pd.isna(category_val) or category_val is None:
         category_part = "Category: grocery"   # safe default for Amazon food catalog
    else:
         category_part = f"Category: {category_val}"

    # Use item_name as the core description; fall back to first 200 chars of raw text
    description_part = f"Description: {row['item_name'] or str(row['catalog_content'])[:200]}"

    # Only add amount if we have both value and unit
    if row['item_value'] is not None and row['item_unit'] is not None:
        amount_part = f"Amount: {row['item_value']} {row['item_unit']}"
    elif row['pack_count'] is not None:
        amount_part = f"Amount: pack of {row['pack_count']}"
    else:
        amount_part = "Amount: unknown"

    # Join with [SEP] separator token
    prompt = f"{category_part} [SEP] {description_part} [SEP] {amount_part}"
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_regex_extraction(input_path: str, output_path: str, failures_path: str):
    """
    Runs the full regex extraction pipeline on the entire dataset.

    Args:
        input_path    : path to train.csv or test.csv
        output_path   : where to save augmented CSV
        failures_path : where to save rows that need LLM imputation
    """

    print(f"\n{'='*60}")
    print(f"  REGEX EXTRACTION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Failures: {failures_path}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()}\n")

    # ── Apply extraction to every row ──────────────────────────────────────
    # .apply() runs extract_fields_from_text on each catalog_content value
    # pd.DataFrame(list(...)) converts list of dicts → DataFrame
    print("  Running regex extraction...")
    extracted = df['catalog_content'].apply(extract_fields_from_text)
    extracted_df = pd.DataFrame(extracted.tolist())

    # ── Merge extracted columns back into original df ──────────────────────
    df = pd.concat([df, extracted_df], axis=1)

    # ── Build structured DeBERTa prompt for each row ───────────────────────
    print("  Building DeBERTa input prompts...")
    df['model_input'] = df.apply(build_deberta_prompt, axis=1)

    # ── Compute extraction statistics ──────────────────────────────────────
    total = len(df)
    success = df['regex_success'].sum()
    failures = total - success
    success_pct = (success / total) * 100

    print(f"\n  ── Extraction Results ──────────────────────────────")
    print(f"  Total rows        : {total:,}")
    print(f"  Regex success     : {success:,}  ({success_pct:.1f}%)")
    print(f"  Needs LLM fill    : {failures:,}  ({100-success_pct:.1f}%)")
    print(f"  item_name found   : {df['item_name'].notna().sum():,}")
    print(f"  item_value found  : {df['item_value'].notna().sum():,}")
    print(f"  item_unit found   : {df['item_unit'].notna().sum():,}")
    print(f"  category found    : {df['category'].notna().sum():,}")
    print(f"  pack_count found  : {df['pack_count'].notna().sum():,}")
    print(f"  ────────────────────────────────────────────────────\n")

    # ── Show a few example prompts ─────────────────────────────────────────
    print("  Sample model_input prompts:")
    for i, row in df.head(3).iterrows():
        print(f"  [{i}] {row['model_input'][:120]}...")
    print()

    # ── Save augmented full dataset ────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved augmented dataset → {output_path}")

    # ── Save failure rows separately for LLM imputation ───────────────────
    failure_df = df[~df['regex_success']].copy()
    failure_df.to_csv(failures_path, index=False)
    print(f"  Saved {len(failure_df):,} failure rows → {failures_path}")
    print(f"\n{'='*60}\n")

    return df, failure_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Process training data
    train_df, train_failures = run_regex_extraction(
        input_path="data/train.csv",
        output_path="data/train_augmented.csv",
        failures_path="data/train_regex_failures.csv"
    )

    # Process test data the same way
    test_df, test_failures = run_regex_extraction(
        input_path="data/test.csv",
        output_path="data/test_augmented.csv",
        failures_path="data/test_regex_failures.csv"
    )

    print("✅ Regex extraction complete.")
    print("   Next step: python augmentation/llm_imputer.py")