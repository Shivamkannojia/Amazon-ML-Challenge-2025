"""
training/dataset.py
====================
WHAT THIS FILE DOES:
    Defines a PyTorch Dataset that:
      1. Loads the augmented CSV (output of regex_extractor.py)
      2. Rebuilds the structured prompt (fixing the nan category issue)
      3. Tokenizes text using DeBERTa's tokenizer
      4. Applies log1p transform to price targets
      5. Returns tensors ready for the DataLoader

KEY CONCEPTS EXPLAINED:
    - PyTorch Dataset: a class that wraps your data and tells PyTorch
      how to fetch one sample at a time (via __getitem__)
    - Tokenization: converts text → token IDs that the model understands
    - log1p transform: log1p(x) = log(x+1), stabilizes right-skewed prices
      Inverse at inference: expm1(x) = exp(x)-1
    - Padding/Truncation: all sequences must be same length for batching

WHY log1p AND NOT log?
    - Some prices could be 0 (though unlikely) → log(0) = -inf → crash
    - log1p(0) = 0 → safe
    - log1p is monotonic → ranking preserved
    - At inference: price = expm1(model_output) = exp(output) - 1
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PROMPT BUILDER (fixed — no more "Category: nan")
# ─────────────────────────────────────────────────────────────────────────────
# dataset.py build_prompt :
def build_prompt(row):
    parts = []
    category = row.get('product_category')
    if pd.notna(category):
        parts.append(f"Category: {str(category).strip()}")
    
    description = row.get('description')
    if pd.notna(description) and str(description).strip():
        parts.append(f"description: {str(description).strip()[:300]}")
    
    value = row.get('value')
    unit  = row.get('unit')
    if pd.notna(value) and pd.notna(unit):
        unit_str = str(unit).strip()
        if not (unit_str.lower() == 'count' and float(value) == 1.0):
            parts.append(f"Amount: {value} {unit_str.capitalize()}")
    
    return " [SEP] ".join(parts)

# // OLDER VERSIONS I TRIED
# def build_prompt(row: pd.Series) -> str:
#     """
#     v2.1 prompt — combines v1 quantity + v2 description
    
#     Key insight: v2 Value: field sometimes stores total weight 
#     (8oz × 4 pack = 32oz) which corrupts price_per_unit.
#     v1 regex extracted the unit size (8oz) more reliably.
#     So we use v1 quantity fields + v2 description field.
#     """
#     parts = []

#     # ── Category: use product_category (v2) ──────────────────────────────
#     category = row.get('product_category')
#     if pd.notna(category) and str(category).strip():
#         parts.append(f"Category: {str(category).strip()}")
#     else:
#         raw = str(row.get('catalog_content', ''))
#         parts.append(f"Category: {raw[:150]}")

#     # ── Description: bullet points (v2 addition) ──────────────────────────
#     description = row.get('description')
#     if pd.notna(description) and str(description).strip():
#         parts.append(f"description: {str(description).strip()[:200]}")

#     # ── Amount: use v1 item_value + item_unit (cleaner) ──────────────────
#     item_value = row.get('item_value')
#     item_unit  = row.get('item_unit')
#     if pd.notna(item_value) and pd.notna(item_unit):
#         parts.append(f"Amount: {item_value} {str(item_unit).capitalize()}")


#     # ── Pack count (v1) ───────────────────────────────────────────────────
#     pack_count = row.get('pack_count')
#     if pd.notna(pack_count):
#         parts.append(f"Pack: {int(pack_count)}")

#     return " [SEP] ".join(parts)
# def build_prompt(row: pd.Series) -> str:
#     """
#     v2 prompt builder 
    
#     FORMAT: "Category: [name] [SEP] description: [desc] [SEP] Amount: [val] [unit]"
    
#     Changes from v1:
#         - Uses product_category instead of item_name
#         - Includes description (bullet points) — was completely missing before
#         - Uses explicit value/unit fields instead of regex-extracted ones
#         - Skips Amount if it's just the default "1 Count" imputed value
#     """
#     parts = []

#     # ── Category (item name) ──────────────────────────────────────────────
#     category = row.get('product_category')
#     if pd.notna(category) and str(category).strip():
#         parts.append(f"Category: {str(category).strip()}")
#     else:
#         raw = str(row.get('catalog_content', ''))
#         parts.append(f"Category: {raw[:150]}")

#     # ── Description (bullet points) — KEY addition vs v1 ─────────────────
#     description = row.get('description')
#     if pd.notna(description) and str(description).strip():
#         desc_text = str(description).strip()[:200]
#         parts.append(f"description: {desc_text}")

#     # ── Amount (explicit value + unit) ────────────────────────────────────
#     value = row.get('value')
#     unit  = row.get('unit')
#     if pd.notna(value) and pd.notna(unit):
#         unit_str = str(unit).strip()
#         # Skip default imputed value (1 Count) — adds noise
#         skip = (unit_str.lower() == 'count' and float(value) == 1.0)
#         if not skip:
#             parts.append(f"Amount: {value} {unit_str.capitalize()}")

#     return " [SEP] ".join(parts)
# // v1
# def build_prompt(row: pd.Series) -> str: 
#     """
#     Builds the structured input string for DeBERTa.

#     FORMAT:
#         "Item: [name] [SEP] Amount: [value] [unit] [SEP] Pack: [count]"

#     DESIGN DECISIONS:
#         - Dropped 'Category' — only 11/75000 rows had it, adds no signal
#         - 'Item' is the strongest price signal — brand + product name
#         - 'Amount' (oz, kg, count) directly affects price
#         - 'Pack' (pack of 2, 3...) multiplies the base price
#         - [SEP] gives DeBERTa explicit field boundaries to attend to

#     FALLBACK STRATEGY:
#         If item_name is missing (7 rows), use first 200 chars of raw text.
#         If amount is missing (10,319 rows), omit the Amount field entirely
#         rather than writing "Amount: unknown" which adds noise.

#     Args:
#         row: one row of the augmented DataFrame

#     Returns:
#         str: structured prompt string
#     """

#     parts = []

#     # ── Item name ─────────────────────────────────────────────────────────
#     item_name = row.get('item_name')
#     if pd.notna(item_name) and str(item_name).strip():
#         parts.append(f"Item: {str(item_name).strip()}")
#     else:
#         # Fallback: first 200 chars of raw catalog text
#         raw = str(row.get('catalog_content', ''))
#         parts.append(f"Item: {raw[:200]}")

#     # ── Amount (value + unit) ─────────────────────────────────────────────
#     item_value = row.get('item_value')
#     item_unit = row.get('item_unit')

#     if pd.notna(item_value) and pd.notna(item_unit):
#         parts.append(f"Amount: {item_value} {item_unit}")

#     # ── Pack count ────────────────────────────────────────────────────────
#     pack_count = row.get('pack_count')
#     if pd.notna(pack_count):
#         parts.append(f"Pack: {int(pack_count)}")

#     # ── IPQ (Item Package Quantity) ───────────────────────────────────────
#     ipq = row.get('ipq')
#     if pd.notna(ipq):
#         parts.append(f"IPQ: {int(ipq)}")

#     # Join all parts with [SEP] token
#     return " [SEP] ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PYTORCH DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AmazonPriceDataset(Dataset):
    """
    PyTorch Dataset for Amazon product price prediction.

    HOW PYTORCH DATASET WORKS:
        PyTorch's DataLoader calls __len__ to know total size,
        and __getitem__(idx) to fetch one sample at a time.
        DataLoader handles batching, shuffling, and parallel loading.

    TOKENIZATION:
        AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
        converts text → {'input_ids', 'attention_mask', 'token_type_ids'}

        input_ids     : integer IDs for each token
        attention_mask: 1 for real tokens, 0 for padding tokens
                        tells model which positions to attend to

        max_length=160: from README hyperparams — long enough for most
                        product descriptions, short enough for speed

    Args:
        df          : augmented DataFrame with item_name, item_value etc.
        tokenizer   : HuggingFace tokenizer instance
        max_length  : max token length (default 160, from README)
        is_test     : if True, no price column → skip target loading
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 160,
        is_test: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # ── Build all prompts upfront (faster than doing it in __getitem__)
        print(f"  Building {len(df):,} prompts...")
        self.prompts = [build_prompt(row) for _, row in df.iterrows()]

        # ── Apply log1p to prices (training only) ─────────────────────────
        if not is_test:
            # log1p(price): transforms right-skewed distribution
            # [0.13, 4.89, 13.12, 2796] → roughly normal after log1p
            # This is CRITICAL for Huber loss to work well
            self.targets = np.log1p(df['price'].values).astype(np.float32)
            print(f"  Price stats after log1p transform:")
            print(f"    min={self.targets.min():.3f}, "
                  f"max={self.targets.max():.3f}, "
                  f"mean={self.targets.mean():.3f}, "
                  f"std={self.targets.std():.3f}")
        else:
            self.targets = None

        print(f"  Dataset ready: {len(self.prompts):,} samples\n")

    def __len__(self) -> int:
        """Returns total number of samples — required by DataLoader."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Fetches one sample by index.

        WHAT TOKENIZER RETURNS:
            {
              'input_ids':      tensor([101, 12453, 2003, ...]),  # token IDs
              'attention_mask': tensor([1, 1, 1, ..., 0, 0]),     # 1=real, 0=pad
            }

        We add 'labels' (log1p price) for training.

        Returns:
            dict of tensors — one per feature the model needs
        """
        prompt = self.prompts[idx]

        # Tokenize: padding='max_length' pads short sequences
        # truncation=True cuts sequences longer than max_length
        # return_tensors='pt' → PyTorch tensors directly
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # .squeeze(0): tokenizer adds batch dim → remove it
        # shape: (1, max_length) → (max_length,)
        sample = {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        # Add label only during training
        if self.targets is not None:
            sample['labels'] = torch.tensor(
                self.targets[idx], dtype=torch.float32
            )

        return sample


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    train_path: str = "data/train_augmented_v2.csv",
    tokenizer_name: str = "microsoft/deberta-v3-large",
    max_length: int = 160,
    batch_size: int = 16,
    val_split: float = 0.1,
    random_seed: int = 42
):
    """
    Loads data, splits train/val, builds DataLoaders.

    TRAIN/VAL SPLIT:
        - 90% train (67,500 rows), 10% val (7,500 rows)
        - random_state=42 for reproducibility
        - stratify not used (regression target, not classification)

    DATALOADER SETTINGS:
        - shuffle=True for train → prevents model memorizing order
        - shuffle=False for val → consistent evaluation
        - num_workers=2 → parallel data loading (faster on multi-core)
        - pin_memory=True → faster CPU→GPU data transfer

    Args:
        train_path    : path to augmented training CSV
        tokenizer_name: HuggingFace model name
        max_length    : max token sequence length
        batch_size    : samples per batch (16 from README)
        val_split     : fraction for validation (0.1 = 10%)
        random_seed   : for reproducibility

    Returns:
        train_loader, val_loader, tokenizer
    """

    print(f"\n{'='*60}")
    print(f"  CREATING DATALOADERS")
    print(f"{'='*60}")
    print(f"  Model     : {tokenizer_name}")
    print(f"  Max length: {max_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Val split : {val_split*100:.0f}%\n")

    # ── Load tokenizer ─────────────────────────────────────────────────────
    # Downloads vocab + config from HuggingFace Hub on first run
    # Cached locally after that (~350MB for deberta-v3-large)
    print(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # ── Load augmented training data ───────────────────────────────────────
    print(f"\n  Loading {train_path}...")
    df = pd.read_csv(train_path)
    print(f"  Loaded {len(df):,} rows")

    # ── Train / Validation split ───────────────────────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=random_seed
    )
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val  : {len(val_df):,} rows\n")

    # ── Create Dataset objects ─────────────────────────────────────────────
    print("  Building training dataset...")
    train_dataset = AmazonPriceDataset(
        train_df, tokenizer, max_length=max_length, is_test=False
    )

    print("  Building validation dataset...")
    val_dataset = AmazonPriceDataset(
        val_df, tokenizer, max_length=max_length, is_test=False
    )

    # ── Wrap in DataLoaders ────────────────────────────────────────────────
    # num_workers=0: no parallel CPU workers → less heat on laptop
    #   (multiprocessing workers spin up extra CPU cores → more heat)
    # pin_memory=False: with num_workers=0 pin_memory has no benefit
    # persistent_workers only valid when num_workers > 0 → omit
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle each epoch → better generalization
        num_workers=0,          # 0 = main process only → cooler on laptop
        pin_memory=False        # no benefit without workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # larger batch for val (no gradients)
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print(f"\n{'='*60}\n")

    return train_loader, val_loader, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test to verify dataset works before full training.
    Loads 100 rows, builds dataset, checks one batch.
    """
    import pandas as pd
    from transformers import AutoTokenizer

    print("Running dataset sanity check on 100 rows...")

    # Use a small fast tokenizer for the check
    TOKENIZER_NAME = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load first 100 rows only
    df = pd.read_csv("data/train_augmented.csv", nrows=100)

    # Build dataset
    dataset = AmazonPriceDataset(df, tokenizer, max_length=160, is_test=False)

    # Check one sample
    sample = dataset[0]
    print("\n  One sample keys:", list(sample.keys()))
    print(f"  input_ids shape : {sample['input_ids'].shape}")
    print(f"  attention_mask  : {sample['attention_mask'].shape}")
    print(f"  label (log1p)   : {sample['labels'].item():.4f}")

    # Decode back to text to verify prompt looks right
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"\n  Decoded prompt (first 200 chars):")
    print(f"  {decoded[:200]}")

    # Check a batch
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"\n  Batch input_ids shape : {batch['input_ids'].shape}")   # (4, 160)
    print(f"  Batch labels          : {batch['labels']}")

    print("\n✅ Dataset sanity check passed.")
    print("   Next step: python training/model.py")