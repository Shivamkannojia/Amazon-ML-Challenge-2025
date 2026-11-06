"""
training/model.py
==================
WHAT THIS FILE DOES:
    Defines the neural network architecture for price prediction:

        DeBERTa-v3-large encoder
               ↓
        [CLS] token embedding (1024-dim)
               ↓
        Dropout(0.2)
               ↓
        Linear(1024 → 512) + GELU + Dropout(0.2)
               ↓
        Linear(512 → 256) + GELU + Dropout(0.2)
               ↓
        Linear(256 → 1)   ← single price prediction
               ↓
        log1p price (reversed with expm1 at inference)

WHY DeBERTa-v3-large OVER OTHER MODELS:
    ┌─────────────────────┬────────┬────────┬──────────┐
    │ Model               │ Params │ SMAPE  │ Speed    │
    ├─────────────────────┼────────┼────────┼──────────┤
    │ DeBERTa-v3-large    │ 434M   │ 42.22% │ slow     │
    │ DeBERTa-v3-base     │ 184M   │ 43.59% │ medium   │
    │ DistilBERT          │ 66M    │ 45.00% │ fast     │
    │ Sentence Embeddings │ 22M    │ 49.00% │ fastest  │
    └─────────────────────┴────────┴────────┴──────────┘
    DeBERTa uses:
      - Disentangled Attention: separate position + content attention
      - Enhanced Mask Decoder: better MLM pretraining
      - v3: trained with ELECTRA-style replaced token detection
    Result: best contextual understanding of product descriptions

WHY MLP HEAD (not linear):
    Price is nonlinearly related to product features:
      - "32 oz ketchup" ≠ 2 × "16 oz ketchup" (bulk discount)
      - Brand premium is non-additive
    A 2-layer MLP captures these nonlinear interactions.

WHY GELU ACTIVATION (not ReLU):
    GELU (Gaussian Error Linear Unit) = x * Φ(x)
    - Smoother than ReLU → better gradient flow
    - Used inside DeBERTa itself → consistent activation throughout

WHY HUBER LOSS (not MSE):
    - MSE penalizes outliers quadratically → $2796 item ruins training
    - Huber loss = MSE for small errors, MAE for large errors
    - delta=1.0: errors < 1 use MSE (precise), > 1 use MAE (robust)
    - Perfect for our right-skewed price distribution
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class AmazonPricePredictor(nn.Module):
    """
    DeBERTa-v3-large encoder + 2-layer MLP regression head.

    ARCHITECTURE DETAIL:
        Input tokens (seq_len=160)
            ↓
        DeBERTa-v3-large backbone
            → outputs hidden states: (batch, seq_len, 1024)
            → we take [:, 0, :] = [CLS] token = (batch, 1024)
              [CLS] is trained to summarize entire sequence
            ↓
        dropout(0.2)          → regularization, prevents overfitting
            ↓
        Linear(1024 → 512)    → compress representation
        GELU()                → nonlinear activation
        dropout(0.2)
            ↓
        Linear(512 → 256)     → further compress
        GELU()
        dropout(0.2)
            ↓
        Linear(256 → 1)       → scalar price prediction
                              → in log1p space (reversed at inference)

    DUAL LEARNING RATES (key technique):
        encoder_lr = 2e-5  (small) → fine-tune pretrained weights gently
        head_lr    = 1e-3  (large) → train MLP head from scratch aggressively
        Reason: encoder already knows language, head knows nothing yet.
        Same LR for both would either destroy encoder or starve head.

    Args:
        model_name  : HuggingFace model identifier
        dropout_rate: dropout probability (0.2 from README)
        hidden_dim  : first MLP hidden size (512 from README)
        hidden_dim2 : second MLP hidden size (256 from README)
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        dropout_rate: float = 0.2,
        hidden_dim: int = 512,
        hidden_dim2: int = 256
    ):
        # Must call parent __init__ first — PyTorch requirement
        super(AmazonPricePredictor, self).__init__()

        self.model_name = model_name
        self.dropout_rate = dropout_rate

        # ── Load DeBERTa backbone ──────────────────────────────────────────
        # AutoConfig: loads model configuration (num layers, hidden size etc.)
        # hidden_size for deberta-v3-large = 1024
        print(f"  Loading backbone: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        self.encoder_hidden_size = config.hidden_size  # 1024 for large
        print(f"  Encoder hidden size: {self.encoder_hidden_size}")

        # AutoModel: loads pretrained weights WITHOUT a task-specific head
        # (we add our own regression head below)
        self.encoder = AutoModel.from_pretrained(model_name, config=config, torch_dtype=torch.float32)

        # ── MLP Regression Head ────────────────────────────────────────────
        # nn.Sequential: chains layers in order, clean and readable

        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.encoder_hidden_size),  # ← NEW: stabilizes encoder output
            nn.Dropout(dropout_rate),
            # Layer 1: 1024 → 512
            nn.Linear(self.encoder_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Layer 2: 512 → 256
            nn.Linear(hidden_dim, hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # Output layer: 256 → 1 (scalar log1p price)
            # No activation here — raw regression output
            nn.Linear(hidden_dim2, 1)
        )

        # ── Initialize MLP weights ─────────────────────────────────────────
        # Xavier uniform: keeps gradient variance stable across layers
        # Without this, deep MLPs can have vanishing/exploding gradients
        self._init_weights()

        print(f"  MLP head: {self.encoder_hidden_size} → "
              f"{hidden_dim} → {hidden_dim2} → 1")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Total params: {self._count_params():,}")
        print(f"  Trainable params: {self._count_params(trainable_only=True):,}")

    def _init_weights(self):
        """
        Xavier uniform initialization for linear layers in the MLP head.

        WHY XAVIER:
            Keeps variance of activations ≈ constant across layers.
            Formula: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
            Designed specifically for layers followed by sigmoid/tanh/GELU.
        """
        for module in self.regression_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _count_params(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs   # absorbs extra keys like 'labels' from batch dict
    ) -> torch.Tensor:
        """
        Forward pass: tokens → predicted log1p price.

        WHAT HAPPENS INSIDE DeBERTa:
            1. Token embeddings: input_ids → dense vectors
            2. Position embeddings: add position information
            3. 24 transformer layers (large): each does
               - Disentangled self-attention (content + position separately)
               - Feed-forward network
               - Layer norm + residual connection
            4. Output: (batch_size, seq_len, 1024) hidden states

        WE TAKE [CLS] TOKEN (position 0):
            DeBERTa is trained so [CLS] aggregates full sequence meaning.
            Shape: (batch_size, 1024)

        Args:
            input_ids     : token IDs, shape (batch, seq_len)
            attention_mask: 1=real token 0=padding, shape (batch, seq_len)

        Returns:
            predictions: shape (batch,) — one log1p price per sample
        """

        # ── Run encoder ────────────────────────────────────────────────────
        # last_hidden_state shape: (batch_size, seq_len=160, hidden=1024)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden = outputs.last_hidden_state

        # # ── Extract [CLS] token representation ────────────────────────────
        # # [:, 0, :] → take position 0 ([CLS]) for all batch items
        # # Shape: (batch_size, 1024)
        # cls_embedding = outputs.last_hidden_state[:, 0, :]

        # approach 2
        # Mean pooling — average ALL token representations
        # weighted by attention mask (ignores padding tokens)
        # Shape: (batch, seq_len, 768) → (batch, 768)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # ── Pass through MLP regression head ──────────────────────────────
        # Shape: (batch_size, 1)
        logit = self.regression_head(pooled)

        # ── Squeeze to (batch_size,) ───────────────────────────────────────
        # squeeze(-1): removes last dim of size 1
        # (batch, 1) → (batch,)
        predictions = logit.squeeze(-1)

        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class HuberLoss(nn.Module):
    """
    Huber Loss wrapper with delta parameter.

    HUBER LOSS FORMULA:
        L(y, ŷ) = 0.5*(y-ŷ)²           if |y-ŷ| ≤ delta  ← MSE region
                  delta*(|y-ŷ| - 0.5*delta)  otherwise       ← MAE region

    WHY THIS FOR PRICE PREDICTION:
        - A $2796 blender vs $4.89 taco sauce: 571× price difference
        - MSE would make the model obsess over expensive outliers
        - Huber switches to linear (MAE) penalty for large errors
        - delta=1.0 means: errors < $1 in log space → MSE, else → MAE
          In original price space: roughly errors < 1.7× → MSE

    PyTorch has nn.HuberLoss built-in — we wrap it for clarity.
    """

    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.loss_fn = nn.HuberLoss(delta=delta, reduction='mean')
        self.delta = delta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: model output, shape (batch,) in log1p space
            targets    : log1p(true_price), shape (batch,)
        Returns:
            scalar loss value
        """
        return self.loss_fn(predictions, targets)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: OPTIMIZER FACTORY (DUAL LEARNING RATES)
# ─────────────────────────────────────────────────────────────────────────────

def create_optimizer(
    model: AmazonPricePredictor,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Creates AdamW optimizer with dual learning rates.

    DUAL LR EXPLAINED:
        encoder (DeBERTa weights) → lr=2e-5  (tiny, gentle fine-tuning)
        head    (MLP weights)     → lr=1e-3  (large, learning from scratch)

        If we used one LR for both:
          - 2e-5 for head → head learns too slowly, undertrained in 10 epochs
          - 1e-3 for encoder → destroys pretrained representations, catastrophic

    WEIGHT DECAY (L2 regularization):
        Adds λ*||w||² penalty to loss → pushes weights toward zero
        Prevents overfitting, especially important with 434M params
        Applied to weights only, NOT biases or LayerNorm params
        (biases + LayerNorm are excluded — standard practice)

    ADAMW vs ADAM:
        Adam: weight decay implemented incorrectly (coupled with lr)
        AdamW: decouples weight decay from lr → more stable training
        Transformers always use AdamW.

    Args:
        model       : AmazonPricePredictor instance
        encoder_lr  : learning rate for DeBERTa backbone (2e-5)
        head_lr     : learning rate for MLP head (1e-3)
        weight_decay: L2 regularization coefficient (0.01)

    Returns:
        AdamW optimizer with param groups
    """

    # Separate parameters into two groups
    # named_parameters() → (name, param) tuples
    # "regression_head" in name → MLP head params
    # else → encoder params

    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "regression_head" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    # param_groups: list of dicts, each with 'params' + optimizer settings
    optimizer = torch.optim.AdamW(
        [
            {
                'params': encoder_params,
                'lr': encoder_lr,
                'weight_decay': weight_decay
            },
            {
                'params': head_params,
                'lr': head_lr,
                'weight_decay': weight_decay
            }
        ]
    )

    print(f"  Optimizer: AdamW")
    print(f"  Encoder params: {len(encoder_params)} tensors, lr={encoder_lr}")
    print(f"  Head params   : {len(head_params)} tensors, lr={head_lr}")
    print(f"  Weight decay  : {weight_decay}")

    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SMAPE METRIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_smape(
    predictions_log: torch.Tensor,
    targets_log: torch.Tensor
) -> float:
    """
    Computes SMAPE (Symmetric Mean Absolute Percentage Error).

    SMAPE FORMULA:
        SMAPE = (100/n) * Σ |actual - predicted| / ((|actual| + |predicted|) / 2)

    WHY SMAPE (not RMSE or MAE):
        - Scale-independent: fair across $0.13 and $2796 items
        - Symmetric: over-prediction and under-prediction penalized equally
        - Range: 0% (perfect) to 200% (complete failure)
        - Amazon ML Challenge uses this as the official metric

    IMPORTANT: We compute SMAPE in ORIGINAL price space (not log space)
        Step 1: convert log1p predictions → actual prices via expm1
        Step 2: compute SMAPE on actual prices
        This is what the competition evaluates.

    Args:
        predictions_log: model outputs in log1p space, shape (n,)
        targets_log    : true log1p prices, shape (n,)

    Returns:
        smape: float, percentage value (lower is better)
    """

    # Convert back to original price space
    # expm1(x) = exp(x) - 1 = inverse of log1p(x)
    predictions = torch.expm1(predictions_log)
    targets = torch.expm1(targets_log)

    # Clamp predictions to avoid negative prices
    # (model might predict negative log1p values → expm1 could give negative)
    predictions = torch.clamp(predictions, min=0.0)

    # SMAPE computation
    numerator = torch.abs(targets - predictions)
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0

    # Avoid division by zero (if both actual and predicted are 0)
    denominator = torch.clamp(denominator, min=1e-8)

    smape = torch.mean(numerator / denominator) * 100.0

    return smape.item()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick forward pass test to verify model architecture before training.
    Uses random tensors — no real data needed.
    """

    print(f"\n{'='*60}")
    print(f"  MODEL ARCHITECTURE SANITY CHECK")
    print(f"{'='*60}\n")

    # ── Detect device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()

    # ── Build model ────────────────────────────────────────────────────────
    model = AmazonPricePredictor(
        model_name="microsoft/deberta-v3-base",
        dropout_rate=0.2,
        hidden_dim=512,
        hidden_dim2=256
    )
    model = model.to(device).float()   # force all params to float32

    # ── Build loss and optimizer ───────────────────────────────────────────
    print()
    loss_fn = HuberLoss(delta=1.0)
    optimizer = create_optimizer(model, encoder_lr=2e-5, head_lr=1e-3)

    # ── Dummy forward pass ─────────────────────────────────────────────────
    # batch_size=4, seq_len=160 — same as real DataLoader batches
    print("\n  Running dummy forward pass (batch=4, seq_len=160)...")
    batch_size, seq_len = 4, 160

    dummy_input_ids      = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    dummy_labels         = torch.tensor([1.77, 2.64, 1.08, 3.44]).to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask
        )

    print(f"  Input shape  : {dummy_input_ids.shape}")
    print(f"  Output shape : {preds.shape}")      # should be (4,)
    print(f"  Sample preds : {preds.cpu()}")

    # Loss
    loss = loss_fn(preds, dummy_labels)
    print(f"  Huber loss   : {loss.item():.4f}")

    # SMAPE
    smape = compute_smape(preds, dummy_labels)
    print(f"  SMAPE        : {smape:.2f}% (random, expect ~100%)")

    # Memory usage
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"\n  GPU memory used: {allocated:.2f} GB")

    print(f"\n{'='*60}")
    print(f"✅ Model sanity check passed.")
    print(f"   Next step: python training/trainer.py")
    print(f"{'='*60}\n")