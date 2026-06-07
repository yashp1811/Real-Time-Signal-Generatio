"""
Transformer-based (PatchTST-style) signal classifier.
Replaces the CNN-BiLSTM-Attention stack with a modern Transformer Encoder.

Architecture
------------
  Input  : (B, T=60, F)  -- normalised feature window

  Stage 1 -- Patch Embedding
    Splits 60 bars into 6 non-overlapping patches of 10 bars each.
    Each patch is projected to d_model=128 dimensions via a linear layer.
    Reduces sequence length from 60 -> 6, cutting attention cost by 100x.
    Output: (B, 6, d_model)

  Stage 2 -- [CLS] token prepend
    A learnable [CLS] token is prepended: (B, 7, d_model).
    The model learns to aggregate the full sequence into this single token.
    Output: (B, 7, d_model)

  Stage 3 -- Positional Encoding
    Fixed sinusoidal PE tells the model which patch is first, second, etc.
    Without this, Transformers are permutation-invariant (no sense of time).

  Stage 4 -- Transformer Encoder x 4 layers  (Pre-LN)
    Each layer = Multi-Head Self-Attention (8 heads) + Feed-Forward (dim=256).
    All 7 tokens attend to each other simultaneously -- no vanishing gradient.
    Pre-LN (norm_first=True) normalises before attention for stable training.
    Output: (B, 7, d_model)

  Stage 5 -- [CLS] pool
    Takes only the [CLS] token as the sequence summary.
    Output: (B, d_model=128)

  Stage 6 -- Classifier head
    LayerNorm -> GELU -> Dropout -> Linear(3)
    3 output logits: down (0), flat (1), up (2)

Why Transformer beats CNN-BiLSTM-Attention
------------------------------------------
  - LSTM has vanishing gradient beyond ~30 bars; Transformer attention spans
    the full 60-bar window uniformly with no decay.
  - Fully parallel computation -- faster GPU training than sequential LSTM.
  - PatchTST-style patching cuts sequence length by 10x (60->6 patches),
    reducing self-attention complexity from O(60^2) to O(6^2).
  - Pre-LN Transformer layers are more stable (no exploding gradients).
  - [CLS] pooling is more expressive than LSTM final hidden state.
  - Attention weights are interpretable: shows which time patches mattered.

Transfer learning mapping (equivalent to old CNN-BiLSTM)
---------------------------------------------------------
  OLD: freeze CNN, fine-tune LSTM + attention + head
  NEW: freeze patch_embed, fine-tune encoder + CLS + head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits (B, T, F) into non-overlapping patches, projects each to d_model.

    If T is not divisible by patch_len, left-padding is applied so that the
    most recent (rightmost) bars are never truncated -- they are most relevant
    for next-bar direction prediction.

    Example: T=60, patch_len=10  ->  6 patches of shape (10 * n_features,)
             each projected to d_model=128.
    """
    def __init__(self, n_features: int, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        # Linear projection: (patch_len * n_features) -> d_model
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        B, T, F = x.shape
        # Left-pad if T is not divisible by patch_len
        pad = (self.patch_len - T % self.patch_len) % self.patch_len
        if pad:
            x = F.pad(x, (0, 0, pad, 0))          # pad on left (time axis)
        # Reshape into patches: (B, n_patches, patch_len * F)
        x = x.reshape(B, -1, self.patch_len * F)
        return self.norm(self.proj(x))              # (B, n_patches, d_model)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Sine/cosine at different frequencies encode each position index uniquely.
    The encoding is additive so patch content and position information are
    both carried forward through all Transformer layers.
    """
    def __init__(self, d_model: int, max_patches: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_patches, d_model)
        pos = torch.arange(max_patches).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignalTransformer(nn.Module):
    """
    PatchTST-style Transformer encoder for directional signal classification.

    Drop-in replacement for the old SignalLSTM -- identical input/output:
      forward(x: Tensor(B, T, F)) -> logits: Tensor(B, 3)

    Parameters
    ----------
    n_features  : number of input features per bar (F)
    patch_len   : bars per patch (default 10; 60/10 = 6 patches)
    d_model     : embedding / attention dimension (default 128)
    n_heads     : attention heads -- d_model must be divisible (default 8)
    n_layers    : Transformer encoder layers (default 4)
    d_ff        : feed-forward hidden dimension (default 256)
    dropout     : dropout rate applied after attention, FFN, and in head
    n_classes   : output classes (3: down=0, flat=1, up=2)
    """
    def __init__(self,
                 n_features: int,
                 patch_len:  int   = 10,
                 d_model:    int   = 128,
                 n_heads:    int   = 8,
                 n_layers:   int   = 4,
                 d_ff:       int   = 256,
                 dropout:    float = 0.3,
                 n_classes:  int   = 3):
        super().__init__()

        # ── Stage 1: Patch embedding  (local feature extraction) ─────────────
        # Equivalent role to CNN front-end in old architecture.
        # Frozen during fine-tuning to preserve general local patterns.
        self.patch_embed = PatchEmbedding(n_features, patch_len, d_model)

        # ── Stage 2: Positional encoding ─────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── Stage 3: Learnable [CLS] token ───────────────────────────────────
        # Acts as a "global summary" that attends to all patch tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Stage 4: Transformer encoder  (Pre-LN for stable training) ───────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = "gelu",     # smoother than ReLU
            batch_first     = True,       # (B, S, E) convention
            norm_first      = True,       # Pre-LN: more stable than Post-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Stage 5: Classifier head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B = x.size(0)

        # 1. Embed patches: (B, T, F) -> (B, n_patches, d_model)
        x = self.patch_embed(x)

        # 2. Prepend [CLS] token: (B, n_patches+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # 3. Add positional encoding (applied after CLS prepend)
        x = self.pos_enc(x)

        # 4. Transformer encoder -- all tokens attend to all other tokens
        x = self.encoder(x)                # (B, n_patches+1, d_model)

        # 5. Use [CLS] token as the full-sequence summary
        cls_out = x[:, 0]                  # (B, d_model)

        # 6. Classify: down / flat / up
        return self.head(cls_out)          # (B, 3)
