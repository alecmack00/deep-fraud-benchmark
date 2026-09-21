"""
Transformer Encoder with Positional Embeddings and [CLS] Token.
Captures temporal transaction dependencies using pure self-attention.
"""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderFraudModel(nn.Module):
    """
    Transformer Encoder Sequence Model:
    1. Linear projection: D -> d_model
    2. Prepends learnable [CLS] token
    3. Adds sinusoidal positional embeddings
    4. 3 Pre-LN TransformerEncoderLayers (H=4, d_ff=512, GELU)
    5. Output [CLS] hidden state -> Linear(d_model, 1) logits
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 1. Feature Projection
        self.feature_proj = nn.Linear(input_dim, d_model)

        # 2. Learnable [CLS] classification token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3. Sinusoidal Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=100, dropout=dropout
        )

        # 4. Transformer Encoder Stack (Pre-LayerNorm architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Final LayerNorm before prediction head
        self.norm = nn.LayerNorm(d_model)

        # 5. Prediction Head on [CLS] representation
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        x_seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: [B, L, D]
            padding_mask: [B, L] bool tensor where True indicates padding
        Returns:
            logits: [B, 1]
        """
        B, _L, _ = x_seq.shape

        # 1. Linear Projection: [B, L, D] -> [B, L, d_model]
        h = self.feature_proj(x_seq)

        # 2. Prepend [CLS] token: [B, 1, d_model] -> [B, L+1, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)

        # 3. Add Positional Encoding
        h = self.pos_encoder(h)

        # 4. Adjust padding mask for the [CLS] token at position 0 (never padded)
        if padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
            extended_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, L+1]
            # Zero out padded positions to eliminate residual positional embeddings
            h = h.masked_fill(extended_mask.unsqueeze(-1), 0.0)
        else:
            extended_mask = None

        # 5. Pass through Transformer Encoder (src_key_padding_mask blocks attention to pads)
        encoded = self.encoder(h, src_key_padding_mask=extended_mask)
        encoded = self.norm(encoded)

        # 6. Extract [CLS] token representation at index 0
        cls_rep = encoded[:, 0, :]  # [B, d_model]

        # 7. Final Linear Classification Logits
        logits = self.head(cls_rep)  # [B, 1]
        return logits

    def predict_proba(
        self,
        x_seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns calibrated probabilities in [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(x_seq, padding_mask)
            return torch.sigmoid(logits)
