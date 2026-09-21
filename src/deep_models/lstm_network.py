"""
Bidirectional LSTM with Attention Pooling for Transaction Sequences.
Maps input sequence [B, L, D] -> [B, 1] fraud logits.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SequenceAttentionPooling(nn.Module):
    """
    Computes query-free self-attention weights over sequence time steps,
    masking out padded positions before softmax.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, hidden_dim]
            mask: [B, L] bool tensor, True where padded / invalid
        Returns:
            pooled: [B, hidden_dim]
        """
        # Energy scores: [B, L, 1]
        scores = self.projection(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
            # Safe negative fill according to active dtype to prevent FP16 overflow / NaN
            min_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = scores.masked_fill(mask_expanded, min_val)

        weights = F.softmax(scores, dim=1)  # [B, L, 1]

        if mask is not None:
            # Explicitly zero out weights on masked positions to disconnect computational graph
            weights = weights.masked_fill(mask.unsqueeze(-1), 0.0)
            # Renormalize to ensure sum(weights) == 1.0 along sequence dimension
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        pooled = torch.sum(x * weights, dim=1)  # [B, hidden_dim]
        return pooled


class BiLSTMFraudModel(nn.Module):
    """
    Bidirectional LSTM with input projection, attention pooling, and classification head.
    [B, L, D] -> [B, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        head_hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 1. Input Linear Projection
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # 2. Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * self.num_directions

        # 3. Attention Pooling
        self.pooling = SequenceAttentionPooling(hidden_dim=lstm_out_dim)

        # 4. Classification Head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_seq: [B, L, D]
            padding_mask: [B, L] bool tensor where True indicates padding
        Returns:
            logits: [B, 1]
        """
        # Project D -> hidden_size
        x_proj = self.input_proj(x_seq)  # [B, L, hidden_size]

        # LSTM pass
        lstm_out, _ = self.lstm(x_proj)  # [B, L, hidden_size * num_directions]

        # Temporal attention aggregation
        pooled = self.pooling(lstm_out, mask=padding_mask)  # [B, lstm_out_dim]

        # Classification logits
        logits = self.head(pooled)  # [B, 1]
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
