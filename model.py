"""ClockPatzerModel architecture (vendored from ChessMimic / 1e4_ai).

This module is derived from ``Training/ClockTrainer.py`` in the upstream
project https://github.com/thomasj02/1e4_ai and is distributed under the
PolyForm Noncommercial License 1.0.0 (see LICENSE and NOTICE).

Required Notice: Copyright 2026 Thomas Johnson (https://github.com/thomasj02/1e4_ai)

Only the inference-relevant architecture is retained. The upstream classes
subclass ``lightning.pytorch.LightningModule``; here they subclass
``torch.nn.Module`` so the model can be loaded and run without the PyTorch
Lightning training dependency. The layer names, shapes, and forward pass are
preserved exactly so the trained checkpoint loads unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.split1_linear = nn.Linear(input_size, hidden_size, bias=False)
        self.split2_linear = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = nn.SiLU()
        self.join_linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        split_1 = self.split1_linear(x)
        split_2 = self.split2_linear(x)
        x = self.activation(split_1) * split_2
        x = self.join_linear(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.self_attention(query=x, key=x, value=x, need_weights=False)[0]
        return x


class ClockPatzerModel(nn.Module):
    """Clock prediction model with discrete (bucket) output."""

    def __init__(
        self,
        recent_moves_sequence_length,
        recent_moves_vocab_size,
        board_sequence_length,
        board_input_vocab_size,
        embedding_dim,
        widening_factor,
        num_layers,
        num_heads,
        n_buckets,
    ):
        super().__init__()

        self.recent_moves_sequence_length = recent_moves_sequence_length
        self.recent_moves_vocab_size = recent_moves_vocab_size
        self.board_sequence_length = board_sequence_length
        self.input_vocab_size = board_input_vocab_size
        self.embedding_dim = embedding_dim
        self.widening_factor = widening_factor
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_buckets = n_buckets

        # Embeddings
        self.board_embedding = nn.Embedding(board_input_vocab_size, embedding_dim)
        self.rating_embedding = nn.Linear(1, embedding_dim)

        # Enhanced clock embedding for 3D input
        self.clock_time_embedding = nn.Linear(3, embedding_dim)

        self.recent_moves_embedding = nn.Embedding(recent_moves_vocab_size, embedding_dim)
        self.learned_positional_encoding = nn.Parameter(
            torch.randn(
                recent_moves_sequence_length + board_sequence_length + 2,
                embedding_dim,
            )
        )  # +2 for rating and clock

        # Transformer blocks
        hidden_size = embedding_dim * widening_factor
        self.mlp_blocks = nn.ModuleList(
            [MlpBlock(embedding_dim, hidden_size, embedding_dim) for _ in range(num_layers)]
        )
        self._attention_blocks = nn.ModuleList(
            [AttentionBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Single classification head
        self.time_classifier = nn.Linear(embedding_dim, n_buckets)

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype (replaces the LightningModule.dtype used upstream)."""
        return self.board_embedding.weight.dtype

    def forward(self, input_ids, attention_mask, scaled_rating, clock_features):
        # Split input_ids into recent moves and board
        recent_moves = input_ids[:, : self.recent_moves_sequence_length]
        board = input_ids[:, self.recent_moves_sequence_length:]

        recent_moves = self.recent_moves_embedding(recent_moves)
        board = self.board_embedding(board)

        # Process rating input
        scaled_rating = scaled_rating.to(dtype=self.dtype)
        embedded_rating = self.rating_embedding(scaled_rating.unsqueeze(1))
        embedded_rating = embedded_rating.unsqueeze(1)

        # Process 3D clock features
        clock_features = clock_features.to(dtype=self.dtype)
        embedded_clock_time = self.clock_time_embedding(clock_features)
        embedded_clock_time = embedded_clock_time.unsqueeze(1)

        # Concatenate all inputs: recent moves, rating, clock time, and board
        x = torch.cat([recent_moves, embedded_rating, embedded_clock_time, board], dim=1)

        x = x + self.learned_positional_encoding
        for mlp_block, attention_block in zip(self.mlp_blocks, self._attention_blocks):
            x = x + attention_block(x)
            x = x + mlp_block(x)

        x = self.layer_norm(x)
        x = x[:, -1, :]  # Only take the last token (CLS token)

        logits = self.time_classifier(x)
        return logits
