"""Clock-model inference pipeline.

Replicates the upstream ``backend/clock_inference.ClockPredictor`` from the
ChessMimic project (https://github.com/thomasj02/1e4_ai), with the C++
``chessmimic_core`` tokenizer swapped for the pure-Python ``tokenizer`` module
in this service. The prediction math (feature scaling, model call, softmax,
bucket sampling, empirical time sampling) is preserved exactly.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from clock_bucket_utils import load_bucket_info
from model import ClockPatzerModel
from tokenizer import (
    INPUT_VOCAB_SIZE,
    NUM_ACTIONS,
    RECENT_MOVES_LENGTH,
    SEQUENCE_LENGTH,
    prepare_recent_moves_tokens,
    tokenize,
)


class ClockPredictor:
    """Predicts human thinking time (seconds) for a chess move."""

    def __init__(self, model_path, scalers_path, bucket_info_path):
        model_path = Path(model_path)
        scalers_path = Path(scalers_path)
        bucket_info_path = Path(bucket_info_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers file not found: {scalers_path}")
        if not bucket_info_path.exists():
            raise FileNotFoundError(f"Bucket info file not found: {bucket_info_path}")

        # This service always runs on CPU (its own instance). The env var is
        # honoured for parity with upstream but defaults to CPU.
        force_cpu = os.getenv("CHESSMIMIC_FORCE_CPU", "true").lower() == "true"
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load scalers.
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)

        self.rating_mean = float(scalers["rating"]["mean"])
        self.rating_std = float(scalers["rating"]["std"])
        self.log_player_clock_mean = float(scalers["log_player_clock"]["mean"])
        self.log_player_clock_std = float(scalers["log_player_clock"]["std"])
        self.log_opponent_clock_mean = float(scalers["log_opponent_clock"]["mean"])
        self.log_opponent_clock_std = float(scalers["log_opponent_clock"]["std"])
        self.log_increment_mean = float(scalers["log_increment"]["mean"])
        self.log_increment_std = float(scalers["log_increment"]["std"])

        # Load bucket info.
        self.bucket_info = load_bucket_info(str(bucket_info_path))

        # Build the model with the training-time hyperparameters.
        self.model = ClockPatzerModel(
            recent_moves_sequence_length=RECENT_MOVES_LENGTH,
            recent_moves_vocab_size=NUM_ACTIONS,
            board_sequence_length=SEQUENCE_LENGTH,
            board_input_vocab_size=INPUT_VOCAB_SIZE,
            embedding_dim=256,
            widening_factor=4,
            num_layers=8,
            num_heads=8,
            n_buckets=self.bucket_info.n_buckets,
        )

        # Load checkpoint and strip training prefixes.
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[6:]
            if k.startswith("_orig_mod."):
                k = k[10:]
            new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_bucket(
        self,
        fen: str,
        recent_moves: List[str],
        rating: int,
        player_clock: float,
        opponent_clock: float,
        increment: float,
    ) -> np.ndarray:
        """Return the bucket probability distribution for a position."""
        with torch.no_grad():
            if recent_moves is None:
                recent_moves = []

            recent_moves_array = prepare_recent_moves_tokens(recent_moves[-RECENT_MOVES_LENGTH:])
            recent_moves_tokens = torch.tensor(
                recent_moves_array[np.newaxis, :], dtype=torch.long
            ).to(self.device)

            fen_tokens_array = tokenize(fen)
            fen_tokens = torch.tensor(
                fen_tokens_array[np.newaxis, :].astype(np.int64), dtype=torch.long
            ).to(self.device)

            input_ids = torch.cat([recent_moves_tokens, fen_tokens], dim=1)
            attention_mask = torch.ones_like(input_ids)

            scaled_rating = (rating - self.rating_mean) / self.rating_std
            scaled_rating = torch.tensor([scaled_rating], dtype=torch.float32).to(self.device)

            player_clock_scaled = (
                math.log(player_clock + 1) - self.log_player_clock_mean
            ) / self.log_player_clock_std
            opponent_clock_scaled = (
                math.log(opponent_clock + 1) - self.log_opponent_clock_mean
            ) / self.log_opponent_clock_std
            increment_scaled = (
                math.log(increment + 1) - self.log_increment_mean
            ) / self.log_increment_std

            clock_features = torch.tensor(
                [[player_clock_scaled, opponent_clock_scaled, increment_scaled]],
                dtype=torch.float32,
            ).to(self.device)

            logits = self.model(input_ids, attention_mask, scaled_rating, clock_features)
            probabilities = F.softmax(logits, dim=1)
            return probabilities.cpu().numpy()[0]

    def predict_thinking_time(
        self,
        fen: str,
        recent_moves: Optional[List[str]] = None,
        rating: int = 1500,
        player_clock: float = 300.0,
        opponent_clock: float = 300.0,
        increment: float = 0.0,
    ) -> float:
        """Predict a concrete thinking time (seconds) for the position.

        Samples a bucket from the model's distribution, then samples a concrete
        time from that bucket's empirical distribution. Returns the raw
        model-scale (blitz) time; the caller is responsible for any scaling.
        """
        bucket_probs = self.predict_bucket(
            fen,
            recent_moves or [],
            rating,
            player_clock,
            opponent_clock,
            increment,
        )

        bucket_idx = int(np.random.choice(len(bucket_probs), p=bucket_probs))
        thinking_time = self.bucket_info.sample_from_bucket_empirical(bucket_idx)
        return float(thinking_time)
