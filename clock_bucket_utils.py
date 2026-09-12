"""Clock bucket utilities (vendored from ChessMimic / 1e4_ai).

Derived from ``Training/clock_bucket_utils.py`` in the upstream project
https://github.com/thomasj02/1e4_ai and distributed under the PolyForm
Noncommercial License 1.0.0 (see LICENSE and NOTICE).

Required Notice: Copyright 2026 Thomas Johnson (https://github.com/thomasj02/1e4_ai)

Trimmed to the loading + empirical-sampling helpers needed for inference.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np


class ClockBucketInfo:
    """Container for clock bucket information."""

    def __init__(self, bucket_data: Dict):
        # Convert None (or JSON Infinity) to float('inf') in boundaries.
        self.boundaries = []
        for b in bucket_data["boundaries"]:
            if b is None:
                self.boundaries.append(float("inf"))
            else:
                self.boundaries.append(float(b))

        self.n_buckets = bucket_data["n_buckets"]
        self.scheme = bucket_data.get("scheme")
        self.time_control = bucket_data.get("time_control")
        self.statistics = bucket_data.get("statistics")
        self.bucket_probabilities = np.array(bucket_data.get("bucket_probabilities", []))

        # Load and validate empirical distributions.
        if "bucket_empirical_distributions" not in bucket_data:
            raise ValueError("bucket_empirical_distributions missing from bucket data")

        self.empirical_distributions = bucket_data["bucket_empirical_distributions"]

        if len(self.empirical_distributions) != self.n_buckets:
            raise ValueError(
                f"Expected {self.n_buckets} empirical distributions, "
                f"but got {len(self.empirical_distributions)}"
            )

        for i, dist in enumerate(self.empirical_distributions):
            if "type" not in dist:
                raise ValueError(f"bucket {i} missing 'type' field")
            if "distribution" not in dist:
                raise ValueError(f"bucket {i} missing 'distribution' field")
            if not dist["distribution"]:
                raise ValueError(f"bucket {i} has empty distribution")

    def get_bucket_range(self, bucket_idx: int) -> Tuple[float, float]:
        """Get range [low, high) for a bucket."""
        if bucket_idx < 0 or bucket_idx >= self.n_buckets:
            raise ValueError(f"Invalid bucket index: {bucket_idx}")
        return self.boundaries[bucket_idx], self.boundaries[bucket_idx + 1]

    def sample_from_bucket_empirical(self, bucket_idx: int) -> float:
        """Sample a time value from a bucket using its empirical distribution."""
        if bucket_idx < 0 or bucket_idx >= self.n_buckets:
            raise ValueError(f"Invalid bucket index: {bucket_idx}")

        dist_info = self.empirical_distributions[bucket_idx]
        dist_type = dist_info["type"]
        distribution = dist_info["distribution"]

        if dist_type == "seconds":
            seconds = []
            probabilities = []
            for sec_str, prob in distribution.items():
                seconds.append(int(sec_str))
                probabilities.append(prob)
            probabilities = np.array(probabilities) / sum(probabilities)
            return float(np.random.choice(seconds, p=probabilities))

        elif dist_type == "frequent_values":
            values = []
            probabilities = []
            for val_str, prob in distribution.items():
                values.append(float(val_str))
                probabilities.append(prob)
            probabilities = np.array(probabilities) / sum(probabilities)
            return float(np.random.choice(values, p=probabilities))

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def __repr__(self) -> str:
        return (
            f"ClockBucketInfo(n_buckets={self.n_buckets}, "
            f"time_control={self.time_control}, scheme={self.scheme})"
        )


def load_bucket_info(path: str) -> ClockBucketInfo:
    """Load bucket information from a JSON file."""
    with open(path, "r") as f:
        bucket_data = json.load(f)
    return ClockBucketInfo(bucket_data)
