"""
Data processing for KumoRFM-2 Relational Transformer on RecSys 2015 (yoochoose).

Converts raw session-click data into the relational format expected by
KumoRFM2RelationalTransformer:
  - Padded item ID sequences with masks
  - Lagged target features (session statistics)
  - Context pools for cross-sample attention
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class SessionSample:
    """A single session sample with context pool."""

    item_ids: Tensor        # (seq_len,) padded item IDs
    mask: Tensor            # (seq_len,) True = padding
    label: float            # 1.0 = purchase, 0.0 = no purchase
    lagged: Tensor          # (3,) normalized session statistics
    context_items: Tensor   # (pool_size, c_seq_len) context sessions
    context_mask: Tensor    # (pool_size,) True = no context at this slot
    context_lagged: Tensor  # (pool_size, 3) context lagged features


class SessionGraphDataset(Dataset):
    """
    Dataset for session-based recommendation using KumoRFM-2 architecture.

    Processes the yoochoose RecSys 2015 click data into padded sessions
    with purchase labels, lagged features, and context pools.

    Args:
        clicks_df: DataFrame with columns [session_id, timestamp, item_id, category]
        buys_df: DataFrame with columns [session_id, timestamp, item_id, price, quantity]
        max_session_len: Maximum items per session (truncated)
        context_pool_size: Number of context sessions for cross-sample attention
        min_session_len: Minimum items to keep a session (filter noise)
        max_sessions: Maximum total sessions (for sampling)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        clicks_df: pd.DataFrame,
        buys_df: pd.DataFrame | None = None,
        max_session_len: int = 200,
        context_pool_size: int = 32,
        min_session_len: int = 3,
        max_sessions: int = 1_000_000,
        seed: int = 42,
    ):
        self.max_session_len = max_session_len
        self.context_pool_size = context_pool_size
        self.rng = np.random.RandomState(seed)

        # --- Preprocess ---
        # Filter sessions with too few interactions
        session_sizes = clicks_df.groupby("session_id").size()
        valid_sessions = session_sizes[session_sizes >= min_session_len].index
        clicks_df = clicks_df[clicks_df.session_id.isin(valid_sessions)]

        # Sample if too many sessions
        unique_sessions = clicks_df.session_id.unique()
        if len(unique_sessions) > max_sessions:
            sampled = self.rng.choice(unique_sessions, max_sessions, replace=False)
            clicks_df = clicks_df[clicks_df.session_id.isin(sampled)]

        # Label encode item IDs
        all_items = clicks_df.item_id.unique()
        self.item_encoder = {item: idx + 1 for idx, item in enumerate(all_items)}  # 0 = padding
        self.num_items = len(all_items) + 1
        clicks_df = clicks_df.copy()
        clicks_df["item_encoded"] = clicks_df.item_id.map(self.item_encoder)

        # Sort by session and timestamp
        clicks_df = clicks_df.sort_values(["session_id", "timestamp"])

        # --- Labels: 1 if session led to a purchase ---
        buy_sessions = set()
        if buys_df is not None:
            buy_sessions = set(buys_df.session_id.unique())
        clicks_df["label"] = clicks_df.session_id.isin(buy_sessions).astype(float)

        # --- Build session sequences ---
        self.sessions = []
        self.labels = []
        self.session_stats = []  # For lagged features

        for session_id, group in clicks_df.groupby("session_id"):
            items = group.item_encoded.values[:max_session_len]
            label = group.label.iloc[0]

            # Session statistics for lagged features
            session_len = len(items)
            unique_items = len(set(items))
            time_span = (group.timestamp.max() - group.timestamp.min()).total_seconds()
            # Normalize: log-scale for robustness
            lagged = np.array([
                np.log1p(session_len) / np.log1p(max_session_len),  # Length
                unique_items / max(session_len, 1),                  # Diversity
                np.log1p(time_span) / np.log1p(3600 * 24),          # Time span (cap at 1 day)
            ], dtype=np.float32)

            self.sessions.append(items)
            self.labels.append(label)
            self.session_stats.append(lagged)

        self.labels = np.array(self.labels)
        self.session_stats = np.array(self.session_stats)

        # --- Precompute context pools (random sampling) ---
        self._build_context_pools()

        n_purchase = int(self.labels.sum())
        print(f"Dataset: {len(self.sessions)} sessions | "
              f"{n_purchase} purchase ({100*n_purchase/len(self.labels):.1f}%) | "
              f"{self.num_items} items")

    def _build_context_pools(self):
        """Precompute random context pools for each session."""
        n = len(self.sessions)
        self.context_indices = np.zeros((n, self.context_pool_size), dtype=int)
        for i in range(n):
            # Sample context: mix of same-label and different-label sessions
            pool = self.rng.choice(n, self.context_pool_size, replace=True)
            self.context_indices[i] = pool

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> SessionSample:
        items = self.sessions[idx]
        seq_len = len(items)

        # Pad item IDs
        padded_items = np.zeros(self.max_session_len, dtype=np.int64)
        padded_items[:seq_len] = items
        mask = np.ones(self.max_session_len, dtype=bool)
        mask[:seq_len] = False  # False = real token, True = padding

        # Lagged features
        lagged = self.session_stats[idx]

        # Context pool
        ctx_indices = self.context_indices[idx]
        context_items = np.zeros((self.context_pool_size, self.max_session_len), dtype=np.int64)
        context_lagged = np.zeros((self.context_pool_size, 3), dtype=np.float32)

        for j, ci in enumerate(ctx_indices):
            ctx_seq = self.sessions[ci]
            ctx_len = min(len(ctx_seq), self.max_session_len)
            context_items[j, :ctx_len] = ctx_seq[:ctx_len]
            context_lagged[j] = self.session_stats[ci]

        return SessionSample(
            item_ids=torch.from_numpy(padded_items),
            mask=torch.from_numpy(mask),
            label=float(self.labels[idx]),
            lagged=torch.from_numpy(lagged),
            context_items=torch.from_numpy(context_items),
            context_mask=torch.zeros(self.context_pool_size, dtype=bool),  # All valid
            context_lagged=torch.from_numpy(context_lagged),
        )


def collate_sessions(batch: list[SessionSample]) -> dict:
    """
    Collate function for DataLoader.

    Returns dict with batched tensors:
        item_ids:        (batch, seq_len)
        mask:            (batch, seq_len)
        labels:          (batch,)
        lagged_features: (batch, 3)
        task_type:       (batch,) — all zeros (purchase prediction)
        context_item_ids:   (batch, pool_size, c_seq_len)
        context_mask:       (batch, pool_size)
        context_lagged:     (batch, pool_size, 3)
    """
    return {
        "item_ids": torch.stack([s.item_ids for s in batch]),
        "mask": torch.stack([s.mask for s in batch]),
        "labels": torch.tensor([s.label for s in batch], dtype=torch.float),
        "lagged_features": torch.stack([s.lagged for s in batch]),
        "task_type": torch.zeros(len(batch), dtype=torch.long),  # 0 = purchase prediction
        "context_item_ids": torch.stack([s.context_items for s in batch]),
        "context_mask": torch.stack([s.context_mask for s in batch]),
        "context_lagged": torch.stack([s.context_lagged for s in batch]),
    }
