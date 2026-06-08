#!/usr/bin/env python3
"""
Training script for KumoRFM-2 Relational Transformer on RecSys 2015 (yoochoose).

Implements the full training pipeline with:
  - KumoRFM-2 hierarchical attention model
  - Class imbalance handling (weighted BCE)
  - Early stopping (addresses overfitting from original GraphSAGE model)
  - Full metrics suite (AUC, F1, Precision, Recall)

Usage:
  python -m kumorfm2.train --clicks-path ../input/yoochoose-clicks.dat \
                            --buys-path ../input/yoochoose-buys.dat \
                            --epochs 20 --batch-size 256

Paper reference: https://arxiv.org/abs/2604.12596
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from kumorfm2.data import SessionGraphDataset, collate_sessions
from kumorfm2.model import KumoRFM2RelationalTransformer, RelationalTransformerConfig


def load_yoochoose(clicks_path: str, buys_path: str, max_sessions: int = 500000):
    """Load yoochoose RecSys 2015 data."""
    clicks = pd.read_csv(
        clicks_path,
        header=None,
        parse_dates=[1],
        low_memory=False,
    )
    clicks.columns = ["session_id", "timestamp", "item_id", "category"]

    buys = pd.read_csv(
        buys_path,
        header=None,
        parse_dates=[1],
    )
    buys.columns = ["session_id", "timestamp", "item_id", "price", "quantity"]

    # Sample sessions if too large
    unique_sessions = clicks.session_id.unique()
    if len(unique_sessions) > max_sessions:
        rng = np.random.RandomState(42)
        sampled = rng.choice(unique_sessions, max_sessions, replace=False)
        clicks = clicks[clicks.session_id.isin(sampled)]
        buys = buys[buys.session_id.isin(sampled)]

    print(f"Loaded {len(clicks):,} clicks across {clicks.session_id.nunique():,} sessions")
    print(f"Loaded {len(buys):,} buys across {buys.session_id.nunique():,} sessions")

    return clicks, buys


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        # Move to device
        item_ids = batch["item_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        lagged = batch["lagged_features"].to(device)
        task_type = batch["task_type"].to(device)
        ctx_items = batch["context_item_ids"].to(device)
        ctx_mask = batch["context_mask"].to(device)
        ctx_lagged = batch["context_lagged"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(
            item_ids=item_ids,
            mask=mask,
            task_type=task_type,
            lagged_features=lagged,
            context_item_ids=ctx_items,
            context_mask=ctx_mask,
            context_lagged=ctx_lagged,
        )

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        item_ids = batch["item_ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"]
        lagged = batch["lagged_features"].to(device)
        task_type = batch["task_type"].to(device)
        ctx_items = batch["context_item_ids"].to(device)
        ctx_mask = batch["context_mask"].to(device)
        ctx_lagged = batch["context_lagged"].to(device)

        logits = model(
            item_ids=item_ids,
            mask=mask,
            task_type=task_type,
            lagged_features=lagged,
            context_item_ids=ctx_items,
            context_mask=ctx_mask,
            context_lagged=ctx_lagged,
        )

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.hstack(all_preds)
    all_labels = np.hstack(all_labels)

    # Threshold predictions for classification metrics
    binary_preds = (all_preds > threshold).astype(int)

    # Handle edge case where all predictions are one class
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5

    try:
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        ap = 0.0

    return {
        "auc": auc,
        "ap": ap,
        "f1": f1_score(all_labels, binary_preds, zero_division=0),
        "precision": precision_score(all_labels, binary_preds, zero_division=0),
        "recall": recall_score(all_labels, binary_preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train KumoRFM-2 Relational Transformer on RecSys 2015"
    )
    parser.add_argument("--clicks-path", type=str, default="../input/yoochoose-clicks.dat")
    parser.add_argument("--buys-path", type=str, default="../input/yoochoose-buys.dat")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-table-layers", type=int, default=3)
    parser.add_argument("--num-graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-session-len", type=int, default=200)
    parser.add_argument("--context-pool-size", type=int, default=16)
    parser.add_argument("--max-sessions", type=int, default=500000)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--save-path", type=str, default="checkpoints/kumorfm2_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Data ---
    clicks, buys = load_yoochoose(
        args.clicks_path, args.buys_path, max_sessions=args.max_sessions
    )

    # Split: 80% train, 10% val, 10% test
    dataset = SessionGraphDataset(
        clicks_df=clicks,
        buys_df=buys,
        max_session_len=args.max_session_len,
        context_pool_size=args.context_pool_size,
        max_sessions=args.max_sessions,
    )

    n = len(dataset)
    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[: int(0.8 * n)]
    val_idx = indices[int(0.8 * n) : int(0.9 * n)]
    test_idx = indices[int(0.9 * n) :]

    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sessions, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sessions, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_sessions, num_workers=4
    )

    # --- Model ---
    config = RelationalTransformerConfig(
        num_items=dataset.num_items,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_table_layers=args.num_table_layers,
        num_graph_layers=args.num_graph_layers,
        dropout=args.dropout,
        max_session_len=args.max_session_len,
        context_pool_size=args.context_pool_size,
    )
    model = KumoRFM2RelationalTransformer(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --- Class imbalance handling ---
    purchase_rate = dataset.labels.mean()
    pos_weight = torch.tensor([(1 - purchase_rate) / max(purchase_rate, 1e-8)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Purchase rate: {purchase_rate:.4f} | pos_weight: {pos_weight.item():.2f}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- Training loop with early stopping ---
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'Train AUC':>9} | {'Val AUC':>8} | {'Val F1':>6} | {'Val AP':>7} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(args.epochs):
        t0 = time.time()

        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        print(
            f"{epoch:5d} | {loss:8.5f} | {train_metrics['auc']:9.5f} | "
            f"{val_metrics['auc']:8.5f} | {val_metrics['f1']:6.4f} | "
            f"{val_metrics['ap']:7.5f} | {elapsed:5.1f}s"
        )

        # Early stopping on validation AUC
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # --- Final test evaluation ---
    print(f"\nLoading best model (val AUC={best_val_auc:.5f})")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"TEST RESULTS (KumoRFM-2 Relational Transformer)")
    print(f"{'='*50}")
    print(f"  AUC:       {test_metrics['auc']:.5f}")
    print(f"  AP:        {test_metrics['ap']:.5f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
