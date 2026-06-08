"""
KumoRFM-2 Relational Transformer — adapted from the architecture described in:
"KumoRFM-2: Scaling Foundation Models for Relational Learning" (arXiv:2604.12596v1, April 2026)

This implementation adapts KumoRFM-2's core hierarchical attention scheme to
session-based recommendation (RecSys Challenge 2015 — yoochoose dataset).

Key architectural elements from the paper:
  1. Hierarchical Attention: alternating column attention + row attention at the
     table level, then foreign-key attention + cross-sample attention at the graph level.
  2. Task conditioning: task/target information injected early via cross-attention.
  3. Context selection: local context (entity features) + global context (similar entities).

Reference: https://arxiv.org/abs/2604.12596
"""

from kumorfm2.model import KumoRFM2RelationalTransformer, RelationalTransformerConfig
from kumorfm2.data import SessionGraphDataset, collate_sessions

__all__ = [
    "KumoRFM2RelationalTransformer",
    "RelationalTransformerConfig",
    "SessionGraphDataset",
    "collate_sessions",
]
