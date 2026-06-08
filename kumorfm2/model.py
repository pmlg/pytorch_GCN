"""
KumoRFM-2 Relational Transformer Model.

Implements the hierarchical attention scheme from KumoRFM-2 (arXiv:2604.12596v1):

  Level 1 — Table-level encoder (lightweight):
    Alternating column attention (feature-level) and row attention (item-level)
    within each session to produce task-conditioned row embeddings.

  Level 2 — Graph-level encoder (larger):
    Cross-sample attention across sessions (context pool) to propagate
    information between similar sessions.

  Task Conditioning:
    Binary purchase signal injected via cross-attention at the earliest layer,
    enabling task-relevant column selection (§3: "injects task information as
    early as possible, enabling sharper selection of task-relevant columns").

  Lagged Targets (simplified):
    Prior session statistics (session length, item diversity) used as
    conditioning features, analogous to the paper's "lagged targets and prior
    subgraphs" mechanism.

References:
  - KumoRFM-2: https://arxiv.org/abs/2604.12596
  - Vaswani et al., "Attention Is All You Need" (2017)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RelationalTransformerConfig:
    """Configuration for the KumoRFM-2 Relational Transformer."""

    num_items: int = 50000          # Vocabulary size for item embeddings
    embed_dim: int = 128            # Item embedding dimension
    hidden_dim: int = 256           # Hidden dimension for attention layers
    num_heads: int = 8              # Number of attention heads
    num_table_layers: int = 3       # Level-1 table encoder layers (column + row attention pairs)
    num_graph_layers: int = 2       # Level-2 graph encoder layers (cross-sample attention)
    dropout: float = 0.1            # Dropout rate (address overfitting from original GraphSAGE model)
    max_session_len: int = 200      # Maximum items per session (for positional encoding)
    context_pool_size: int = 32     # Number of context sessions for cross-sample attention
    use_lagged_targets: bool = True # Enable lagged target conditioning (§3 "Smarter Context and Lagged Targets")
    use_task_conditioning: bool = True  # Enable task conditioning (§3 "injects task information as early as possible")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for session item ordering."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional masking."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_v, d_model)
            mask:  (batch, seq_q, seq_k) or None. True = masked (ignore).

        Returns:
            (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # (batch, heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (batch, heads, seq_q, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(out)


class ColumnAttention(nn.Module):
    """
    KumoRFM-2 Level-1a: Column Attention.

    In the original paper, column attention operates across features (columns)
    within each table row, enabling task-relevant feature selection.

    For session data, we treat each item's embedding dimensions as "columns"
    and apply self-attention across feature groups, allowing the model to
    dynamically weight different aspects of item representations (e.g.,
    popularity, recency, category) based on the task.

    Implementation: feature-group attention via grouped linear projections.
    """

    def __init__(self, d_model: int, num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.group_dim = d_model // num_groups

        # Attention weights for each feature group
        self.group_projections = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim) for _ in range(num_groups)
        ])
        self.group_gate = nn.Linear(d_model, num_groups)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Split into feature groups
        groups = x.view(batch_size, seq_len, self.num_groups, self.group_dim)

        # Compute attention weights for each group
        gates = torch.sigmoid(self.group_gate(x))  # (batch, seq_len, num_groups)

        # Transform each group
        transformed = []
        for i, proj in enumerate(self.group_projections):
            g = groups[:, :, i, :]  # (batch, seq_len, group_dim)
            g = F.relu(proj(g))
            g = g * gates[:, :, i:i + 1]  # Gate each group
            transformed.append(g)

        out = torch.cat(transformed, dim=-1)  # (batch, seq_len, d_model)
        out = self.dropout(out)

        return self.norm(x + out)


class RowAttention(nn.Module):
    """
    KumoRFM-2 Level-1b: Row Attention.

    Self-attention across rows (items) within a session/table.
    This captures item-to-item relationships within each session,
    replacing the message-passing aggregation of GraphSAGE.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) padding mask — True = padding
        Returns: (batch, seq_len, d_model)
        """
        # Self-attention
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq, seq)

        attn_out = self.attention(x, x, x, mask=attn_mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TableEncoderLayer(nn.Module):
    """
    KumoRFM-2 Level-1: One layer of the table encoder.

    Alternating column attention → row attention, as described in §3:
    "A lightweight network first extracts task-conditioned row embeddings
    from individual tables through alternating column and row attention."
    """

    def __init__(self, d_model: int, num_heads: int, num_feature_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.column_attn = ColumnAttention(d_model, num_feature_groups, dropout)
        self.row_attn = RowAttention(d_model, num_heads, dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # Column attention first (feature selection), then row attention (item interaction)
        x = self.column_attn(x)
        x = self.row_attn(x, mask)
        return x


class CrossSampleAttention(nn.Module):
    """
    KumoRFM-2 Level-2: Cross-Sample Attention.

    "A larger network then distributes and relates these embeddings across
    tables and context samples via foreign key and cross-sample attention."

    For session data, this attends across sessions in the batch/context pool,
    allowing the model to leverage information from similar sessions.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """
        query:  (batch, 1, d_model) — session-level representation to update
        context: (batch, pool_size, d_model) — context session representations
        context_mask: (batch, pool_size) — True = padding
        Returns: (batch, 1, d_model)
        """
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1)  # (batch, 1, pool_size)

        attn_out = self.attention(query, context, context, mask=attn_mask)
        query = self.norm1(query + attn_out)

        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query


class TaskConditioning(nn.Module):
    """
    KumoRFM-2 Task Conditioning (§3):
    "KumoRFM-2 injects task information as early as possible, enabling sharper
    selection of task-relevant columns and improved robustness to noisy data."

    Generates a task embedding from task metadata (e.g., prediction type,
    temporal horizon) and modulates item representations via cross-attention.
    """

    def __init__(self, d_model: int, task_embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        # Task type embedding: 0 = purchase prediction, 1 = churn, 2 = next-item, etc.
        self.task_type_embed = nn.Embedding(10, task_embed_dim)
        self.task_proj = nn.Linear(task_embed_dim, d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, task_type: Tensor) -> Tensor:
        """
        x: (batch, seq_len, d_model)
        task_type: (batch,) — integer task type
        Returns: (batch, seq_len, d_model)
        """
        task_embed = self.task_proj(self.task_type_embed(task_type))  # (batch, d_model)
        task_embed = task_embed.unsqueeze(1)  # (batch, 1, d_model)

        # Cross-attention: items attend to task embedding
        task_out = self.cross_attn(x, task_embed, task_embed)
        return self.norm(x + task_out)


class KumoRFM2RelationalTransformer(nn.Module):
    """
    Full KumoRFM-2 Relational Transformer adapted for session-based prediction.

    Architecture (bottom to top):
      1. Item embedding + positional encoding
      2. [Optional] Task conditioning (task info injected early)
      3. Level-1: Table encoder (N layers of column attention → row attention)
      4. Session pooling (mean + max → concatenated → projected)
      5. [Optional] Lagged target conditioning
      6. Level-2: Graph encoder (M layers of cross-sample attention)
      7. Classification head

    This replaces the original GraphSAGE + TopKPooling architecture with
    KumoRFM-2's hierarchical attention, which provides:
      - Better feature selection (column attention vs fixed conv filters)
      - Direct item-item interaction (row attention vs neighborhood sampling)
      - Cross-session information transfer (cross-sample attention)
      - Task-aware representations (task conditioning)
    """

    def __init__(self, config: RelationalTransformerConfig):
        super().__init__()
        self.config = config

        # --- Embedding ---
        self.item_embedding = nn.Embedding(config.num_items, config.embed_dim)
        self.pos_encoding = PositionalEncoding(config.embed_dim, config.max_session_len)
        self.embed_proj = nn.Linear(config.embed_dim, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # --- Task Conditioning (§3: early task injection) ---
        if config.use_task_conditioning:
            self.task_conditioning = TaskConditioning(
                config.hidden_dim, task_embed_dim=32, dropout=config.dropout
            )

        # --- Level-1: Table Encoder (column + row attention) ---
        self.table_encoder = nn.ModuleList([
            TableEncoderLayer(
                d_model=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_table_layers)
        ])

        # --- Session Pooling ---
        self.session_pool_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # --- Lagged Target Conditioning (§3: "Smarter Context and Lagged Targets") ---
        if config.use_lagged_targets:
            # Lagged features: session_length_norm, item_diversity_norm, time_span_norm
            self.lagged_proj = nn.Linear(3, config.hidden_dim)
            self.lagged_norm = nn.LayerNorm(config.hidden_dim)

        # --- Level-2: Graph Encoder (cross-sample attention) ---
        self.graph_encoder = nn.ModuleList([
            CrossSampleAttention(
                d_model=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_graph_layers)
        ])

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_session(
        self,
        item_ids: Tensor,
        mask: Tensor | None = None,
        task_type: Tensor | None = None,
        lagged_features: Tensor | None = None,
    ) -> Tensor:
        """
        Encode a batch of sessions into session-level representations.

        Args:
            item_ids: (batch, seq_len) — padded item IDs
            mask: (batch, seq_len) — True = padding position
            task_type: (batch,) — task type for conditioning
            lagged_features: (batch, 3) — normalized session statistics

        Returns:
            session_repr: (batch, hidden_dim)
        """
        batch_size, seq_len = item_ids.shape

        # 1. Embed + positional encoding
        x = self.item_embedding(item_ids)  # (batch, seq_len, embed_dim)
        x = self.pos_encoding(x)
        x = self.embed_proj(x)  # (batch, seq_len, hidden_dim)
        x = self.embed_dropout(x)

        # 2. Task conditioning (early injection, §3)
        if self.config.use_task_conditioning and task_type is not None:
            x = self.task_conditioning(x, task_type)

        # 3. Level-1: Table encoder (alternating column + row attention)
        for layer in self.table_encoder:
            x = layer(x, mask)

        # 4. Session pooling: global mean + max pooling
        if mask is not None:
            # Mask out padding for pooling
            mask_exp = mask.unsqueeze(-1).expand_as(x)  # (batch, seq, hidden)
            x_masked = x.masked_fill(mask_exp, 0.0)
            mean_pool = x_masked.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)

            # Max pool with -inf for padding
            x_for_max = x.masked_fill(mask_exp, float("-inf"))
            max_pool = x_for_max.max(dim=1).values
            max_pool = torch.nan_to_num(max_pool, nan=0.0)  # Handle all-masked rows
        else:
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1).values

        session_repr = torch.cat([max_pool, mean_pool], dim=-1)  # (batch, hidden*2)
        session_repr = self.session_pool_proj(session_repr)  # (batch, hidden)

        # 5. Lagged target conditioning (§3: "Smarter Context and Lagged Targets")
        if self.config.use_lagged_targets and lagged_features is not None:
            lagged_embed = F.relu(self.lagged_proj(lagged_features))
            session_repr = self.lagged_norm(session_repr + lagged_embed)

        return session_repr

    def forward(
        self,
        item_ids: Tensor,
        mask: Tensor | None = None,
        task_type: Tensor | None = None,
        lagged_features: Tensor | None = None,
        context_item_ids: Tensor | None = None,
        context_mask: Tensor | None = None,
        context_lagged: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for a batch of sessions.

        Args:
            item_ids: (batch, seq_len) — item IDs for target sessions
            mask: (batch, seq_len) — padding mask (True = padding)
            task_type: (batch,) — task type IDs
            lagged_features: (batch, 3) — normalized session statistics
            context_item_ids: (batch, pool_size, c_seq_len) — context session items
            context_mask: (batch, pool_size) — context padding mask (True = no context)
            context_lagged: (batch, pool_size, 3) — context lagged features

        Returns:
            logits: (batch,) — raw logits for BCEWithLogitsLoss
        """
        batch_size = item_ids.size(0)

        # Encode target sessions
        session_repr = self.encode_session(
            item_ids, mask, task_type, lagged_features
        )  # (batch, hidden_dim)

        # 6. Level-2: Cross-sample attention (if context provided)
        if context_item_ids is not None and len(self.graph_encoder) > 0:
            batch_size, pool_size, c_seq_len = context_item_ids.shape

            # Encode context sessions
            context_reprs = []
            for i in range(pool_size):
                ctx_items = context_item_ids[:, i, :]  # (batch, c_seq_len)
                ctx_repr = self.encode_session(
                    ctx_items,
                    task_type=task_type,
                    lagged_features=context_lagged[:, i, :] if context_lagged is not None else None,
                )
                context_reprs.append(ctx_repr)

            context_reprs = torch.stack(context_reprs, dim=1)  # (batch, pool_size, hidden)

            # Cross-sample attention
            query = session_repr.unsqueeze(1)  # (batch, 1, hidden)
            ctx_pad_mask = None
            if context_mask is not None:
                ctx_pad_mask = context_mask

            for layer in self.graph_encoder:
                query = layer(query, context_reprs, ctx_pad_mask)

            session_repr = query.squeeze(1)  # (batch, hidden)

        # 7. Classification
        logits = self.classifier(session_repr).squeeze(-1)  # (batch,)

        return logits
