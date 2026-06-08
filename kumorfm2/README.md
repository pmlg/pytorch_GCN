# KumoRFM-2 Relational Transformer

Adaptation of the **KumoRFM-2** architecture from [arXiv:2604.12596](https://arxiv.org/abs/2604.12596) (April 2026) to session-based recommendation on the [RecSys Challenge 2015](https://2015.recsyschallenge.com/) (yoochoose) dataset.

## Background

KumoRFM-2 (now acquired by NVIDIA, June 2026) introduced a **hierarchical attention** scheme for relational data that replaces graph convolutions with:

- **Column attention** — task-conditioned feature selection within table rows
- **Row attention** — item-to-item interaction within sessions (replaces GraphSAGE message passing)
- **Cross-sample attention** — information transfer across related sessions (context pool)
- **Task conditioning** — task metadata injected early for task-aware representations
- **Lagged target conditioning** — session-level statistics as conditioning features

This branch replaces the original `TorchGeometric.ipynb` GraphSAGE + TopKPooling model with this transformer architecture.

## Architecture

```
Item Embedding + Positional Encoding
         │
         ▼
  Task Conditioning (early task injection)
         │
         ▼
┌─────────────────────────────────┐
│  Level-1: Table Encoder (×N)    │
│  ┌─────────────────────────┐    │
│  │ Column Attention        │    │  ← Feature selection
│  │ Row Attention           │    │  ← Item interaction
│  └─────────────────────────┘    │
└─────────────────────────────────┘
         │
         ▼
  Session Pooling (Mean + Max)
         │
         ▼
  Lagged Target Conditioning
         │
         ▼
┌─────────────────────────────────┐
│  Level-2: Graph Encoder (×M)    │
│  Cross-Sample Attention         │  ← Session-to-session transfer
└─────────────────────────────────┘
         │
         ▼
  Classification Head
```

## Improvements over Original GraphSAGE Model

| Issue | Original (GraphSAGE) | KumoRFM-2 Branch |
|-------|---------------------|-------------------|
| Overfitting | Train AUC 0.93 vs Test 0.68 | Dropout, weight decay, early stopping |
| Class imbalance | Not addressed | Weighted BCE loss |
| Architecture | Fixed conv filters | Task-conditioned attention |
| Cross-session info | None | Cross-sample attention |
| Feature selection | None | Column attention |
| Metrics | AUC, F1 | AUC, AP, F1, Precision, Recall |

## Usage

```bash
# Install dependencies
pip install torch pandas scikit-learn

# Train
python -m kumorfm2.train \
    --clicks-path ../input/yoochoose-clicks.dat \
    --buys-path ../input/yoochoose-buys.dat \
    --epochs 20 \
    --batch-size 256 \
    --lr 1e-4
```

## File Structure

```
kumorfm2/
├── __init__.py       # Package exports
├── model.py          # KumoRFM2RelationalTransformer + all attention modules
├── data.py           # SessionGraphDataset + collate function
├── train.py          # Training pipeline with early stopping
└── README.md         # This file
```

## References

- **KumoRFM-2 Paper:** [KumoRFM-2: Scaling Foundation Models for Relational Learning](https://arxiv.org/abs/2604.12596) (Fey et al., April 2026)
- **KumoRFM-1 Paper:** [KumoRFM: A Foundation Model for In-Context Learning on Relational Data](https://arxiv.org/abs/2412.18934) (Fey et al., 2025)
- **Original Kumo Company:** [kumo.ai](https://kumo.ai) (acquired by NVIDIA, June 2026)
- **RecSys 2015:** [yoochoose Dataset](https://2015.recsyschallenge.com/challenge.html)
