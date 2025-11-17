# Imperfect Scrobbles Resolution

This project builds a modular pipeline to detect and reconcile "imperfect scrobbles"—Last.fm track/artist/album entries that should resolve to the same canonical entity but differ due to metadata inconsistencies.

## Components

1. **Data generation** (`data/`, `scripts/generate_pairs.py`)
   - Seed entity definitions and multilingual variants
   - Variation, noise, and hard-negative generators
   - User profile + scrobble history simulation for contextual features

2. **Semantic model training** (`scripts/train_semantic.py`)
   - Fine-tunes a multilingual sentence-transformer
   - Produces standalone embeddings for entity strings

3. **Context-aware adjustment** (`inference/matcher.py`, `scripts/train_context_mlp.py`)
   - Optional rule-based or learned boosters that leverage scrobble context

4. **Evaluation + reporting** (`scripts/evaluate.py`, `reports/`)
   - Side-by-side metrics with/without context
   - Breakdown by entity type, variation type, and hard negatives

## Quick start

1. Install dependencies: `pip install -r requirements.txt`
2. Generate synthetic data (configures ratios/simulation via YAML):
   ```bash
   python scripts/generate_pairs.py --config configs/data.yaml
   ```
3. Fine-tune the semantic encoder:
   ```bash
   python scripts/train_semantic.py --data data/dataset.jsonl --output-dir models/semantic
   ```
4. (Optional) Train the learned context fusion head:
   ```bash
   python scripts/train_context_mlp.py --data data/dataset.jsonl --output models/context/mlp.pt
   ```
5. Evaluate semantic vs. context-aware scoring:
   ```bash
   python scripts/evaluate.py --data data/dataset.jsonl --semantic-checkpoint models/semantic --context-head models/context/mlp.pt
   ```

## Repository layout

```
imperfect_scrobbles/
  data/                # Synthetic data + context simulators
  scripts/             # CLI entrypoints for generation/training/eval
  inference/           # Runtime matcher and API
  configs/             # YAML configs for reproducibility
  models/              # Saved checkpoints
  context/             # Optional learned context heads
  reports/             # Evaluation outputs (metrics, plots)
  tests/               # Unit/regression tests
```

Each module includes docstrings that explain responsibilities and extension points.
