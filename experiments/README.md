# Pilot Experiment: Cross-Lingual Semantic Invariance

Tests predictions P2 (cross-lingual semantic invariance) and P7 (spacing robustness) from the paper.

## Setup

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OpenAI key if using OpenAI embeddings
```

## Run

```bash
python scripts/run_all.py
```

## What it does

1. Generates 20 stimuli (10 computational + 10 judgment) × 5 languages
2. Embeds all descriptions using multilingual sentence-transformers
3. Computes discriminability ratio R = d_inter / d_intra
4. Tests P2: R_C > R_J (computational ops cluster by operation, not language)
5. Tests P7: R_spacing > 1 (Korean spacing variants cluster by meaning)
6. Generates figures in `results/figures/`

## Predictions

- **P2**: R_C > R_J — cross-lingual same-operation similarity exceeds within-language different-operation similarity, more so for computational than judgment operations
- **P7**: R_spacing > 1 — spacing variation produces less Z distance than semantic variation
