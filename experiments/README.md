# Z-Gap Experiments

Tests predictions P1–P7 from the paper: cross-lingual semantic invariance (P2), dialect continuum (P2-dialect), NL-code alignment (P3), spacing robustness (P7), and scale-convergence (P1).

## Setup

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add API keys (OpenAI, Mistral)
```

## Run

```bash
python scripts/run_all.py
```

## What it does

1. Generates 100 stimuli (50 computational + 50 judgment) × 5 languages × dialectal variants (~1,800 total)
2. Embeds through 8 models: MiniLM, E5-small/base/large, BGE-M3, Qwen3-Embedding, jina-v3, Codestral Embed
3. Computes discriminability ratio R = d\_inter / d\_intra
4. Tests P2: cross-lingual invariance (and cross-dialectal continuum)
5. Tests P7: spacing/punctuation robustness
6. Generates figures in `results/figures/`

## Models

| Model | Dim | Role |
|-------|-----|------|
| MiniLM-L12 | 384 | Baseline |
| E5-small / base / large | 384 / 768 / 1024 | P1 scale-convergence |
| BGE-M3 | 1024 | Cross-lingual retrieval |
| Qwen3-Embedding-8B | 4096 | MTEB multilingual SOTA |
| jina-embeddings-v3 | 1024 | Multilingual |
| Codestral Embed | 1024 | Code-specialized |

## Predictions

- **P1**: NL-code cosine distance decreases with model scale (E5 family)
- **P2**: R\_C > R\_J for computational vs judgment operations
- **P2-dialect**: R degrades continuously: within-dialect > cross-dialect > cross-lingual
- **P7**: R\_spacing > 1 — spacing variation produces less Z distance than semantic variation
