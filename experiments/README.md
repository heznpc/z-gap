# Z-Gap Experiments

Tests predictions P1–P7 from the paper: cross-lingual semantic invariance (P2), dialect continuum (P2-dialect), NL-code alignment (P3), spacing robustness (P7), and scale-convergence (P1).

## Reproducibility envelope

- **Python**: `3.11` or `3.12` (pinned in `.python-version`; `kiwipiepy` and `torch` wheels for `3.13` are not yet consistently available)
- **Random seed**: `np.random.default_rng(42)` is used throughout `src/predictions.py`, `src/metrics.py`, `src/code_alignment.py`, `src/vocab_mediation.py`, `src/hidden_state_analysis.py`, and the strategy runners (17+ call sites). Override via the `seed` kwarg where exposed.
- **OS**: tested on macOS (Apple Silicon) and Ubuntu 24.04. Windows untested.
- **Hardware**: CPU sufficient for the 100-op pilot (~3–5h end-to-end across 7 embedding models). MPS/CUDA optional and only used by `scripts/run_v2_extract.py` for 8B decoder hidden-state extraction.
- **External APIs**: OpenAI Embeddings (`text-embedding-3-small`/`-large`) and Mistral Codestral Embed (`codestral-embed-2505`). Both calls now retry on 429/5xx with exponential backoff (`max_retries=5`).
- **Data sent to providers**: synthetic stimuli only (`data/stimuli/*.json`). No PII.

## Setup

From the repository root:

```bash
make setup     # creates experiments/.venv (Python 3.12) and installs requirements
cp experiments/.env.example experiments/.env   # add OPENAI_API_KEY, MISTRAL_API_KEY
make smoke     # validates imports + stimuli JSON without burning API budget
```

Or manually:

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
make reproduce     # full pipeline (scripts/run_all.py)
make figures       # cross-experiment synthesis only
```

The CI workflow (`.github/workflows/reproduce-smoke.yml`) runs the smoke target on every push touching `experiments/` — it confirms imports succeed and stimuli JSON parses, but does not burn API credits.

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

## Logging

Scripts ship with `print()` for transcript continuity. New code should use the helper:

```python
from src.logging_config import configure_logging, get_logger

configure_logging()                    # call once at entry point
logger = get_logger(__name__)
logger.info("Embedded %d stimuli", n)
```

Set `Z_GAP_LOG_LEVEL=DEBUG` to escalate. Output goes to stderr so stdout stays clean for redirection.
