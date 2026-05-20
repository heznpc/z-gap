"""Frozen model registry for Strategy D / E / F experiments.

Each model is pinned to the HuggingFace `main` branch commit SHA observed
on 2026-05-21 via `huggingface_hub.HfApi().model_info(repo).sha`.
sentence-transformers >=5.5 honors the `revision=` kwarg in
SentenceTransformer.__init__, so the experiments load exactly the weights
captured at review time even if the upstream `main` branch moves later.

This closes C3 from the 2026-05-21 pre-experiment review: explicit
revision pin in addition to the existing embedding-level `.npz` cache.

To refresh: re-run the snippet at the bottom of this file and commit the
new SHAs as a single chore PR. Do NOT update individual model SHAs
silently — keeping all 7 frozen at the same review point lets cross-
experiment comparisons (D / E / F) stay valid.
"""

from __future__ import annotations

# Frozen SHA snapshot: 2026-05-21
MODELS_7_FROZEN: list[tuple[str, str, dict]] = [
    ("microsoft/unixcoder-base", "UniXcoder (code)", {
        "revision": "5604afdc964f6c53782a6813140ade5216b99006",
    }),
    ("paraphrase-multilingual-MiniLM-L12-v2", "MiniLM-L12 (NL)", {
        # sentence-transformers/* namespace, but sentence-transformers
        # library auto-prefixes when the bare model name is used.
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    }),
    ("nomic-ai/nomic-embed-text-v1.5", "Nomic v1.5 (NL+code)", {
        "trust_remote_code": True,
        "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
    }),
    ("intfloat/multilingual-e5-small", "E5-small (NL)", {
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    }),
    ("intfloat/multilingual-e5-base", "E5-base (NL)", {
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
    }),
    ("intfloat/multilingual-e5-large", "E5-large (NL)", {
        "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    }),
    ("BAAI/bge-m3", "BGE-M3 (NL+code)", {
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    }),
]


def registry_sha_summary() -> dict:
    """Return a serializable model -> revision mapping for run_meta dumps."""
    return {model: kwargs.get("revision", "unpinned") for model, _, kwargs in MODELS_7_FROZEN}


# ---------------------------------------------------------------------------
# Refresh helper (manual; NOT called by experiments)
# ---------------------------------------------------------------------------
# Run interactively when you intentionally want to roll the frozen SHAs:
#
#   experiments/.venv/bin/python -c "
#   from huggingface_hub import HfApi
#   from experiments.src.model_registry import MODELS_7_FROZEN
#   api = HfApi()
#   for model, label, kwargs in MODELS_7_FROZEN:
#       info = api.model_info(model)
#       print(f'    ({model!r}, {label!r}, {{\"revision\": {info.sha!r}, ...}}),')
#   "
#
# Review the diff, commit as a chore PR, and re-run Strategy D / E / F so the
# results JSON _meta blocks pick up the new SHAs.
