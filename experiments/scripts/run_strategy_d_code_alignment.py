#!/usr/bin/env python3
"""Strategy D: Enhanced NL-Code Alignment (per-language × per-model R_code matrix).

Extends the original NL-code alignment experiment from 2 models × aggregate
to 4 models × 5 languages with per-cell statistical testing.

Models:
  1. UniXcoder (code-trained, 768d) — existing
  2. MiniLM-L12 (NL-only, 384d) — existing
  3. Nomic Embed Text v1.5 (NL+code, 768d) — new
  4. E5-large (NL multilingual, 1024d) — existing embedder, new for code alignment

Usage:
    python experiments/scripts/run_strategy_d_code_alignment.py
"""

import json
import sys
import gc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.code_alignment import CODE_EQUIVALENTS, compute_per_language_R_code

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

MODELS = [
    ("microsoft/unixcoder-base", "UniXcoder (code)", {}),
    ("paraphrase-multilingual-MiniLM-L12-v2", "MiniLM-L12 (NL)", {}),
    ("nomic-ai/nomic-embed-text-v1.5", "Nomic v1.5 (NL+code)", {"trust_remote_code": True}),
    ("intfloat/multilingual-e5-large", "E5-large (NL)", {}),
]


def run_model(model_name: str, label: str, kwargs: dict) -> dict:
    """Run per-language R_code for one model."""
    print(f"\n{'='*60}")
    print(f"  {label} ({model_name})")
    print(f"{'='*60}")

    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name, **kwargs)
    print(f"  dim={model.dimension}")

    # Embed NL descriptions (comp only)
    nl_texts, nl_keys = [], []
    for op in ops:
        if op.category != "computational":
            continue
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                nl_texts.append(desc)
                nl_keys.append(f"{op.id}_{lang}")

    nl_array = cache.get_or_compute(model, nl_texts)
    nl_embeddings = {k: nl_array[i] for i, k in enumerate(nl_keys)}

    # Embed code snippets
    code_texts, code_keys = [], []
    for op_id in comp_ids:
        if op_id in CODE_EQUIVALENTS:
            code_texts.append(CODE_EQUIVALENTS[op_id])
            code_keys.append(op_id)

    code_array = cache.get_or_compute(model, code_texts)
    code_embeddings = {k: code_array[i] for i, k in enumerate(code_keys)}

    print(f"  {len(nl_embeddings)} NL embeddings, {len(code_embeddings)} code embeddings")

    # Per-language R_code with statistics
    print("  Computing per-language R_code (permutation + bootstrap)...")
    result = compute_per_language_R_code(
        nl_embeddings, code_embeddings, comp_ids, LANGUAGES,
        n_perm=10000, n_boot=10000,
    )

    # Print results
    print(f"\n  {'Lang':<6s}  {'R_code':>7s}  {'p':>8s}  {'CI_lo':>7s}  {'CI_hi':>7s}  {'d':>6s}  {'d_match':>8s}")
    print(f"  {'─'*58}")
    for lang in LANGUAGES:
        r = result[lang]
        if r.get("skip"):
            print(f"  {lang:<6s}  (skipped)")
            continue
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"  {lang:<6s}  {r['R_code']:>7.3f}  {r['p_value']:>8.4f}  "
              f"{r['ci_95'][0]:>7.3f}  {r['ci_95'][1]:>7.3f}  "
              f"{r['cohens_d']:>6.3f}  {r['d_match_mean']:>8.4f} {sig}")

    agg = result["aggregate"]
    print(f"  {'agg':<6s}  {agg['R_code']:>7.3f}")

    del model, nl_array, code_array, nl_embeddings, code_embeddings
    gc.collect()

    return {"model": model_name, "label": label, "per_language": result}


def _load_model(model_name, kwargs):
    """Load sentence-transformers model with extra kwargs."""
    from sentence_transformers import SentenceTransformer

    class _WrappedST:
        def __init__(self):
            self._model = SentenceTransformer(model_name, **kwargs)
            self._model_name = model_name

        def encode(self, texts):
            return self._model.encode(texts, normalize_embeddings=True)

        @property
        def name(self):
            return f"st_{self._model_name.split('/')[-1]}"

        @property
        def dimension(self):
            return self._model.get_sentence_embedding_dimension()

    return _WrappedST()


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


def make_figures(all_results: list[dict]):
    """Generate Strategy D figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_models = len(all_results)
    n_langs = len(LANGUAGES)

    # Heatmap: R_code matrix (model × language)
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = np.zeros((n_models, n_langs))
    model_labels = []
    for mi, res in enumerate(all_results):
        model_labels.append(res["label"])
        for li, lang in enumerate(LANGUAGES):
            r = res["per_language"].get(lang, {})
            matrix[mi, li] = r.get("R_code", 0.0)

    import seaborn as sns
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="YlGn",
        xticklabels=LANGUAGES, yticklabels=model_labels,
        vmin=0.9, vmax=max(1.5, matrix.max()),
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Strategy D: Per-Language R_code (NL-Code Alignment)\n"
                 "R_code > 1 = NL closer to matching code than mismatching code")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_d_rcode_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: strategy_d_rcode_heatmap.png")

    # Bar chart: per-language d_match across models
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_langs)
    width = 0.8 / n_models
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for mi, res in enumerate(all_results):
        d_matches = [res["per_language"].get(lang, {}).get("d_match_mean", 0) for lang in LANGUAGES]
        offset = (mi - n_models / 2 + 0.5) * width
        ax.bar(x + offset, d_matches, width, label=res["label"], color=colors[mi % len(colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES)
    ax.set_ylabel("d_match (NL → same code, lower = closer)")
    ax.set_title("Per-Language NL-Code Distance Across Models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_d_dmatch_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: strategy_d_dmatch_bars.png")


def main():
    print("=" * 60)
    print("Strategy D: Enhanced NL-Code Alignment")
    print("Per-Language × Per-Model R_code Matrix")
    print("=" * 60)

    all_results = []
    for model_name, label, kwargs in MODELS:
        result = run_model(model_name, label, kwargs)
        all_results.append(result)

    # Holm-Bonferroni correction across all per-language p-values
    all_p = []
    p_index = []  # (model_idx, lang)
    for mi, res in enumerate(all_results):
        for lang in LANGUAGES:
            r = res["per_language"].get(lang, {})
            if not r.get("skip"):
                all_p.append(r["p_value"])
                p_index.append((mi, lang))

    if all_p:
        corrected = holm_bonferroni(all_p)
        for (mi, lang), p_corr in zip(p_index, corrected):
            all_results[mi]["per_language"][lang]["p_corrected"] = p_corr

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY (Holm-Bonferroni corrected)")
    print(f"{'='*60}")
    print(f"\n{'Model':<25s}", end="")
    for lang in LANGUAGES:
        print(f"  {lang:>6s}", end="")
    print(f"  {'agg':>6s}")
    print(f"{'─'*75}")

    n_supported = 0
    n_total = 0
    for res in all_results:
        print(f"{res['label']:<25s}", end="")
        for lang in LANGUAGES:
            r = res["per_language"].get(lang, {})
            if r.get("skip"):
                print(f"  {'--':>6s}", end="")
            else:
                p = r.get("p_corrected", r["p_value"])
                sig = "*" if p < 0.05 else ""
                print(f"  {r['R_code']:>5.2f}{sig}", end="")
                n_total += 1
                if r["R_code"] > 1.0 and p < 0.05:
                    n_supported += 1
        agg = res["per_language"]["aggregate"]
        print(f"  {agg['R_code']:>5.2f}")

    print(f"\n  R_code > 1 and significant: {n_supported}/{n_total} cells")

    # Figures
    make_figures(all_results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "strategy_d_code_alignment.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
