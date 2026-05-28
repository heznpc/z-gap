#!/usr/bin/env python3
"""Strategy F: OOD NL-Code Alignment (contamination caveat — deferred portion of C1).

The pre-experiment review (2026-05-21) added a contamination caveat to paper
§5.5: the 50 computational operations in Strategy D are all Python stdlib
1-liners (`sorted(lst)`, `max(lst)`, `len(lst)`, ...), so R_code > 1 cannot
be cleanly separated from pretraining memorization. The caveat explicitly
pointed at tier2/tier3 stimuli as the deferred OOD test.

This runner re-executes the Strategy D R_code matrix on those OOD stimuli:

  - tier2_multistep.json:       30 multi-step algorithms (binary_search,
                                merge_sort, quicksort, BFS, DFS, ...)
  - tier3_compositional.json:   20 compositional algorithms (bellman_ford,
                                topological_sort, A*, dynamic programming, ...)

Total: 50 OOD ops × 5 languages = 250 NL stimuli + 50 multi-line code.

Hypothesis structure (decided pre-run, not post-hoc):
  H_strong: R_code > 1 holds across all/most cells -> alignment is not
            primarily memorization-driven; PRH for code is supported beyond
            pretraining co-occurrence statistics.
  H_weak:   R_code drops to ~1 (or below) on OOD -> the contamination caveat
            is empirically confirmed; the Strategy D effect is largely
            memorization. Paper §5.5 contamination paragraph stays as-is.
  H_partial:Some models / languages keep R_code > 1; others do not -> the
            caveat applies to specific model families. Paper updated to
            reflect which models retain OOD alignment.

Uses the same 7-model set, statistical pipeline (permutation n=10k + bootstrap
n=10k + Holm-Bonferroni), and embedding cache as Strategy D, so the OOD
result is directly comparable to the tier1 result without confounds.
"""

from __future__ import annotations

import datetime
import gc
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.code_alignment import compute_per_language_R_code
from src.model_registry import MODELS_7_FROZEN, registry_sha_summary

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "stimuli"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

LANGUAGES = ["en", "ko", "zh", "ar", "es"]
SEED = 42

# Frozen 7-model set with HuggingFace revision SHAs pinned at 2026-05-21.
# See experiments/src/model_registry.py.
MODELS = MODELS_7_FROZEN


def load_ood_stimuli() -> tuple[list[dict], dict[str, str]]:
    """Load tier2 + tier3 OOD operations.

    Returns:
        ops: list of op dicts with `id`, `descriptions` (per-lang), `code`.
        code_equivalents: {op_id: code_text}
    """
    with open(DATA_DIR / "tier2_multistep.json") as f:
        tier2 = json.load(f)
    with open(DATA_DIR / "tier3_compositional.json") as f:
        tier3 = json.load(f)
    ops = tier2 + tier3
    # V12 (review-2026-05-21): assert op_id uniqueness across the two tiers
    # so a future id collision does not silently double-count pairings in
    # compute_per_language_R_code.
    op_ids = [op["id"] for op in ops]
    if len(set(op_ids)) != len(op_ids):
        from collections import Counter
        dups = [k for k, v in Counter(op_ids).items() if v > 1]
        raise ValueError(f"tier2/tier3 op_id collision: {dups}")
    code_equivalents = {op["id"]: op["code"] for op in ops}
    return ops, code_equivalents


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        cummax = max(cummax, adjusted)
        corrected[orig_idx] = min(cummax, 1.0)
    return corrected


def run_model_ood(model_name: str, label: str, kwargs: dict, ops: list, code_equivalents: dict) -> dict:
    """Run per-language R_code on OOD stimuli for one model."""
    print(f"\n{'='*60}")
    print(f"  {label} ({model_name})")
    print(f"{'='*60}")

    op_ids = [op["id"] for op in ops]
    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name, **kwargs)
    print(f"  dim={model.dimension}")

    # Embed NL descriptions (5 langs each)
    nl_texts, nl_keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op["descriptions"].get(lang)
            if desc:
                nl_texts.append(desc)
                nl_keys.append(f"{op['id']}_{lang}")
    nl_array = cache.get_or_compute(model, nl_texts)
    nl_embeddings = {k: nl_array[i] for i, k in enumerate(nl_keys)}

    # Embed code (multi-line Python functions)
    code_texts = [code_equivalents[op_id] for op_id in op_ids if op_id in code_equivalents]
    code_keys = [op_id for op_id in op_ids if op_id in code_equivalents]
    code_array = cache.get_or_compute(model, code_texts)
    code_embeddings = {k: code_array[i] for i, k in enumerate(code_keys)}

    print(f"  {len(nl_embeddings)} NL embeddings, {len(code_embeddings)} code embeddings")

    # Per-language R_code (same statistics as Strategy D)
    print("  Computing per-language R_code (permutation + bootstrap)...")
    result = compute_per_language_R_code(
        nl_embeddings, code_embeddings, op_ids, LANGUAGES,
        n_perm=10000, n_boot=10000, seed=SEED,
    )

    print(f"\n  {'Lang':<6s}  {'R_code':>7s}  {'p':>8s}  {'CI_lo':>7s}  {'CI_hi':>7s}  {'d':>6s}  {'null_R':>7s}")
    print(f"  {'─'*60}")
    for lang in LANGUAGES:
        r = result.get(lang, {})
        if r.get("skip"):
            print(f"  {lang:<6s}  (skipped)")
            continue
        sig = "*" if r["p_value"] < 0.05 else ""
        null_R = r.get("random_baseline_R_mean", 1.0)
        print(f"  {lang:<6s}  {r['R_code']:>7.3f}  {r['p_value']:>8.4f}  "
              f"{r['ci_95'][0]:>7.3f}  {r['ci_95'][1]:>7.3f}  "
              f"{r['cohens_d']:>6.3f}  {null_R:>7.3f} {sig}")
    agg = result.get("aggregate", {})
    print(f"  {'agg':<6s}  {agg.get('R_code', 0):>7.3f}")

    del model, nl_array, code_array, nl_embeddings, code_embeddings
    gc.collect()

    return {"model": model_name, "label": label, "per_language": result}


def make_figure(all_results: list[dict], tier1_compare: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_models = len(all_results)
    n_langs = len(LANGUAGES)
    matrix = np.zeros((n_models, n_langs))
    labels = []
    for mi, res in enumerate(all_results):
        labels.append(res["label"])
        for li, lang in enumerate(LANGUAGES):
            r = res["per_language"].get(lang, {})
            matrix[mi, li] = r.get("R_code", 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="YlGn",
        xticklabels=LANGUAGES, yticklabels=labels,
        vmin=0.9, vmax=max(1.5, matrix.max()),
        linewidths=0.5, ax=ax,
    )
    ax.set_title(
        "Strategy F: OOD R_code (tier2 multi-step + tier3 compositional)\n"
        "Null R from permutation baseline ≈ 1.000"
    )
    fig.tight_layout()
    path = FIGURES_DIR / "strategy_f_ood_rcode_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {path.name}")


def _build_run_meta() -> dict:
    try:
        import sentence_transformers as _st
        st_version = _st.__version__
    except Exception:
        st_version = "unknown"
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"
    return {
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sentence_transformers": st_version,
        "torch": torch_version,
        "numpy": np.__version__,
        "seed": SEED,
        "n_perm": 10000,
        "n_boot": 10000,
        "review_id": "review-2026-05-21",
        "closes": "C1 deferred portion (contamination via OOD stimuli)",
        "model_revisions": registry_sha_summary(),
    }


def main():
    print("=" * 60)
    print("Strategy F: OOD NL-Code Alignment")
    print("(tier2 multi-step + tier3 compositional, 50 ops × 5 langs)")
    print("=" * 60)

    run_meta = _build_run_meta()
    print(f"\n  started_at_utc={run_meta['started_at_utc']}")
    print(f"  python={run_meta['python']}  st={run_meta['sentence_transformers']}  torch={run_meta['torch']}")
    print(f"  seed={run_meta['seed']}  n_perm={run_meta['n_perm']}")

    ops, code_equivalents = load_ood_stimuli()
    print(f"\n  Loaded {len(ops)} OOD ops "
          f"(tier2={sum(1 for o in ops if o['tier']==2)}, "
          f"tier3={sum(1 for o in ops if o['tier']==3)})")
    print(f"  First 3 op ids: {[op['id'] for op in ops[:3]]}")
    print(f"  Code length: mean={np.mean([len(c) for c in code_equivalents.values()]):.0f} chars, "
          f"max={max(len(c) for c in code_equivalents.values())} chars")

    all_results = []
    failed = []
    for model_name, label, kwargs in MODELS:
        try:
            res = run_model_ood(model_name, label, kwargs, ops, code_equivalents)
            all_results.append(res)
        except Exception as exc:  # noqa: BLE001
            err = {
                "model": model_name,
                "label": label,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            failed.append(err)
            print(
                f"\n  [SKIP] {label} ({model_name}) failed: "
                f"{err['error_type']}: {err['error_message']}",
                file=sys.stderr,
            )
            gc.collect()

    # V8 (review-2026-05-21): refuse partial results so paper's "35/35 OOD
    # cells" claim is never silently invalidated by a model dropout.
    import os as _os
    if failed and _os.environ.get("Z_GAP_ALLOW_PARTIAL_RESULTS") != "1":
        print(
            f"\n[FATAL] {len(failed)}/{len(MODELS)} model(s) failed; "
            f"refusing to write partial Strategy F results.\n"
            f"        Failed: {[f['label'] for f in failed]}\n"
            f"        Set Z_GAP_ALLOW_PARTIAL_RESULTS=1 to override.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Holm-Bonferroni across all (model, language) cells
    all_p, p_index = [], []
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
    print("OOD CROSS-MODEL SUMMARY (Holm-Bonferroni corrected)")
    print(f"{'='*60}")
    print(f"\n{'Model':<25s}", end="")
    for lang in LANGUAGES:
        print(f"  {lang:>6s}", end="")
    print(f"  {'agg':>6s}")
    print(f"{'─'*75}")

    n_sig = 0
    n_total = 0
    for res in all_results:
        print(f"{res['label']:<25s}", end="")
        for lang in LANGUAGES:
            r = res["per_language"].get(lang, {})
            if r.get("skip"):
                print(f"  {'--':>6s}", end="")
            else:
                p = r.get("p_corrected", r["p_value"])
                sig = "*" if (p < 0.05 and r["R_code"] > 1.0) else ""
                print(f"  {r['R_code']:>5.2f}{sig}", end="")
                n_total += 1
                if r["R_code"] > 1.0 and p < 0.05:
                    n_sig += 1
        agg = res["per_language"].get("aggregate", {})
        print(f"  {agg.get('R_code', 0):>5.2f}")

    print(f"\n  OOD R_code > 1 and significant: {n_sig}/{n_total} cells")
    print(f"  (Strategy D tier1 baseline: 35/35 cells)")

    # V7 (review-2026-05-21): save BEFORE figures.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_meta["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    run_meta["n_models_attempted"] = len(MODELS)
    run_meta["n_models_succeeded"] = len(all_results)
    run_meta["failed_models"] = failed
    run_meta["n_ood_ops"] = len(ops)
    run_meta["n_cells_significant"] = n_sig
    run_meta["n_cells_total"] = n_total

    out_path = RESULTS_DIR / "strategy_f_ood_alignment.json"
    payload = {"_meta": run_meta, "results": all_results}

    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_convert)
    print(f"\n  Results saved: {out_path}")

    # Figures last (best-effort).
    try:
        make_figure(all_results)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] make_figure failed: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
