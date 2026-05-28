#!/usr/bin/env python3
"""Strategy E: Multi-Model P3 Cross-Lingual Probing.

Closes M5 from the 2026-05-21 pre-experiment review. Extends the paper's
P3 result (originally MiniLM-L12 only; see paper/main.tex §5.5 "P3 Results")
to the 7-model set used in Strategy D so the Z_sem stratification claim
no longer rests on a single model.

For each of the 7 models:
  - Embed all 100 operations × 5 languages (uses EmbeddingCache for hits)
  - Train LogisticRegression on English embeddings:
      Probe 1: category (computational vs judgment, chance 50%)
      Probe 2: operation identity (100-way, chance 1%)
  - Test cross-lingual transfer accuracy on each non-English language
  - Compute binomial p-values against chance for each cell

Outputs:
  - results/strategy_e_multimodel_probing.json (full per-cell data + meta)
  - results/figures/strategy_e_category_heatmap.png
  - results/figures/strategy_e_operation_heatmap.png

Run-meta (review-2026-05-21 pattern): timestamp, python/torch/st versions,
seed, per-model success/failure with try/except wrap.
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

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.model_registry import MODELS_7_FROZEN, registry_sha_summary

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

# Frozen 7-model set with HuggingFace revision SHAs pinned at 2026-05-21.
# See experiments/src/model_registry.py.
MODELS = MODELS_7_FROZEN

# Random seed mirrors Strategy D for cross-experiment consistency
SEED = 42


def _binomial_p_vs_chance(n_correct: int, n_total: int, p_chance: float) -> float:
    """One-sided binomial test: P(X >= n_correct | n_total, p_chance)."""
    from scipy import stats as scipy_stats
    return float(scipy_stats.binomtest(n_correct, n_total, p=p_chance, alternative="greater").pvalue)


def run_model_probing(model_name: str, label: str, kwargs: dict) -> dict:
    """Run P3 (category + operation) probes for one model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print(f"\n{'='*60}")
    print(f"  {label} ({model_name})")
    print(f"{'='*60}")

    ops = get_all_operations()
    categories = {op.id: op.category for op in ops}
    all_ids = [op.id for op in ops]

    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name, **kwargs)
    print(f"  dim={model.dimension}")

    # Embed all 100 ops × 5 langs (cache hits if Strategy D / earlier P3 ran)
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}
    print(f"  {len(embeddings)} NL embeddings ready ({len(ops)} ops × {len(LANGUAGES)} langs)")

    # V11 (review-2026-05-21): guard against missing categories. The
    # original `categories[op_id]` raised KeyError on any op without a
    # category field, which the outer try/except silently classified as a
    # whole-model failure. We now skip the op explicitly and surface a
    # warning so the failure mode is visible.
    def _label(op_id: str) -> int | None:
        cat = categories.get(op_id)
        if cat is None:
            return None
        if cat not in ("computational", "judgment"):
            return None
        return 1 if cat == "computational" else 0

    skipped_ops_train = []
    # --- Probe 1: category (chance 50%) ---
    X_train, y_train = [], []
    for op_id in all_ids:
        key = f"{op_id}_en"
        if key in embeddings:
            lbl = _label(op_id)
            if lbl is None:
                skipped_ops_train.append(op_id)
                continue
            X_train.append(embeddings[key])
            y_train.append(lbl)
    if skipped_ops_train:
        print(f"  [WARN] skipped {len(skipped_ops_train)} train ops with unknown category: "
              f"{skipped_ops_train[:5]}{'...' if len(skipped_ops_train) > 5 else ''}",
              file=sys.stderr)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if len(X_train) == 0:
        raise RuntimeError("no labeled training samples — every op had an unknown category")

    clf_cat = LogisticRegression(max_iter=2000, random_state=SEED, C=1.0)
    clf_cat.fit(X_train, y_train)
    cat_results = {}
    for lang in LANGUAGES:
        X_test, y_test = [], []
        for op_id in all_ids:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                lbl = _label(op_id)
                if lbl is None:
                    continue
                X_test.append(embeddings[key])
                y_test.append(lbl)
        # V11: guard empty test set so the script reports it instead of
        # crashing on `clf.predict(np.array([]))`.
        if not X_test:
            cat_results[lang] = {
                "accuracy": float("nan"),
                "n_correct": 0, "n_total": 0,
                "p_value_vs_chance": float("nan"),
                "skip": True,
            }
            continue
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        preds = clf_cat.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        n_correct = int(np.sum(preds == y_test))
        n_total = int(len(y_test))
        cat_results[lang] = {
            "accuracy": acc,
            "n_correct": n_correct,
            "n_total": n_total,
            "p_value_vs_chance": _binomial_p_vs_chance(n_correct, n_total, 0.5),
        }

    _non_en_cat = [r["accuracy"] for lang, r in cat_results.items()
                   if lang != "en" and not r.get("skip")]
    cat_transfer = float(np.nanmean(_non_en_cat)) if _non_en_cat else float("nan")

    # --- Probe 2: operation identity (chance 1%) ---
    op_to_idx = {op_id: i for i, op_id in enumerate(all_ids)}
    X_train2, y_train2 = [], []
    for op_id in all_ids:
        key = f"{op_id}_en"
        if key in embeddings:
            X_train2.append(embeddings[key])
            y_train2.append(op_to_idx[op_id])
    X_train2 = np.array(X_train2)
    y_train2 = np.array(y_train2)

    clf_op = LogisticRegression(max_iter=3000, random_state=SEED, C=1.0)
    clf_op.fit(X_train2, y_train2)
    op_results = {}
    chance_op = 1.0 / len(all_ids)
    for lang in LANGUAGES:
        X_test, y_test = [], []
        for op_id in all_ids:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                X_test.append(embeddings[key])
                y_test.append(op_to_idx[op_id])
        # V11: guard empty test set.
        if not X_test:
            op_results[lang] = {
                "accuracy": float("nan"),
                "n_correct": 0, "n_total": 0,
                "p_value_vs_chance": float("nan"),
                "skip": True,
            }
            continue
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        preds = clf_op.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        n_correct = int(np.sum(preds == y_test))
        n_total = int(len(y_test))
        op_results[lang] = {
            "accuracy": acc,
            "n_correct": n_correct,
            "n_total": n_total,
            "p_value_vs_chance": _binomial_p_vs_chance(n_correct, n_total, chance_op),
        }
    _non_en_op = [r["accuracy"] for lang, r in op_results.items()
                  if lang != "en" and not r.get("skip")]
    op_transfer = float(np.nanmean(_non_en_op)) if _non_en_op else float("nan")

    # Print
    print(f"\n  Probe 1 (category, chance 50%):")
    print(f"  {'Lang':<6s}  {'acc':>6s}  {'n_correct/n_total':>18s}  {'p_vs_chance':>12s}")
    print(f"  {'─'*48}")
    for lang in LANGUAGES:
        r = cat_results[lang]
        marker = "(train)" if lang == "en" else ""
        print(f"  {lang:<6s}  {r['accuracy']:>6.3f}  {r['n_correct']:>9d}/{r['n_total']:<8d}  {r['p_value_vs_chance']:>12.4g} {marker}")
    print(f"  mean transfer (non-en): {cat_transfer:.3f}")

    print(f"\n  Probe 2 (operation 100-way, chance 1%):")
    print(f"  {'Lang':<6s}  {'acc':>6s}  {'n_correct/n_total':>18s}  {'p_vs_chance':>12s}")
    print(f"  {'─'*48}")
    for lang in LANGUAGES:
        r = op_results[lang]
        marker = "(train)" if lang == "en" else ""
        print(f"  {lang:<6s}  {r['accuracy']:>6.3f}  {r['n_correct']:>9d}/{r['n_total']:<8d}  {r['p_value_vs_chance']:>12.4g} {marker}")
    print(f"  mean transfer (non-en): {op_transfer:.3f}")

    result = {
        "model": model_name,
        "label": label,
        "dim": int(model.dimension),
        "category_probe": {
            "per_language": cat_results,
            "mean_transfer": cat_transfer,
        },
        "operation_probe": {
            "per_language": op_results,
            "mean_transfer": op_transfer,
        },
    }

    del model, embeddings_array, embeddings
    gc.collect()
    return result


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
    try:
        import sklearn
        skl_version = sklearn.__version__
    except Exception:
        skl_version = "unknown"
    return {
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sentence_transformers": st_version,
        "torch": torch_version,
        "sklearn": skl_version,
        "numpy": np.__version__,
        "seed": SEED,
        "review_id": "review-2026-05-21",
        "closes": "M5 (multi-model P3 probing)",
        "model_revisions": registry_sha_summary(),
    }


def make_heatmaps(all_results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_models = len(all_results)
    n_langs = len(LANGUAGES)

    for probe_name, probe_key, vmin, vmax, chance_line in [
        ("category", "category_probe", 0.5, 1.0, 0.5),
        ("operation", "operation_probe", 0.0, 1.0, 0.01),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        matrix = np.zeros((n_models, n_langs))
        labels = []
        for mi, res in enumerate(all_results):
            labels.append(res["label"])
            for li, lang in enumerate(LANGUAGES):
                # V11: per_language may have been skipped (empty test set);
                # fall back to NaN so seaborn shows a blank cell instead of
                # KeyError on a missing key.
                cell = res[probe_key]["per_language"].get(lang, {})
                matrix[mi, li] = cell.get("accuracy", float("nan"))
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="YlGn",
            xticklabels=LANGUAGES, yticklabels=labels,
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax,
        )
        ax.set_title(
            f"Strategy E: {probe_name.capitalize()} probe — cross-lingual transfer accuracy\n"
            f"Train on English; chance = {chance_line:.2f}"
        )
        fig.tight_layout()
        path = FIGURES_DIR / f"strategy_e_{probe_name}_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {path.name}")


def main():
    print("=" * 60)
    print("Strategy E: Multi-Model P3 Cross-Lingual Probing")
    print(f"({len(MODELS)} models × {len(LANGUAGES)} languages)")
    print("=" * 60)

    run_meta = _build_run_meta()
    print(f"\n  started_at_utc={run_meta['started_at_utc']}")
    print(f"  python={run_meta['python']}  st={run_meta['sentence_transformers']}  sklearn={run_meta['sklearn']}")
    print(f"  seed={run_meta['seed']}")

    all_results = []
    failed_models = []
    for model_name, label, kwargs in MODELS:
        try:
            res = run_model_probing(model_name, label, kwargs)
            all_results.append(res)
        except Exception as exc:  # noqa: BLE001
            err = {
                "model": model_name,
                "label": label,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            failed_models.append(err)
            print(
                f"\n  [SKIP] {label} ({model_name}) failed: "
                f"{err['error_type']}: {err['error_message']}",
                file=sys.stderr,
            )
            gc.collect()

    # V8 (review-2026-05-21): same partial-success guard as Strategy D.
    import os as _os
    if failed_models and _os.environ.get("Z_GAP_ALLOW_PARTIAL_RESULTS") != "1":
        print(
            f"\n[FATAL] {len(failed_models)}/{len(MODELS)} model(s) failed; "
            f"refusing to write partial Strategy E results.\n"
            f"        Failed: {[f['label'] for f in failed_models]}\n"
            f"        Set Z_GAP_ALLOW_PARTIAL_RESULTS=1 to override.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL P3 SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<25s}  {'cat_en':>7s}  {'cat_transfer':>12s}  {'op_en':>6s}  {'op_transfer':>12s}")
    print(f"{'─'*70}")
    for res in all_results:
        cat_en = res["category_probe"]["per_language"]["en"]["accuracy"]
        cat_xfer = res["category_probe"]["mean_transfer"]
        op_en = res["operation_probe"]["per_language"]["en"]["accuracy"]
        op_xfer = res["operation_probe"]["mean_transfer"]
        print(f"{res['label']:<25s}  {cat_en:>7.3f}  {cat_xfer:>12.3f}  {op_en:>6.3f}  {op_xfer:>12.3f}")

    # V7 (review-2026-05-21): save BEFORE figures so a matplotlib failure
    # does not lose the probing results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_meta["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    run_meta["n_models_attempted"] = len(MODELS)
    run_meta["n_models_succeeded"] = len(all_results)
    run_meta["failed_models"] = failed_models

    out_path = RESULTS_DIR / "strategy_e_multimodel_probing.json"
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

    try:
        make_heatmaps(all_results)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] make_heatmaps failed: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
