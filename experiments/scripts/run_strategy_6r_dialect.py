#!/usr/bin/env python3
"""Strategy 6-R: Minimal Dialect Evidence (paraphrase baseline + English dialects).

Tests the continuum prediction:
    d_paraphrase < d_dialect < d_cross_lingual

Using English dialect pairs (American vs British, American vs Indian English)
and paraphrase baselines, with existing cross-lingual distances.

Usage:
    python experiments/scripts/run_strategy_6r_dialect.py
"""

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"
DATA_DIR = ROOT / "data"

MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384),
    ("intfloat/multilingual-e5-large", 1024),
    ("BAAI/bge-m3", 1024),
]

DIALECT_PAIRS = [
    ("original", "british", "Am→Br"),
    ("original", "indian", "Am→In"),
]


def cosine_dist(a, b):
    from scipy.spatial.distance import cosine
    return float(cosine(a, b))


def load_dialect_stimuli():
    with open(DATA_DIR / "dialect_stimuli.json") as f:
        return json.load(f)


def run_single_model(model_name: str, dim: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  {model_name} ({dim}d)")
    print(f"{'='*60}")

    ops = get_all_operations()
    stimuli = load_dialect_stimuli()
    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name)

    op_ids = list(stimuli.keys())
    categories = {op.id: op.category for op in ops}

    # --- Step 1: Embed original descriptions (all languages, reuse cache) ---
    all_texts, all_keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                all_texts.append(desc)
                all_keys.append(f"{op.id}_{lang}")

    all_array = cache.get_or_compute(model, all_texts)
    embeddings = {k: all_array[i] for i, k in enumerate(all_keys)}

    # --- Step 2: Embed paraphrases + dialect variants ---
    extra_texts, extra_keys = [], []
    for op_id, stim in stimuli.items():
        for i, para in enumerate(stim["paraphrases"]):
            extra_texts.append(para)
            extra_keys.append(f"{op_id}_para_{i}")
        extra_texts.append(stim["british"])
        extra_keys.append(f"{op_id}_british")
        extra_texts.append(stim["indian"])
        extra_keys.append(f"{op_id}_indian")

    extra_array = cache.get_or_compute(model, extra_texts)
    for i, k in enumerate(extra_keys):
        embeddings[k] = extra_array[i]

    print(f"  Total embeddings: {len(embeddings)}")

    # --- Step 3: Compute three distance levels ---
    d_paraphrase = []  # same op, same language, different phrasing
    d_dialect_british = []   # same op, American vs British
    d_dialect_indian = []    # same op, American vs Indian
    d_cross_lingual = []     # same op, different languages

    for op_id in op_ids:
        orig_key = f"{op_id}_en"
        if orig_key not in embeddings:
            continue
        orig_vec = embeddings[orig_key]

        # d_paraphrase: original vs each paraphrase
        for i in range(3):
            para_key = f"{op_id}_para_{i}"
            if para_key in embeddings:
                d_paraphrase.append(cosine_dist(orig_vec, embeddings[para_key]))

        # paraphrase-paraphrase pairs
        para_vecs = [embeddings[f"{op_id}_para_{i}"] for i in range(3)
                     if f"{op_id}_para_{i}" in embeddings]
        for a, b in combinations(para_vecs, 2):
            d_paraphrase.append(cosine_dist(a, b))

        # d_dialect: American vs British/Indian
        br_key = f"{op_id}_british"
        in_key = f"{op_id}_indian"
        if br_key in embeddings:
            d_dialect_british.append(cosine_dist(orig_vec, embeddings[br_key]))
        if in_key in embeddings:
            d_dialect_indian.append(cosine_dist(orig_vec, embeddings[in_key]))

        # d_cross_lingual: English vs other languages (same op)
        for lang in LANGUAGES:
            if lang == "en":
                continue
            other_key = f"{op_id}_{lang}"
            if other_key in embeddings:
                d_cross_lingual.append(cosine_dist(orig_vec, embeddings[other_key]))

    d_para_arr = np.array(d_paraphrase)
    d_br_arr = np.array(d_dialect_british)
    d_in_arr = np.array(d_dialect_indian)
    d_dial_arr = np.concatenate([d_br_arr, d_in_arr])  # combined dialect
    d_cross_arr = np.array(d_cross_lingual)

    # --- Step 4: Bootstrap tests ---
    n_boot = 10000
    rng = np.random.default_rng(42)

    def bootstrap_diff(a, b, n=n_boot):
        diffs = []
        for _ in range(n):
            ba = rng.choice(a, size=len(a), replace=True)
            bb = rng.choice(b, size=len(b), replace=True)
            diffs.append(np.mean(bb) - np.mean(ba))
        diffs = np.array(diffs)
        p = float(np.mean(diffs <= 0))
        ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        return p, ci

    # Test 1: d_dialect > d_paraphrase
    p1, ci1 = bootstrap_diff(d_para_arr, d_dial_arr)
    # Test 2: d_cross_lingual > d_dialect
    p2, ci2 = bootstrap_diff(d_dial_arr, d_cross_arr)

    # Per-category analysis
    comp_ids = [oid for oid in op_ids if categories.get(oid) == "computational"]
    judg_ids = [oid for oid in op_ids if categories.get(oid) == "judgment"]

    def per_cat_distances(id_list):
        dp, dd, dc = [], [], []
        for op_id in id_list:
            orig_key = f"{op_id}_en"
            if orig_key not in embeddings:
                continue
            orig_vec = embeddings[orig_key]
            for i in range(3):
                pk = f"{op_id}_para_{i}"
                if pk in embeddings:
                    dp.append(cosine_dist(orig_vec, embeddings[pk]))
            bk = f"{op_id}_british"
            ik = f"{op_id}_indian"
            if bk in embeddings:
                dd.append(cosine_dist(orig_vec, embeddings[bk]))
            if ik in embeddings:
                dd.append(cosine_dist(orig_vec, embeddings[ik]))
            for lang in LANGUAGES:
                if lang == "en":
                    continue
                ok = f"{op_id}_{lang}"
                if ok in embeddings:
                    dc.append(cosine_dist(orig_vec, embeddings[ok]))
        return np.array(dp), np.array(dd), np.array(dc)

    comp_dp, comp_dd, comp_dc = per_cat_distances(comp_ids)
    judg_dp, judg_dd, judg_dc = per_cat_distances(judg_ids)

    # Print results
    print(f"\n  Distance Level        Mean      Std      N")
    print(f"  {'─'*50}")
    print(f"  d_paraphrase       {np.mean(d_para_arr):.5f}  {np.std(d_para_arr):.5f}  {len(d_para_arr)}")
    print(f"  d_dialect(British) {np.mean(d_br_arr):.5f}  {np.std(d_br_arr):.5f}  {len(d_br_arr)}")
    print(f"  d_dialect(Indian)  {np.mean(d_in_arr):.5f}  {np.std(d_in_arr):.5f}  {len(d_in_arr)}")
    print(f"  d_dialect(combined){np.mean(d_dial_arr):.5f}  {np.std(d_dial_arr):.5f}  {len(d_dial_arr)}")
    print(f"  d_cross_lingual    {np.mean(d_cross_arr):.5f}  {np.std(d_cross_arr):.5f}  {len(d_cross_arr)}")

    continuum = np.mean(d_para_arr) < np.mean(d_dial_arr) < np.mean(d_cross_arr)
    print(f"\n  Continuum holds: {continuum}")
    print(f"  d_para < d_dialect:       p={p1:.4f}  CI=[{ci1[0]:+.5f}, {ci1[1]:+.5f}]")
    print(f"  d_dialect < d_cross:      p={p2:.4f}  CI=[{ci2[0]:+.5f}, {ci2[1]:+.5f}]")
    print(f"  Ratios: d_dial/d_para = {np.mean(d_dial_arr)/np.mean(d_para_arr):.2f}x, "
          f"d_cross/d_dial = {np.mean(d_cross_arr)/np.mean(d_dial_arr):.2f}x")

    print(f"\n  Per-category:")
    for label, dp, dd, dc in [("Computational", comp_dp, comp_dd, comp_dc),
                               ("Judgment", judg_dp, judg_dd, judg_dc)]:
        holds = np.mean(dp) < np.mean(dd) < np.mean(dc) if len(dp) and len(dd) and len(dc) else False
        print(f"    {label}: d_para={np.mean(dp):.5f} → d_dial={np.mean(dd):.5f} → d_cross={np.mean(dc):.5f}  [{holds}]")

    return {
        "model": model_name, "dim": dim,
        "distances": {
            "d_paraphrase": {"mean": float(np.mean(d_para_arr)), "std": float(np.std(d_para_arr)), "n": len(d_para_arr)},
            "d_dialect_british": {"mean": float(np.mean(d_br_arr)), "std": float(np.std(d_br_arr)), "n": len(d_br_arr)},
            "d_dialect_indian": {"mean": float(np.mean(d_in_arr)), "std": float(np.std(d_in_arr)), "n": len(d_in_arr)},
            "d_dialect_combined": {"mean": float(np.mean(d_dial_arr)), "std": float(np.std(d_dial_arr)), "n": len(d_dial_arr)},
            "d_cross_lingual": {"mean": float(np.mean(d_cross_arr)), "std": float(np.std(d_cross_arr)), "n": len(d_cross_arr)},
        },
        "tests": {
            "para_vs_dialect": {"p": p1, "ci": ci1},
            "dialect_vs_cross": {"p": p2, "ci": ci2},
        },
        "continuum_holds": continuum,
        "per_category": {
            "computational": {"d_para": float(np.mean(comp_dp)), "d_dial": float(np.mean(comp_dd)), "d_cross": float(np.mean(comp_dc))},
            "judgment": {"d_para": float(np.mean(judg_dp)), "d_dial": float(np.mean(judg_dd)), "d_cross": float(np.mean(judg_dc))},
        },
    }


def make_figure(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n = len(all_results)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        d = res["distances"]
        labels = ["Paraphrase", "Dialect\n(British)", "Dialect\n(Indian)", "Cross-\nlingual"]
        means = [d["d_paraphrase"]["mean"], d["d_dialect_british"]["mean"],
                 d["d_dialect_indian"]["mean"], d["d_cross_lingual"]["mean"]]
        stds = [d["d_paraphrase"]["std"], d["d_dialect_british"]["std"],
                d["d_dialect_indian"]["std"], d["d_cross_lingual"]["std"]]
        colors = ["#4CAF50", "#FF9800", "#FF5722", "#2196F3"]

        bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{m:.4f}", ha="center", fontsize=8)
        ax.set_ylabel("Mean Cosine Distance")
        short = res["model"].split("/")[-1][:25]
        holds = "HOLDS" if res["continuum_holds"] else "FAILS"
        ax.set_title(f"{short}\nContinuum: {holds}")

    fig.suptitle("Strategy 6-R: Linguistic Distance Continuum\n"
                 "d_paraphrase < d_dialect < d_cross_lingual",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_6r_dialect_continuum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: strategy_6r_dialect_continuum.png")


def main():
    print("=" * 60)
    print("Strategy 6-R: Minimal Dialect Evidence")
    print("English Paraphrase + Dialect Continuum")
    print("=" * 60)

    all_results = []
    for model_name, dim in MODELS:
        result = run_single_model(model_name, dim)
        all_results.append(result)

    # Cross-model summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<30s}  {'d_para':>7s}  {'d_dial':>7s}  {'d_cross':>7s}  {'Holds':>5s}")
    print(f"{'─'*65}")
    for r in all_results:
        d = r["distances"]
        short = r["model"].split("/")[-1][:28]
        holds = "YES" if r["continuum_holds"] else "NO"
        print(f"{short:<30s}  {d['d_paraphrase']['mean']:>7.4f}  "
              f"{d['d_dialect_combined']['mean']:>7.4f}  "
              f"{d['d_cross_lingual']['mean']:>7.4f}  {holds:>5s}")

    make_figure(all_results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "strategy_6r_dialect_results.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    # Strip non-serializable fields
    clean = []
    for r in all_results:
        clean.append({k: v for k, v in r.items()})
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2, default=_convert)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
