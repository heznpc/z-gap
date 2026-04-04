#!/usr/bin/env python3
"""Strategy B: Language-Pair Decomposition of d_intra.

Motivation
----------
P2 failed (R_C < R_J) across all models.  The core metric d_intra averages
over ALL C(5,2)=10 language pairs, but different pairs have very different
typological distances (en-es is close/Latin, en-ko is far/SOV+agglutinative).
This averaging may mask heterogeneous effects.

This experiment decomposes the existing per-operation, per-language-pair
distances to answer three questions:

  Q1. Is the comp > judg d_intra gap uniform across all 10 language pairs,
      or concentrated in specific pairs?

  Q2. English-pivot hypothesis: is the gap larger for en-X pairs (4 pairs)
      vs non-English pairs (6 pairs)?  If so, English-dominant training
      creates asymmetric alignment.

  Q3. Typological distance gradient: does the comp-judg gap increase
      with typological distance between the pair's languages?

Protocol
--------
- Uses existing embeddings (no new embedding calls).
- Reads per-operation detail from compute_per_operation_detail(), which
  already provides lang_pair_distances for each operation.
- Statistical tests: Mann-Whitney U per pair (Bonferroni-corrected),
  Spearman correlation for typological gradient.

Usage:
    python experiments/scripts/run_strategy2_langpair.py
"""

import json
import sys
import gc
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.metrics import compute_per_operation_detail

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

# Models to analyse (same as Strategy 1 / main P2 suite)
MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384),
    ("intfloat/multilingual-e5-small", 384),
    ("intfloat/multilingual-e5-base", 768),
    ("intfloat/multilingual-e5-large", 1024),
    ("BAAI/bge-m3", 1024),
]

# All C(5,2) = 10 language pairs (deterministic order from itertools)
LANG_PAIRS = [f"{l1}-{l2}" for l1, l2 in combinations(LANGUAGES, 2)]

# Pairs involving English vs not
EN_PAIRS = [p for p in LANG_PAIRS if p.startswith("en-")]
NON_EN_PAIRS = [p for p in LANG_PAIRS if not p.startswith("en-")]

# Approximate typological distance ranking (ordinal, 1 = closest).
# Based on: script family, word order, morphological type, and
# Hammarstrom et al. (2023) structural vector distances.
# Used ONLY for rank-order Spearman; exact values do not matter.
TYPOLOGICAL_RANK = {
    # Canonical pairs from combinations(LANGUAGES, 2) where
    # LANGUAGES = ["en", "ko", "zh", "ar", "es"]:
    #   en-ko, en-zh, en-ar, en-es, ko-zh, ko-ar, ko-es, zh-ar, zh-es, ar-es
    "en-es":  1,   # both SVO, Latin script, Indo-European
    "ko-zh":  2,   # shared CJK cultural sphere, some lexical borrowing
    "en-zh":  3,   # SVO vs SVO (Mandarin), but different script/family
    "zh-ar":  4,   # both head-initial but different script, family
    "en-ar":  5,   # SVO vs VSO, Latin vs Arabic script
    "ar-es":  6,   # historical contact (Andalusia) but different typology
    "ko-es":  7,   # SOV vs SVO, agglutinative vs fusional, no contact
    "en-ko":  8,   # SVO vs SOV, analytic vs agglutinative
    "ko-ar":  9,   # SOV vs VSO, agglutinative vs root-pattern, maximal
    "zh-es": 10,   # different family, script, cultural sphere
}

# Normalise aliases: itertools.combinations always yields sorted pairs
# so "ko-zh" is the canonical form, not "zh-ko".  Double-check.
# LANGUAGES = ["en", "ko", "zh", "ar", "es"], so combinations gives:
#   en-ko, en-zh, en-ar, en-es, ko-zh, ko-ar, ko-es, zh-ar, zh-es, ar-es
# Map each canonical pair to its rank.
PAIR_TYPO_RANK = {}
for pair in LANG_PAIRS:
    if pair in TYPOLOGICAL_RANK:
        PAIR_TYPO_RANK[pair] = TYPOLOGICAL_RANK[pair]
    else:
        # try reversed
        rev = "-".join(reversed(pair.split("-")))
        PAIR_TYPO_RANK[pair] = TYPOLOGICAL_RANK.get(rev, 5)  # fallback mid


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_per_pair_dintra(
    per_op_details: list[dict],
) -> dict[str, dict[str, list[float]]]:
    """Extract per-pair d_intra vectors, grouped by category.

    Returns:
        {pair: {"computational": [d1, d2, ...], "judgment": [d1, d2, ...]}}
    """
    result: dict[str, dict[str, list[float]]] = {
        pair: {"computational": [], "judgment": []} for pair in LANG_PAIRS
    }
    for op in per_op_details:
        cat = op["category"]
        if cat not in ("computational", "judgment"):
            continue
        for pair in LANG_PAIRS:
            d = op["lang_pair_distances"].get(pair)
            if d is not None:
                result[pair][cat].append(d)
    return result


def _mannwhitney_per_pair(
    per_pair: dict[str, dict[str, list[float]]],
) -> dict[str, dict]:
    """Mann-Whitney U test (comp vs judg d_intra) for each language pair.

    Returns per-pair: U statistic, raw p-value, Bonferroni-corrected p,
    medians, means, effect size (rank-biserial r).
    """
    n_tests = len(LANG_PAIRS)
    results = {}
    for pair in LANG_PAIRS:
        comp = np.array(per_pair[pair]["computational"])
        judg = np.array(per_pair[pair]["judgment"])
        if len(comp) < 2 or len(judg) < 2:
            results[pair] = {"skip": True}
            continue

        u_stat, p_raw = stats.mannwhitneyu(comp, judg, alternative="two-sided")
        # Rank-biserial effect size: r = 1 - (2U)/(n1*n2)
        n1, n2 = len(comp), len(judg)
        r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)

        results[pair] = {
            "skip": False,
            "n_comp": n1,
            "n_judg": n2,
            "mean_comp": float(np.mean(comp)),
            "mean_judg": float(np.mean(judg)),
            "median_comp": float(np.median(comp)),
            "median_judg": float(np.median(judg)),
            "gap": float(np.mean(comp) - np.mean(judg)),
            "gap_pct": float((np.mean(comp) - np.mean(judg)) / np.mean(judg) * 100)
                        if np.mean(judg) > 1e-10 else 0.0,
            "U": float(u_stat),
            "p_raw": float(p_raw),
            "p_bonferroni": float(min(p_raw * n_tests, 1.0)),
            "rank_biserial_r": float(r_rb),
            "significant_bonferroni": float(p_raw * n_tests) < 0.05,
        }
    return results


def _english_pivot_test(
    per_pair: dict[str, dict[str, list[float]]],
) -> dict:
    """Compare the comp-judg d_intra gap for en-X pairs vs non-English pairs.

    Aggregates across operations within each group, then runs a
    permutation test on the group-level gap difference.
    """
    # Per-pair gap (comp_mean - judg_mean)
    pair_gaps = {}
    for pair in LANG_PAIRS:
        comp = per_pair[pair]["computational"]
        judg = per_pair[pair]["judgment"]
        if comp and judg:
            pair_gaps[pair] = float(np.mean(comp) - np.mean(judg))

    en_gaps = [pair_gaps[p] for p in EN_PAIRS if p in pair_gaps]
    non_en_gaps = [pair_gaps[p] for p in NON_EN_PAIRS if p in pair_gaps]

    if not en_gaps or not non_en_gaps:
        return {"skip": True}

    mean_en_gap = float(np.mean(en_gaps))
    mean_non_en_gap = float(np.mean(non_en_gaps))
    observed_diff = mean_en_gap - mean_non_en_gap

    # Permutation test: under H0 the en/non-en grouping is arbitrary
    all_gaps = en_gaps + non_en_gaps
    n_en = len(en_gaps)
    rng = np.random.default_rng(42)
    n_perm = 10000
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(all_gaps)
        perm_diffs[i] = np.mean(perm[:n_en]) - np.mean(perm[n_en:])

    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

    # Interpretation
    if abs(observed_diff) < 0.005 or p_value > 0.05:
        interpretation = "UNIFORM: gap is similar for en-X and non-English pairs"
    elif observed_diff > 0:
        interpretation = "ENGLISH_PIVOT: gap is larger for en-X pairs (asymmetric alignment)"
    else:
        interpretation = "ENGLISH_EQUALIZER: gap is smaller for en-X pairs"

    return {
        "skip": False,
        "en_pair_gaps": {p: pair_gaps[p] for p in EN_PAIRS if p in pair_gaps},
        "non_en_pair_gaps": {p: pair_gaps[p] for p in NON_EN_PAIRS if p in pair_gaps},
        "mean_en_gap": mean_en_gap,
        "mean_non_en_gap": mean_non_en_gap,
        "observed_diff": observed_diff,
        "p_value_permutation": p_value,
        "permutation_ci_95": (
            float(np.percentile(perm_diffs, 2.5)),
            float(np.percentile(perm_diffs, 97.5)),
        ),
        "interpretation": interpretation,
    }


def _typological_gradient(
    per_pair: dict[str, dict[str, list[float]]],
) -> dict:
    """Spearman correlation between typological distance rank and comp-judg gap."""
    ranks = []
    gaps = []
    pair_labels = []
    for pair in LANG_PAIRS:
        comp = per_pair[pair]["computational"]
        judg = per_pair[pair]["judgment"]
        if not comp or not judg:
            continue
        ranks.append(PAIR_TYPO_RANK[pair])
        gaps.append(float(np.mean(comp) - np.mean(judg)))
        pair_labels.append(pair)

    if len(ranks) < 4:
        return {"skip": True}

    rho, p = stats.spearmanr(ranks, gaps)

    # Interpretation
    if rho > 0 and p < 0.05:
        interpretation = ("GRADIENT: gap increases with typological distance "
                          "(supports Z stratification / continuous D_train effect)")
    elif rho < 0 and p < 0.05:
        interpretation = ("INVERSE: gap decreases with typological distance "
                          "(closer languages show more divergence)")
    else:
        interpretation = ("FLAT: no significant relationship between typological "
                          "distance and comp-judg gap (gap is uniform)")

    return {
        "skip": False,
        "pairs": pair_labels,
        "typological_ranks": ranks,
        "gaps": gaps,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "interpretation": interpretation,
    }


# ── Single-model runner ─────────────────────────────────────────────

def run_single_model(model_name: str, expected_dim: int) -> dict:
    """Run Strategy B for one embedding model."""
    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    judg_ids = [op.id for op in ops if op.category == "judgment"]
    all_ids = comp_ids + judg_ids
    categories = {op.id: op.category for op in ops}

    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name)

    # Embed all operations x languages (uses cache)
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}

    # Get per-operation detail (contains lang_pair_distances)
    per_op_details = compute_per_operation_detail(
        embeddings, all_ids, LANGUAGES, categories
    )

    # Step 1: extract per-pair d_intra vectors by category
    per_pair = _extract_per_pair_dintra(per_op_details)

    # Step 2: per-pair Mann-Whitney U tests
    mw_results = _mannwhitney_per_pair(per_pair)

    # Step 3: English-pivot hypothesis
    pivot_result = _english_pivot_test(per_pair)

    # Step 4: typological distance gradient
    gradient_result = _typological_gradient(per_pair)

    # Summary statistics
    n_sig = sum(1 for v in mw_results.values()
                if not v.get("skip") and v.get("significant_bonferroni"))
    all_gaps = [v["gap"] for v in mw_results.values() if not v.get("skip")]
    gap_cv = float(np.std(all_gaps) / np.mean(all_gaps)) if all_gaps and np.mean(all_gaps) > 1e-10 else 0.0

    # Clean up
    del model, embeddings_array, embeddings
    gc.collect()

    return {
        "model": model_name,
        "dim": expected_dim,
        "per_pair_tests": mw_results,
        "english_pivot": pivot_result,
        "typological_gradient": gradient_result,
        "summary": {
            "n_pairs_tested": len([v for v in mw_results.values() if not v.get("skip")]),
            "n_significant_bonferroni": n_sig,
            "mean_gap": float(np.mean(all_gaps)) if all_gaps else 0.0,
            "std_gap": float(np.std(all_gaps)) if all_gaps else 0.0,
            "gap_cv": gap_cv,
            "gap_uniform": gap_cv < 0.5,  # heuristic: CV < 50% = fairly uniform
        },
    }


# ── Figures ──────────────────────────────────────────────────────────

def make_figures(all_results: list[dict]):
    """Generate Strategy B diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Use the primary model (MiniLM) for detailed figures
    primary = all_results[0]
    mw = primary["per_pair_tests"]
    pivot = primary["english_pivot"]
    gradient = primary["typological_gradient"]

    # ── Figure 1: 4-panel decomposition ──────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # Panel A: Per-pair comp vs judg mean d_intra (grouped bar)
    ax = fig.add_subplot(gs[0, 0])
    pairs_sorted = sorted(
        [p for p in LANG_PAIRS if not mw[p].get("skip")],
        key=lambda p: mw[p]["gap"],
        reverse=True,
    )
    x = np.arange(len(pairs_sorted))
    width = 0.35
    comp_means = [mw[p]["mean_comp"] for p in pairs_sorted]
    judg_means = [mw[p]["mean_judg"] for p in pairs_sorted]
    ax.bar(x - width / 2, comp_means, width, color="#2196F3", alpha=0.8, label="Computational")
    ax.bar(x + width / 2, judg_means, width, color="#FF5722", alpha=0.8, label="Judgment")
    # Significance stars
    for i, p in enumerate(pairs_sorted):
        if mw[p].get("significant_bonferroni"):
            y_top = max(mw[p]["mean_comp"], mw[p]["mean_judg"]) * 1.02
            ax.text(i, y_top, "*", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs_sorted, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean d_intra")
    ax.set_title("A: Per-Pair d_intra (comp vs judg)\n* = significant after Bonferroni")
    ax.legend(fontsize=9)

    # Panel B: Gap heatmap (10 pairs as a matrix)
    ax = fig.add_subplot(gs[0, 1])
    n_lang = len(LANGUAGES)
    gap_matrix = np.full((n_lang, n_lang), np.nan)
    for i, l1 in enumerate(LANGUAGES):
        for j, l2 in enumerate(LANGUAGES):
            if i == j:
                gap_matrix[i, j] = 0.0
                continue
            # find the canonical pair
            pair = f"{l1}-{l2}" if f"{l1}-{l2}" in mw else f"{l2}-{l1}"
            if pair in mw and not mw[pair].get("skip"):
                gap_matrix[i, j] = mw[pair]["gap"]

    import seaborn as sns
    mask = np.eye(n_lang, dtype=bool)
    vmax = np.nanmax(np.abs(gap_matrix[~mask])) if not np.all(np.isnan(gap_matrix[~mask])) else 0.1
    sns.heatmap(
        gap_matrix, annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, vmin=-vmax, vmax=vmax,
        xticklabels=LANGUAGES, yticklabels=LANGUAGES,
        mask=mask, ax=ax, linewidths=0.5,
    )
    ax.set_title("B: Comp-Judg d_intra Gap by Pair\n(red = comp > judg)")

    # Panel C: English-pivot comparison
    ax = fig.add_subplot(gs[1, 0])
    if not pivot.get("skip"):
        en_gaps = list(pivot["en_pair_gaps"].values())
        non_en_gaps = list(pivot["non_en_pair_gaps"].values())
        bp = ax.boxplot(
            [en_gaps, non_en_gaps],
            labels=["en-X pairs (4)", "non-English pairs (6)"],
            patch_artist=True, widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#9C27B0")
        bp["boxes"][1].set_alpha(0.6)

        # Overlay individual points
        for i, (data, color) in enumerate(zip(
            [en_gaps, non_en_gaps], ["#4CAF50", "#9C27B0"]
        )):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(data))
            ax.scatter(np.full(len(data), i + 1) + jitter, data,
                       color=color, s=50, alpha=0.8, zorder=5, edgecolors="white")

        p_perm = pivot["p_value_permutation"]
        ax.set_ylabel("Comp-Judg d_intra Gap")
        ax.set_title(
            f"C: English-Pivot Hypothesis\n"
            f"(permutation p = {p_perm:.4f})"
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        ax.set_title("C: English-Pivot Hypothesis")

    # Panel D: Typological gradient
    ax = fig.add_subplot(gs[1, 1])
    if not gradient.get("skip"):
        ranks = gradient["typological_ranks"]
        gaps = gradient["gaps"]
        pair_labels = gradient["pairs"]
        ax.scatter(ranks, gaps, s=80, c="#FF9800", edgecolors="black", zorder=5)

        # Label each point
        for r, g, lbl in zip(ranks, gaps, pair_labels):
            ax.annotate(lbl, (r, g), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, alpha=0.8)

        # Regression line
        z = np.polyfit(ranks, gaps, 1)
        x_line = np.linspace(min(ranks), max(ranks), 50)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1)

        rho = gradient["spearman_rho"]
        p_sp = gradient["spearman_p"]
        ax.set_xlabel("Typological Distance Rank (1 = closest)")
        ax.set_ylabel("Comp-Judg d_intra Gap")
        ax.set_title(
            f"D: Typological Gradient\n"
            f"(Spearman rho = {rho:.3f}, p = {p_sp:.4f})"
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        ax.set_title("D: Typological Gradient")

    short_name = primary["model"].split("/")[-1][:30]
    fig.suptitle(
        f"Strategy B: Language-Pair Decomposition of d_intra  ({short_name})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.savefig(
        FIGURES_DIR / "strategy2_langpair_decomposition.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Figure saved: strategy2_langpair_decomposition.png")

    # ── Figure 2: Cross-model consistency ────────────────────────────
    n_models = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: per-pair gap across models (heatmap)
    ax = axes[0]
    model_labels = [r["model"].split("/")[-1][:20] for r in all_results]
    gap_data = np.zeros((n_models, len(LANG_PAIRS)))
    for mi, res in enumerate(all_results):
        for pi, pair in enumerate(LANG_PAIRS):
            info = res["per_pair_tests"].get(pair, {})
            gap_data[mi, pi] = info.get("gap", 0.0)
    sns.heatmap(
        gap_data, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
        xticklabels=LANG_PAIRS, yticklabels=model_labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("A: Comp-Judg Gap by Pair x Model")
    ax.tick_params(axis="x", rotation=45)

    # Panel B: English-pivot effect across models
    ax = axes[1]
    en_means = [r["english_pivot"].get("mean_en_gap", 0) for r in all_results]
    non_en_means = [r["english_pivot"].get("mean_non_en_gap", 0) for r in all_results]
    x = np.arange(n_models)
    width = 0.35
    ax.bar(x - width / 2, en_means, width, color="#4CAF50", alpha=0.7, label="en-X pairs")
    ax.bar(x + width / 2, non_en_means, width, color="#9C27B0", alpha=0.7, label="non-English")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean Comp-Judg Gap")
    ax.set_title("B: English-Pivot Effect Across Models")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)

    # Panel C: Spearman rho (typological gradient) across models
    ax = axes[2]
    rhos = [r["typological_gradient"].get("spearman_rho", 0) for r in all_results]
    p_vals = [r["typological_gradient"].get("spearman_p", 1) for r in all_results]
    colors = ["#4CAF50" if p < 0.05 else "#9E9E9E" for p in p_vals]
    ax.barh(model_labels, rhos, color=colors, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Spearman rho (typo rank vs gap)")
    ax.set_title("C: Typological Gradient\n(green = p < 0.05)")

    fig.suptitle(
        "Strategy B: Cross-Model Consistency",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "strategy2_langpair_crossmodel.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Figure saved: strategy2_langpair_crossmodel.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Strategy B: Language-Pair Decomposition of P2 d_intra")
    print("=" * 65)

    all_results = []
    for model_name, dim in MODELS:
        print(f"\n--- {model_name} ({dim}d) ---")
        result = run_single_model(model_name, dim)

        # Print per-pair summary
        mw = result["per_pair_tests"]
        print(f"\n  {'Pair':<10s}  {'Gap':>8s}  {'Gap%':>7s}  {'U':>8s}  "
              f"{'p_raw':>8s}  {'p_bonf':>8s}  {'r_rb':>6s}  Sig?")
        print(f"  {'─' * 72}")
        for pair in LANG_PAIRS:
            info = mw[pair]
            if info.get("skip"):
                print(f"  {pair:<10s}  (skipped)")
                continue
            sig = "*" if info["significant_bonferroni"] else ""
            print(f"  {pair:<10s}  {info['gap']:>+8.4f}  {info['gap_pct']:>+6.1f}%  "
                  f"{info['U']:>8.0f}  {info['p_raw']:>8.4f}  {info['p_bonferroni']:>8.4f}  "
                  f"{info['rank_biserial_r']:>+6.3f}  {sig}")

        # English-pivot summary
        piv = result["english_pivot"]
        if not piv.get("skip"):
            print(f"\n  English-pivot: en-X gap = {piv['mean_en_gap']:+.4f}, "
                  f"non-en gap = {piv['mean_non_en_gap']:+.4f}, "
                  f"diff = {piv['observed_diff']:+.4f}, "
                  f"p = {piv['p_value_permutation']:.4f}")
            print(f"  => {piv['interpretation']}")

        # Typological gradient summary
        grad = result["typological_gradient"]
        if not grad.get("skip"):
            print(f"\n  Typological gradient: rho = {grad['spearman_rho']:.3f}, "
                  f"p = {grad['spearman_p']:.4f}")
            print(f"  => {grad['interpretation']}")

        # Overall summary
        s = result["summary"]
        print(f"\n  Summary: {s['n_significant_bonferroni']}/{s['n_pairs_tested']} "
              f"pairs significant (Bonferroni)")
        print(f"  Mean gap = {s['mean_gap']:+.4f}, std = {s['std_gap']:.4f}, "
              f"CV = {s['gap_cv']:.2f}")
        uniformity = "UNIFORM" if s["gap_uniform"] else "HETEROGENEOUS"
        print(f"  Gap distribution: {uniformity}")

        all_results.append(result)

    # ── Figures ──
    make_figures(all_results)

    # ── Cross-model summary ──
    print(f"\n{'=' * 65}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'=' * 65}")

    print(f"\n{'Model':<30s}  {'MeanGap':>8s}  {'CV':>6s}  {'Uniform':>7s}  "
          f"{'Pivot':>10s}  {'Gradient':>10s}")
    print(f"{'─' * 80}")
    for r in all_results:
        short = r["model"].split("/")[-1][:28]
        s = r["summary"]
        piv_interp = r["english_pivot"].get("interpretation", "N/A")[:10]
        grad_rho = r["typological_gradient"].get("spearman_rho", 0)
        grad_p = r["typological_gradient"].get("spearman_p", 1)
        grad_str = f"rho={grad_rho:+.2f}" if grad_p < 0.05 else "n.s."
        print(f"{short:<30s}  {s['mean_gap']:>+8.4f}  {s['gap_cv']:>6.2f}  "
              f"{'YES' if s['gap_uniform'] else 'NO':>7s}  "
              f"{piv_interp:>10s}  {grad_str:>10s}")

    # ── Narrative integration ──
    # Count consistent patterns across models
    n_uniform = sum(1 for r in all_results if r["summary"]["gap_uniform"])
    n_gradient = sum(1 for r in all_results
                     if r["typological_gradient"].get("spearman_p", 1) < 0.05
                     and r["typological_gradient"].get("spearman_rho", 0) > 0)
    n_pivot = sum(1 for r in all_results
                  if "PIVOT" in r["english_pivot"].get("interpretation", ""))

    print(f"\n  Narrative integration:")
    if n_uniform >= len(all_results) * 0.6:
        print("  => Gap is UNIFORM across language pairs in most models.")
        print("     'The communicability gap is a robust property of computational")
        print("      vocabulary, independent of language-pair typological distance.'")
    elif n_gradient >= len(all_results) * 0.6:
        print("  => Gap INCREASES with typological distance in most models.")
        print("     'The communicability gap is modulated by linguistic distance,")
        print("      consistent with D_train dependence and Z_proc cultural mediation.'")
    elif n_pivot >= len(all_results) * 0.6:
        print("  => English-pivot effect dominates in most models.")
        print("     'English-dominant training creates asymmetric cross-lingual")
        print("      alignment that inflates d_intra for en-X pairs.'")
    else:
        print("  => MIXED pattern: no single narrative dominates across models.")
        print("     Report model-dependent decomposition as supplementary analysis.")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "strategy2_langpair_results.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
