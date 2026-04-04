#!/usr/bin/env python3
"""DEPRECATED: Strategy 4 prerequisite check (run_strategy4_prereq.py) confirmed
that k-NN accuracy is redundant with distance ratio R (|rho| > 0.7 for MiniLM).
CSLS hubness correction does not flip the comp/judg ordering.

This script is retained for reference only. Do not use results from this script
as independent evidence for P2 failure interpretation.

Original description below:
---
Strategy 4: Aristotelian k-NN reanalysis of P2 failure.

Motivation
----------
P2 FAILED using discriminability ratio R = d_inter / d_intra (a global
distance metric).  Gröger et al. (2026, arXiv:2602.14486) showed that
global geometric convergence (CKA, distance ratios) can be confounded
by linear transforms, but LOCAL neighborhood topology persists.

Hypothesis: the distance ratio R is a flawed metric for convergence.
If we use k-NN accuracy (local topology) instead of R (global geometry),
P2 may be supported: computational operations may have correct local
neighborhoods even if their global distances are larger.

Protocol
--------
For each (operation, language) query point:
  1. Find its k nearest neighbors across ALL 500 points.
  2. Hit = at least one neighbor is the SAME operation in a DIFFERENT
     language.
  3. Aggregate by category (computational vs judgment).

P2-kNN: kNN_accuracy_C > kNN_accuracy_J
  => P2 holds at the topology level; the R-based failure is a metric
     artifact.  Z_sem converges locally even when global geometry diverges.

kNN_accuracy_C <= kNN_accuracy_J
  => the failure is genuine even at the local topology level.

Additional metrics:
  - Mean Reciprocal Rank (MRR) of first correct cross-lingual match
  - Recall@k (fraction of cross-lingual targets found)
  - Neighborhood overlap (Jaccard) across languages
  - Hubness analysis (high-dim distortion diagnostic)
  - Permutation test for statistical significance
"""

import json
import sys
import gc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.metrics import compute_topology_suite
from src.predictions import test_p2_knn, test_p2_cross_lingual_invariance

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

# Models to test (same as P1 scale analysis)
MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384, "118M"),
    ("paraphrase-multilingual-mpnet-base-v2", 768, "278M"),
    ("intfloat/multilingual-e5-large", 1024, "560M"),
]

K_VALUES = [1, 3, 5, 10]


def embed_all(model_name: str) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Embed all operations x languages, return embeddings dict and id lists."""
    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    judg_ids = [op.id for op in ops if op.category == "judgment"]

    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name)

    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}

    del model, embeddings_array
    gc.collect()

    return embeddings, comp_ids, judg_ids


def run_single_model(model_name: str, dim: int, params: str) -> dict:
    """Run Strategy 4 analysis for one model."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name} ({params} params, {dim}d)")
    print(f"{'=' * 60}")

    embeddings, comp_ids, judg_ids = embed_all(model_name)
    all_ids = comp_ids + judg_ids
    categories = {oid: "computational" for oid in comp_ids}
    categories.update({oid: "judgment" for oid in judg_ids})

    # --- Original P2 (distance-based) for comparison ---
    print("\n  [1/3] P2 distance-based (R = d_inter / d_intra)...")
    p2_dist = test_p2_cross_lingual_invariance(embeddings, comp_ids, judg_ids, LANGUAGES)
    R_C = p2_dist.details["R_C"]
    R_J = p2_dist.details["R_J"]
    print(f"    R_C = {R_C:.4f}, R_J = {R_J:.4f}")
    print(f"    P2-distance: {'SUPPORTED' if p2_dist.supported else 'FAILED'} "
          f"(effect={p2_dist.effect_size:.4f}, p={p2_dist.p_value:.4f})")

    # --- P2-kNN (topology-based) ---
    print("\n  [2/3] P2-kNN topology reanalysis...")
    p2_knn = test_p2_knn(
        embeddings, comp_ids, judg_ids, LANGUAGES,
        k_values=K_VALUES, n_permutations=10000,
    )

    print(f"    MRR_C = {p2_knn.details['mrr_C']:.4f}, MRR_J = {p2_knn.details['mrr_J']:.4f}")
    print(f"    Hubness skewness = {p2_knn.details['hubness_skewness']:.3f} "
          f"({'DETECTED' if p2_knn.details['hubness_detected'] else 'not detected'})")
    print()
    print(f"    {'k':>4s}  {'Acc_C':>7s}  {'Acc_J':>7s}  {'Delta':>7s}  {'Baseline':>8s}  Verdict")
    print(f"    {'─' * 50}")
    for k in K_VALUES:
        pk = p2_knn.details["per_k"][k]
        bl = p2_knn.details["random_baseline"][k]
        verdict = "C > J" if pk["supported"] else "J >= C"
        print(f"    {k:>4d}  {pk['accuracy_C']:>7.4f}  {pk['accuracy_J']:>7.4f}  "
              f"{pk['delta_accuracy']:>+7.4f}  {bl:>8.4f}  {verdict}")

    print(f"\n    P2-kNN (k={p2_knn.details['primary_k']}): "
          f"{'SUPPORTED' if p2_knn.supported else 'NOT SUPPORTED'} "
          f"(effect={p2_knn.effect_size:.4f}, p={p2_knn.p_value:.4f})")

    # --- Recall@k ---
    print(f"\n    Recall@k:")
    print(f"    {'k':>4s}  {'Rec_C':>7s}  {'Rec_J':>7s}  {'Delta':>7s}")
    print(f"    {'─' * 30}")
    for k in K_VALUES:
        pk = p2_knn.details["per_k"][k]
        print(f"    {k:>4d}  {pk['recall_C']:>7.4f}  {pk['recall_J']:>7.4f}  "
              f"{pk['delta_recall']:>+7.4f}")

    # --- Neighborhood overlap ---
    print(f"\n    Neighborhood overlap (Jaccard, k=10):")
    no_c = p2_knn.details["neighborhood_overlap_C"]
    no_j = p2_knn.details["neighborhood_overlap_J"]
    print(f"    Computational: {no_c:.4f}")
    print(f"    Judgment:      {no_j:.4f}")
    print(f"    Delta:         {no_c - no_j:+.4f}")

    # --- Full topology suite for detailed output ---
    print("\n  [3/3] Generating detailed topology report...")
    topo = compute_topology_suite(embeddings, all_ids, LANGUAGES, categories, K_VALUES)

    # Interpretation
    print(f"\n  {'=' * 50}")
    if p2_dist.supported and p2_knn.supported:
        interp = "BOTH_SUPPORTED"
        print("  Interpretation: P2 holds at BOTH distance and topology levels.")
    elif not p2_dist.supported and p2_knn.supported:
        interp = "TOPOLOGY_ONLY"
        print("  Interpretation: P2 FAILS at distance level but HOLDS at topology level.")
        print("  => The R-based failure is a metric artifact (Aristotelian view).")
        print("  => Z_sem converges locally; global geometry is confounded.")
    elif p2_dist.supported and not p2_knn.supported:
        interp = "DISTANCE_ONLY"
        print("  Interpretation: P2 holds at distance level but FAILS at topology level.")
        print("  => Unusual; distance metric may be accidentally favorable.")
    else:
        interp = "BOTH_FAILED"
        print("  Interpretation: P2 fails at BOTH distance and topology levels.")
        print("  => The failure is genuine, not a metric artifact.")
    print(f"  {'=' * 50}")

    return {
        "model": model_name,
        "dim": dim,
        "params": params,
        "p2_distance": {
            "supported": p2_dist.supported,
            "R_C": R_C,
            "R_J": R_J,
            "effect_size": p2_dist.effect_size,
            "p_value": p2_dist.p_value,
        },
        "p2_knn": {
            "supported": p2_knn.supported,
            "effect_size": p2_knn.effect_size,
            "p_value": p2_knn.p_value,
            "primary_k": p2_knn.details["primary_k"],
            "per_k": p2_knn.details["per_k"],
            "mrr_C": p2_knn.details["mrr_C"],
            "mrr_J": p2_knn.details["mrr_J"],
            "neighborhood_overlap_C": p2_knn.details["neighborhood_overlap_C"],
            "neighborhood_overlap_J": p2_knn.details["neighborhood_overlap_J"],
            "hubness_skewness": p2_knn.details["hubness_skewness"],
            "hubness_detected": p2_knn.details["hubness_detected"],
            "random_baseline": p2_knn.details["random_baseline"],
            "permutation_ci_95": p2_knn.details["permutation_ci_95"],
        },
        "interpretation": interp,
    }


def plot_results(all_results: list[dict]):
    """Generate Strategy 4 figures."""
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_models = len(all_results)

    # --- Figure 1: Accuracy@k comparison across models ---
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        per_k = res["p2_knn"]["per_k"]
        baseline = res["p2_knn"]["random_baseline"]
        ks = sorted(per_k.keys())
        acc_c = [per_k[k]["accuracy_C"] for k in ks]
        acc_j = [per_k[k]["accuracy_J"] for k in ks]
        bl = [baseline[k] for k in ks]

        ax.plot(ks, acc_c, "o-", color="#2196F3", label="Computational", markersize=8)
        ax.plot(ks, acc_j, "s-", color="#FF5722", label="Judgment", markersize=8)
        ax.plot(ks, bl, "x--", color="gray", label="Random baseline", markersize=6)
        ax.set_xlabel("k")
        ax.set_title(f"{res['model'].split('/')[-1][:30]}\n({res['params']})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("k-NN Cross-Lingual Accuracy")
    fig.suptitle("Strategy 4: Aristotelian k-NN Reanalysis of P2",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy4_knn_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: strategy4_knn_accuracy.png")

    # --- Figure 2: Distance vs Topology comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_labels = [r["model"].split("/")[-1][:20] for r in all_results]
    x = np.arange(n_models)
    width = 0.35

    # Panel A: R (distance)
    R_Cs = [r["p2_distance"]["R_C"] for r in all_results]
    R_Js = [r["p2_distance"]["R_J"] for r in all_results]
    axes[0].bar(x - width / 2, R_Cs, width, color="#2196F3", label="R_C (computational)")
    axes[0].bar(x + width / 2, R_Js, width, color="#FF5722", label="R_J (judgment)")
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Discriminability Ratio R")
    axes[0].set_title("P2 (distance): R = d_inter / d_intra")
    axes[0].legend(fontsize=8)

    # Panel B: kNN accuracy@5 (topology)
    primary_k = all_results[0]["p2_knn"]["primary_k"]
    acc_Cs = [r["p2_knn"]["per_k"][primary_k]["accuracy_C"] for r in all_results]
    acc_Js = [r["p2_knn"]["per_k"][primary_k]["accuracy_J"] for r in all_results]
    bl = [r["p2_knn"]["random_baseline"][primary_k] for r in all_results]

    axes[1].bar(x - width / 2, acc_Cs, width, color="#2196F3", label=f"kNN_C (k={primary_k})")
    axes[1].bar(x + width / 2, acc_Js, width, color="#FF5722", label=f"kNN_J (k={primary_k})")
    axes[1].scatter(x, bl, marker="x", color="gray", s=60, zorder=5, label="Random baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel(f"k-NN Accuracy (k={primary_k})")
    axes[1].set_title(f"P2-kNN (topology): Accuracy@{primary_k}")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)

    fig.suptitle("P2 Failure Reanalysis: Global Distance vs Local Topology",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy4_distance_vs_topology.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: strategy4_distance_vs_topology.png")

    # --- Figure 3: MRR and Neighborhood Overlap ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mrr_Cs = [r["p2_knn"]["mrr_C"] for r in all_results]
    mrr_Js = [r["p2_knn"]["mrr_J"] for r in all_results]
    axes[0].bar(x - width / 2, mrr_Cs, width, color="#2196F3", label="MRR_C")
    axes[0].bar(x + width / 2, mrr_Js, width, color="#FF5722", label="MRR_J")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Reciprocal Rank")
    axes[0].set_title("MRR: Rank of First Correct Cross-Lingual Match")
    axes[0].legend(fontsize=8)

    no_Cs = [r["p2_knn"]["neighborhood_overlap_C"] for r in all_results]
    no_Js = [r["p2_knn"]["neighborhood_overlap_J"] for r in all_results]
    axes[1].bar(x - width / 2, no_Cs, width, color="#2196F3", label="Jaccard_C")
    axes[1].bar(x + width / 2, no_Js, width, color="#FF5722", label="Jaccard_J")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("Neighborhood Overlap (Jaccard)")
    axes[1].set_title("Cross-Lingual Neighborhood Overlap (k=10)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Strategy 4: Auxiliary Topology Metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy4_mrr_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: strategy4_mrr_overlap.png")


def main():
    print("=" * 60)
    print("Strategy 4: Aristotelian k-NN Reanalysis of P2 Failure")
    print("Gröger et al. (2026): local topology > global geometry")
    print("=" * 60)

    all_results = []
    for model_name, dim, params in MODELS:
        result = run_single_model(model_name, dim, params)
        all_results.append(result)

    # Plot
    plot_results(all_results)

    # Cross-model summary
    print(f"\n{'=' * 60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Model':<35s}  {'P2-dist':>8s}  {'P2-kNN':>8s}  Interpretation")
    print(f"{'─' * 75}")
    for r in all_results:
        model_short = r["model"].split("/")[-1][:30]
        dist_v = "PASS" if r["p2_distance"]["supported"] else "FAIL"
        knn_v = "PASS" if r["p2_knn"]["supported"] else "FAIL"
        print(f"{model_short:<35s}  {dist_v:>8s}  {knn_v:>8s}  {r['interpretation']}")

    # Count interpretation patterns
    interps = [r["interpretation"] for r in all_results]
    if all(i == "TOPOLOGY_ONLY" for i in interps):
        print("\n  => CONSISTENT: P2 failure is a metric artifact across all models.")
        print("     Z_sem converges locally (Aristotelian view supported).")
    elif all(i == "BOTH_SUPPORTED" for i in interps):
        print("\n  => P2 holds at both levels; the reanalysis was unnecessary.")
    elif all(i == "BOTH_FAILED" for i in interps):
        print("\n  => P2 failure is genuine at both levels; not a metric artifact.")
    else:
        print("\n  => MIXED results across models; metric sensitivity is model-dependent.")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "strategy4_knn_results.json"

    # Convert numpy types for JSON serialization
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
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
