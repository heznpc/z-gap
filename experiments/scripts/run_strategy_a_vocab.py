#!/usr/bin/env python3
"""Run Strategy A: Vocabulary mediation analysis for P2.

Tests whether text-level vocabulary features predict per-operation d_intra.
NO embeddings are used for features (avoids circularity).

Usage:
    python experiments/scripts/run_strategy_a_vocab.py
"""

import json
import sys
import gc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.metrics import compute_d_intra
from src.vocab_mediation import (
    compute_text_features,
    attach_d_intra,
    analyze_vocabulary_mediation,
    FEATURE_NAMES,
    BONFERRONI_ALPHA,
)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

# Same model suite as the main P2 experiment
MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384),
    ("intfloat/multilingual-e5-small", 384),
    ("intfloat/multilingual-e5-base", 768),
    ("intfloat/multilingual-e5-large", 1024),
    ("BAAI/bge-m3", 1024),
]


def run_single_model(
    model_name: str,
    expected_dim: int,
    text_features: list[dict],
) -> dict:
    """Run Strategy A for one model.

    text_features are precomputed once (text-only, model-independent).
    Only d_intra depends on the model.
    """
    ops = get_all_operations()
    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name)

    # Embed all operations x languages (reuses cache from main experiment)
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}

    # Compute per-operation d_intra (the DEPENDENT variable)
    all_ids = [op.id for op in ops]
    d_intra_map = compute_d_intra(embeddings, all_ids, LANGUAGES)

    # Deep copy text features and attach d_intra
    import copy
    records = copy.deepcopy(text_features)
    attach_d_intra(records, d_intra_map)

    # Run analysis
    analysis = analyze_vocabulary_mediation(records)

    # Clean up
    del model, embeddings_array, embeddings
    gc.collect()

    return {
        "model": model_name,
        "dim": expected_dim,
        "records": records,
        "analysis": analysis,
    }


def make_figure(all_results: list[dict]):
    """Generate Strategy A diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    primary = all_results[0]
    records = primary["records"]
    analysis = primary["analysis"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    cats = np.array([r["category"] for r in records])
    d_intra = np.array([r["d_intra"] for r in records])

    # ── Panel A: Technical term ratio vs d_intra ──
    ax = axes[0, 0]
    tech = np.array([r["technical_ratio"] for r in records])
    comp_mask = cats == "computational"
    judg_mask = cats == "judgment"

    ax.scatter(tech[comp_mask], d_intra[comp_mask],
               c="#2196F3", alpha=0.6, label="Computational", s=40)
    ax.scatter(tech[judg_mask], d_intra[judg_mask],
               c="#FF5722", alpha=0.6, label="Judgment", s=40)

    pooled = analysis["pooled_correlations"]["technical_ratio"]
    ax.set_xlabel("Technical Term Ratio (English description)")
    ax.set_ylabel("d_intra (cross-lingual distance)")
    sig = "*" if pooled["significant_bonferroni"] else " (ns)"
    ax.set_title(
        f"A: Technical Vocabulary vs d_intra\n"
        f"rho={pooled['rho']:.3f}, p_bonf={pooled['p_bonferroni']:.4f}{sig}"
    )
    ax.legend()

    # ── Panel B: En-Es cognate ratio vs d_intra ──
    ax = axes[0, 1]
    cog = np.array([r["en_es_cognate"] for r in records])

    ax.scatter(cog[comp_mask], d_intra[comp_mask],
               c="#2196F3", alpha=0.6, label="Computational", s=40)
    ax.scatter(cog[judg_mask], d_intra[judg_mask],
               c="#FF5722", alpha=0.6, label="Judgment", s=40)

    pooled_cog = analysis["pooled_correlations"]["en_es_cognate"]
    sig_cog = "*" if pooled_cog["significant_bonferroni"] else " (ns)"
    ax.set_xlabel("En-Es Cognate Ratio")
    ax.set_ylabel("d_intra")
    ax.set_title(
        f"B: Cognate Ratio (en-es) vs d_intra\n"
        f"rho={pooled_cog['rho']:.3f}, p_bonf={pooled_cog['p_bonferroni']:.4f}{sig_cog}"
    )
    ax.legend()

    # ── Panel C: Within-category correlations ──
    ax = axes[1, 0]
    # Show rho for each feature, grouped by within-comp and within-judg
    feat_labels = [f.replace("_", "\n") for f in FEATURE_NAMES]
    x_pos = np.arange(len(FEATURE_NAMES))
    w = 0.35

    rho_comp = [analysis["within_category"]["computational"][f]["rho"]
                for f in FEATURE_NAMES]
    rho_judg = [analysis["within_category"]["judgment"][f]["rho"]
                for f in FEATURE_NAMES]

    ax.bar(x_pos - w/2, rho_comp, w, label="Within Comp (n=50)",
           color="#2196F3", alpha=0.7)
    ax.bar(x_pos + w/2, rho_judg, w, label="Within Judg (n=50)",
           color="#FF5722", alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feat_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Spearman rho (with d_intra)")
    ax.set_title("C: Within-Category Correlations\n(genuine effect = nonzero here)")
    ax.legend(fontsize=8)

    # ── Panel D: Cross-model consistency of pooled rho ──
    ax = axes[1, 1]
    # For each model, show the rho for technical_ratio
    model_names = []
    tech_rhos = []
    tech_pvals = []
    for res in all_results:
        short = res["model"].split("/")[-1][:22]
        model_names.append(short)
        pr = res["analysis"]["pooled_correlations"]["technical_ratio"]
        tech_rhos.append(pr["rho"])
        tech_pvals.append(pr["p_bonferroni"])

    colors = ["#4CAF50" if p < 0.05 else "#9E9E9E" for p in tech_pvals]
    ax.barh(model_names, tech_rhos, color=colors, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Spearman rho (technical_ratio vs d_intra)")
    ax.set_title("D: Cross-Model Consistency\n(green = Bonferroni p < 0.05)")

    fig.suptitle(
        "Strategy A: Vocabulary Mediates the Communicability Gap",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "strategy_a_vocab_mediation.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Figure saved: strategy_a_vocab_mediation.png")


def print_analysis(model_name: str, analysis: dict):
    """Print a concise summary of the analysis for one model."""
    short = model_name.split("/")[-1]
    print(f"\n--- {short} ---")

    # Pooled correlations
    print(f"  Pooled correlations (Bonferroni alpha = {BONFERRONI_ALPHA:.4f}):")
    for feat in FEATURE_NAMES:
        pr = analysis["pooled_correlations"][feat]
        sig = "***" if pr["p_bonferroni"] < 0.001 else \
              "**" if pr["p_bonferroni"] < 0.01 else \
              "*" if pr["p_bonferroni"] < 0.05 else "ns"
        print(f"    {feat:20s}  rho={pr['rho']:+.3f}  "
              f"p_bonf={pr['p_bonferroni']:.4f} [{sig}]  "
              f"CI=[{pr['ci_95'][0]:+.3f}, {pr['ci_95'][1]:+.3f}]")

    # Within-category highlights
    print(f"  Within-computational (n=50):")
    for feat in FEATURE_NAMES:
        wc = analysis["within_category"]["computational"][feat]
        if abs(wc["rho"]) > 0.10:
            print(f"    {feat:20s}  rho={wc['rho']:+.3f}  p={wc['p_raw']:.4f}")

    print(f"  Within-judgment (n=50):")
    for feat in FEATURE_NAMES:
        wj = analysis["within_category"]["judgment"][feat]
        if abs(wj["rho"]) > 0.10:
            print(f"    {feat:20s}  rho={wj['rho']:+.3f}  p={wj['p_raw']:.4f}")

    # Power
    pa = analysis["power_analysis"]
    print(f"  Power: detect |rho| >= {pa['min_detectable_rho_n100']:.3f} (n=100) "
          f"or |rho| >= {pa['min_detectable_rho_n50']:.3f} (n=50)")

    # Interpretation
    interp = analysis["interpretation"]
    if interp["genuine_mediators"]:
        print(f"  Genuine mediators: {', '.join(interp['genuine_mediators'])}")
    if interp["category_proxies"]:
        print(f"  Category proxies only: {', '.join(interp['category_proxies'])}")
    if not interp["genuine_mediators"] and not interp["category_proxies"]:
        print(f"  No significant vocabulary features after Bonferroni correction.")


def main():
    print("=" * 65)
    print("Strategy A: Vocabulary Mediation Analysis")
    print("=" * 65)
    print()
    print("Design notes:")
    print("  - All features from raw text (no embeddings -> no circularity)")
    print("  - No cross-script comparisons (ko/zh/ar never compared as text)")
    print("  - No mediation model (Spearman correlations only)")
    print(f"  - Bonferroni correction: {len(FEATURE_NAMES)} features, "
          f"alpha = {BONFERRONI_ALPHA:.4f}")
    print("  - Within-category analysis to distinguish genuine from confounded")

    # Compute text features ONCE (model-independent)
    ops = get_all_operations()
    text_features = compute_text_features(ops)
    print(f"\n  Computed text features for {len(text_features)} operations")

    all_results = []
    for model_name, dim in MODELS:
        print(f"\n{'─'*50}")
        result = run_single_model(model_name, dim, text_features)
        print_analysis(model_name, result["analysis"])
        all_results.append(result)

    # Generate figure
    make_figure(all_results)

    # Cross-model summary
    print("\n" + "=" * 65)
    print("Cross-Model Summary")
    print("=" * 65)

    # Count how many models show significant pooled correlation for each feature
    for feat in FEATURE_NAMES:
        n_sig = sum(
            1 for r in all_results
            if r["analysis"]["pooled_correlations"][feat]["significant_bonferroni"]
        )
        if n_sig > 0:
            print(f"  {feat}: significant in {n_sig}/{len(all_results)} models")

    # Save results
    serializable = []
    for r in all_results:
        entry = {
            "model": r["model"],
            "dim": r["dim"],
            "analysis": r["analysis"],
            "per_op_summary": [
                {k: v for k, v in rec.items()
                 if k in ("op_id", "category", "d_intra") or k in FEATURE_NAMES}
                for rec in r["records"]
            ],
        }
        serializable.append(entry)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "strategy_a_vocab_mediation.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
