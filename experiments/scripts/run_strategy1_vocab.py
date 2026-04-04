#!/usr/bin/env python3
"""DEPRECATED: Superseded by run_strategy_a_vocab.py (Strategy A).

This script uses vocab_internationality.py which has known flaws:
  - English-pivot cosine creates circularity with d_intra
  - Token overlap is meaningless for non-Latin script pairs
  - Baron & Kenny mediation is inappropriate for n=100
See AUDIT_P2_STRATEGIES.md for details.

Use run_strategy_a_vocab.py instead.
"""
raise SystemExit(
    "DEPRECATED: This script has known flaws. Use run_strategy_a_vocab.py instead.\n"
    "See AUDIT_P2_STRATEGIES.md for details."
)

import json
import sys
import gc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.vocab_internationality import (
    compute_internationality_scores,
    analyze_strategy1,
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


def run_single_model(model_name: str, expected_dim: int) -> dict:
    """Run Strategy 1 analysis for one model."""
    ops = get_all_operations()
    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name)

    # Embed all operations x languages
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}

    # Compute internationality scores
    scores = compute_internationality_scores(ops, embeddings, LANGUAGES)

    # Run statistical analysis
    analysis = analyze_strategy1(scores)

    # Clean up
    del model, embeddings_array, embeddings
    gc.collect()

    return {
        "model": model_name,
        "dim": expected_dim,
        "per_operation_scores": scores,
        "analysis": analysis,
    }


def make_figure(all_results: list[dict]):
    """Generate Strategy 1 diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Use the first model for the detailed scatter plot
    primary = all_results[0]
    scores = primary["per_operation_scores"]

    intl = [s["internationality"] for s in scores]
    d_intra = [s["d_intra"] for s in scores]
    cats = [s["category"] for s in scores]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel A: Scatter (internationality vs d_intra) ──
    ax = axes[0, 0]
    comp_x = [intl[i] for i in range(len(scores)) if cats[i] == "computational"]
    comp_y = [d_intra[i] for i in range(len(scores)) if cats[i] == "computational"]
    judg_x = [intl[i] for i in range(len(scores)) if cats[i] == "judgment"]
    judg_y = [d_intra[i] for i in range(len(scores)) if cats[i] == "judgment"]

    ax.scatter(comp_x, comp_y, c="#2196F3", alpha=0.6, label="Computational", s=40)
    ax.scatter(judg_x, judg_y, c="#FF5722", alpha=0.6, label="Judgment", s=40)

    # Regression line
    z = np.polyfit(intl, d_intra, 1)
    x_line = np.linspace(min(intl), max(intl), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1)

    r_sp = primary["analysis"]["correlation"]["spearman_r"]
    p_sp = primary["analysis"]["correlation"]["spearman_p"]
    ax.set_xlabel("Vocabulary Internationality Score (z-scored composite)")
    ax.set_ylabel("d_intra (cross-lingual distance)")
    ax.set_title(f"A: Internationality vs d_intra\n(rho={r_sp:.3f}, p={p_sp:.4f})")
    ax.legend()

    # ── Panel B: Group comparison (comp vs judg internationality) ──
    ax = axes[0, 1]
    comp_intl = [s["internationality"] for s in scores if s["category"] == "computational"]
    judg_intl = [s["internationality"] for s in scores if s["category"] == "judgment"]

    bp = ax.boxplot(
        [comp_intl, judg_intl],
        labels=["Computational", "Judgment"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][1].set_facecolor("#FF5722")
    for box in bp["boxes"]:
        box.set_alpha(0.5)

    p_mw = primary["analysis"]["group_difference"]["p_mannwhitney"]
    d_eff = primary["analysis"]["group_difference"]["effect_size_d"]
    ax.set_ylabel("Internationality Score")
    ax.set_title(f"B: Comp vs Judg Internationality\n(Mann-Whitney p={p_mw:.4f}, Cohen's d={d_eff:.2f})")

    # ── Panel C: Sub-metric breakdown ──
    ax = axes[1, 0]
    sub = primary["analysis"]["sub_metrics"]
    metrics = ["token_overlap", "roman_sim", "en_pivot"]
    labels = ["Token\nOverlap", "Romanization\nSimilarity", "English-Pivot\nCosine"]
    x_pos = np.arange(len(metrics))
    w = 0.35
    comp_means = [sub[m]["comp_mean"] for m in metrics]
    judg_means = [sub[m]["judg_mean"] for m in metrics]
    comp_stds = [sub[m]["comp_std"] for m in metrics]
    judg_stds = [sub[m]["judg_std"] for m in metrics]

    ax.bar(x_pos - w/2, comp_means, w, yerr=comp_stds, label="Computational",
           color="#2196F3", alpha=0.7, capsize=3)
    ax.bar(x_pos + w/2, judg_means, w, yerr=judg_stds, label="Judgment",
           color="#FF5722", alpha=0.7, capsize=3)

    for i, m in enumerate(metrics):
        p = sub[m]["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y = max(comp_means[i] + comp_stds[i], judg_means[i] + judg_stds[i]) * 1.05
        ax.text(x_pos[i], y, sig, ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("C: Sub-metric Breakdown by Category")
    ax.legend()

    # ── Panel D: Cross-model consistency ──
    ax = axes[1, 1]
    model_names = []
    spearman_rs = []
    p_values = []
    for res in all_results:
        short = res["model"].split("/")[-1][:20]
        model_names.append(short)
        spearman_rs.append(res["analysis"]["correlation"]["spearman_r"])
        p_values.append(res["analysis"]["correlation"]["spearman_p"])

    colors = ["#4CAF50" if p < 0.05 else "#9E9E9E" for p in p_values]
    bars = ax.barh(model_names, spearman_rs, color=colors, alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Spearman rho (internationality vs d_intra)")
    ax.set_title("D: Cross-Model Consistency\n(green = p < 0.05)")

    fig.suptitle(
        "Strategy 1: Does Vocabulary Internationality Explain P2 Failure?",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "strategy1_vocab_internationality.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Figure saved: strategy1_vocab_internationality.png")


def main():
    print("=" * 65)
    print("Strategy 1: Vocabulary Internationality Explains P2 Failure")
    print("=" * 65)

    all_results = []
    for model_name, dim in MODELS:
        print(f"\n--- {model_name} ({dim}d) ---")
        result = run_single_model(model_name, dim)

        # Print summary
        a = result["analysis"]
        print(f"  Spearman rho(intl, d_intra) = {a['correlation']['spearman_r']:.3f}"
              f"  (p = {a['correlation']['spearman_p']:.4f})")
        print(f"  Comp intl mean = {a['group_difference']['comp_internationality_mean']:.3f}")
        print(f"  Judg intl mean = {a['group_difference']['judg_internationality_mean']:.3f}")
        print(f"  Mann-Whitney p = {a['group_difference']['p_mannwhitney']:.4f}")
        print(f"  Partial r(cat, d_intra | intl) = {a['partial_correlation']['r']:.3f}"
              f"  (p = {a['partial_correlation']['p']:.4f})")
        med = a["mediation"]
        print(f"  Mediation: {abs(med['proportion_mediated'])*100:.1f}% mediated"
              f"  (Sobel z = {med['sobel_z']:.2f}, p = {med['sobel_p']:.4f})")
        print(f"  Conclusion: {a['conclusion']['interpretation']}")

        all_results.append(result)

    # Generate figures
    make_figure(all_results)

    # Cross-model summary
    print("\n" + "=" * 65)
    print("Cross-Model Summary")
    print("=" * 65)

    n_support = sum(
        1 for r in all_results
        if r["analysis"]["conclusion"]["vocab_explains_p2"]
    )
    print(f"  Models supporting Strategy 1: {n_support}/{len(all_results)}")

    # Save results (strip per-operation vectors for JSON serialization)
    serializable = []
    for r in all_results:
        entry = {
            "model": r["model"],
            "dim": r["dim"],
            "analysis": r["analysis"],
            "per_op_summary": [
                {k: v for k, v in s.items()
                 if k in ("op_id", "category", "internationality",
                          "token_overlap", "roman_sim", "en_pivot", "d_intra")}
                for s in r["per_operation_scores"]
            ],
        }
        serializable.append(entry)

    out_path = RESULTS_DIR / "strategy1_vocab_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
