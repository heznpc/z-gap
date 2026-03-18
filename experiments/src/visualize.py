"""Visualization: embedding projections, discriminability charts, heatmaps, diagnostics."""

from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

CAT_COLORS = {"computational": "#2196F3", "judgment": "#FF5722"}
LANG_MARKERS = {"en": "o", "ko": "s", "zh": "^", "ar": "D", "es": "v"}


def plot_embedding_space(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    categories: dict[str, str],  # op_id -> "computational" | "judgment"
    output_path: Path,
    title: str = "Embedding Space (t-SNE)",
):
    """t-SNE projection colored by operation, shaped by language."""
    keys, vecs, ops, langs = [], [], [], []
    for op_id in operation_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                keys.append(key)
                vecs.append(embeddings[key])
                ops.append(op_id)
                langs.append(lang)

    if len(vecs) < 5:
        print("Not enough embeddings for t-SNE")
        return

    X = np.array(vecs)
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_2d = tsne.fit_transform(X)

    lang_markers = LANG_MARKERS
    cat_colors = CAT_COLORS

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i, (op, lang) in enumerate(zip(ops, langs)):
        cat = categories.get(op, "computational")
        ax.scatter(X_2d[i, 0], X_2d[i, 1],
                   c=cat_colors[cat], marker=lang_markers.get(lang, "o"),
                   s=40, alpha=0.7)

    # Legend
    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, label=cat, s=60)
    for lang, marker in lang_markers.items():
        ax.scatter([], [], c="gray", marker=marker, label=lang, s=60)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_discriminability(
    R_C: float, R_J: float,
    model_name: str,
    output_path: Path,
):
    """Bar chart comparing R_C vs R_J."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Computational (R_C)", "Judgment (R_J)"], [R_C, R_J],
                  color=["#2196F3", "#FF5722"], width=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="R = 1")
    ax.set_ylabel("Discriminability Ratio (R)")
    ax.set_title(f"Cross-Lingual Semantic Invariance — {model_name}")
    ax.legend()

    for bar, val in zip(bars, [R_C, R_J]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_spacing_robustness(
    R_spacing: float,
    mean_d_spacing: float,
    mean_d_semantic: float,
    model_name: str,
    output_path: Path,
):
    """Bar chart: d_spacing vs d_semantic."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["d_spacing\n(same meaning,\ndifferent spacing)",
            "d_semantic\n(different meaning,\nsame spacing)"],
           [mean_d_spacing, mean_d_semantic],
           color=["#4CAF50", "#9C27B0"], width=0.5)
    ax.set_ylabel("Mean Cosine Distance")
    ax.set_title(f"Spacing Robustness — {model_name}\nR_spacing = {R_spacing:.2f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- New diagnostic visualizations ---


def plot_d_intra_distributions(
    per_op_details: list[dict],
    model_name: str,
    output_path: Path,
):
    """Violin + strip plot comparing d_intra distributions: computational vs judgment."""
    import pandas as pd
    from scipy.stats import mannwhitneyu

    rows = [{"category": d["category"], "d_intra": d["d_intra"]} for d in per_op_details]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(
        [df[df.category == "computational"]["d_intra"].values,
         df[df.category == "judgment"]["d_intra"].values],
        positions=[0, 1], showmedians=True, showextrema=False,
    )
    for i, cat in enumerate(["computational", "judgment"]):
        body = parts["bodies"][i]
        body.set_facecolor(CAT_COLORS[cat])
        body.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Overlay strip
    for i, cat in enumerate(["computational", "judgment"]):
        vals = df[df.category == cat]["d_intra"].values
        jitter = np.random.default_rng(42).normal(0, 0.03, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   c=CAT_COLORS[cat], s=20, alpha=0.5, zorder=3)

    # Mann-Whitney U
    comp_vals = df[df.category == "computational"]["d_intra"].values
    judg_vals = df[df.category == "judgment"]["d_intra"].values
    if len(comp_vals) > 0 and len(judg_vals) > 0:
        _, p = mannwhitneyu(comp_vals, judg_vals, alternative="two-sided")
        ax.set_title(f"d_intra Distribution — {model_name}\nMann-Whitney p = {p:.4f}")
    else:
        ax.set_title(f"d_intra Distribution — {model_name}")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Computational", "Judgment"])
    ax.set_ylabel("d_intra (cross-lingual distance)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_d_intra_vs_d_inter(
    per_op_details: list[dict],
    model_name: str,
    output_path: Path,
):
    """Scatter: d_intra (invariance) vs mean d_inter (distinctiveness) per operation."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for d in per_op_details:
        cat = d["category"]
        ax.scatter(d["d_intra"], d["mean_d_inter"],
                   c=CAT_COLORS.get(cat, "gray"), s=30, alpha=0.7)

    for cat, color in CAT_COLORS.items():
        ax.scatter([], [], c=color, label=cat, s=50)
    ax.legend()
    ax.set_xlabel("d_intra (cross-lingual distance, lower = more invariant)")
    ax.set_ylabel("mean d_inter (distance to other ops, higher = more distinctive)")
    ax.set_title(f"Invariance vs Distinctiveness — {model_name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_operation_d_intra(
    per_op_details: list[dict],
    model_name: str,
    output_path: Path,
    top_n: int = 25,
):
    """Sorted horizontal bar chart of d_intra per operation (top + bottom N)."""
    sorted_ops = sorted(per_op_details, key=lambda d: d["d_intra"])

    if len(sorted_ops) > 2 * top_n:
        show = sorted_ops[:top_n] + sorted_ops[-top_n:]
    else:
        show = sorted_ops

    labels = []
    for d in show:
        parts = d["op_id"].split("_", 2)
        label = parts[-1] if len(parts) > 2 else d["op_id"]
        prefix = "C" if d["category"] == "computational" else "J"
        labels.append(f"{prefix}: {label}")

    values = [d["d_intra"] for d in show]
    colors = [CAT_COLORS.get(d["category"], "gray") for d in show]

    fig, ax = plt.subplots(figsize=(8, max(6, len(show) * 0.25)))
    y_pos = range(len(show))
    ax.barh(y_pos, values, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("d_intra (cross-lingual distance)")
    ax.set_title(f"Per-Operation Cross-Lingual Invariance — {model_name}")
    ax.invert_yaxis()

    for cat, color in CAT_COLORS.items():
        ax.barh([], [], color=color, label=cat)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_p1_scale_trend(
    p1_result: dict,
    output_path: Path,
):
    """Line plot: R vs model dimension for P1 scale-convergence."""
    dims = p1_result["dims"]
    R_Cs = p1_result["R_Cs"]
    R_Js = p1_result["R_Js"]
    names = p1_result["model_names"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, R_Cs, "o-", color=CAT_COLORS["computational"], label=f"R_C (rho={p1_result['rho_C']:.2f})", markersize=8)
    ax.plot(dims, R_Js, "s-", color=CAT_COLORS["judgment"], label=f"R_J (rho={p1_result['rho_J']:.2f})", markersize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="R = 1")

    for i, name in enumerate(names):
        short = name.split("/")[-1][:20]
        ax.annotate(short, (dims[i], R_Cs[i]), textcoords="offset points",
                    xytext=(5, 8), fontsize=6, alpha=0.7)

    ax.set_xlabel("Model Dimension")
    ax.set_ylabel("Discriminability Ratio (R)")
    ax.set_title(f"P1: Scale-Convergence Trend (n={p1_result['n_models']} models)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_lingual_heatmap(
    per_op_details: list[dict],
    languages: list[str],
    category: str,
    model_name: str,
    output_path: Path,
):
    """Heatmap: mean cross-lingual similarity per language pair, split by category."""
    ops = [d for d in per_op_details if d["category"] == category]
    if not ops:
        return

    lang_pairs = [f"{l1}-{l2}" for l1, l2 in combinations(languages, 2)]
    pair_means = {}
    for pair in lang_pairs:
        vals = [d["lang_pair_distances"].get(pair, np.nan) for d in ops]
        pair_means[pair] = float(np.nanmean(vals))

    n = len(languages)
    sim_matrix = np.eye(n)
    for i, l1 in enumerate(languages):
        for j, l2 in enumerate(languages):
            if i < j:
                pair_key = f"{l1}-{l2}"
                dist = pair_means.get(pair_key, 0.0)
                sim_matrix[i, j] = 1.0 - dist
                sim_matrix[j, i] = 1.0 - dist

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = "Blues" if category == "computational" else "Oranges"
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap=cmap,
                xticklabels=languages, yticklabels=languages,
                vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f"Cross-Lingual Similarity ({category}) — {model_name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
