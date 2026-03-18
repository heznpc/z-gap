"""Visualization: embedding projections, discriminability charts, heatmaps."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


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

    lang_markers = {"en": "o", "ko": "s", "zh": "^", "ar": "D", "es": "v"}
    cat_colors = {"computational": "#2196F3", "judgment": "#FF5722"}

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
