#!/usr/bin/env python3
"""P1 (scale-convergence) and P3 (stratification separability) experiments.

P1: Load models one at a time, embed, cache, delete weights. Test R vs scale.
P3: Train linear probe on English embeddings → test cross-lingual generalization.
"""

import json
import sys
import gc
import shutil
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.metrics import discriminability_ratio
from src.predictions import test_p2_cross_lingual_invariance

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

# Models ordered by parameter count (ascending)
MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384, "118M"),
    ("paraphrase-multilingual-mpnet-base-v2", 768, "278M"),
    ("intfloat/multilingual-e5-large", 1024, "560M"),
]

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def clear_model_cache(model_name: str):
    """Delete model weights from HF cache to free disk."""
    safe_name = model_name.replace("/", "--")
    for d in HF_CACHE.iterdir():
        if safe_name in d.name and d.is_dir():
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            shutil.rmtree(d)
            print(f"  Freed {size_mb:.0f} MB from {d.name}")


def run_p1():
    """P1: Scale-convergence — R vs model scale."""
    print("=" * 60)
    print("P1: Scale-Convergence Experiment")
    print("=" * 60)

    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    judg_ids = [op.id for op in ops if op.category == "judgment"]
    all_ids = comp_ids + judg_ids

    cache = EmbeddingCache(CACHE_DIR)
    results = []

    for model_name, expected_dim, param_count in MODELS:
        print(f"\n--- {model_name} ({param_count} params, {expected_dim}d) ---")

        # Load model
        model = SentenceTransformerEmbedder(model_name)
        actual_dim = model.dimension

        # Embed all ops × languages
        texts, keys = [], []
        for op in ops:
            for lang in LANGUAGES:
                desc = op.descriptions.get(lang)
                if desc:
                    texts.append(desc)
                    keys.append(f"{op.id}_{lang}")

        embeddings_array = cache.get_or_compute(model, texts)
        embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}

        # Compute R
        p2 = test_p2_cross_lingual_invariance(embeddings, comp_ids, judg_ids, LANGUAGES)
        R_C = p2.details["R_C"]
        R_J = p2.details["R_J"]
        R_total = R_C + R_J

        results.append({
            "model": model_name,
            "params": param_count,
            "dim": actual_dim,
            "R_C": R_C,
            "R_J": R_J,
            "R_total": R_total,
        })

        print(f"  R_C={R_C:.3f}, R_J={R_J:.3f}, R_total={R_total:.3f}")

        # Free memory
        del model, embeddings_array, embeddings
        gc.collect()

        # Don't delete MiniLM (small, keep for other experiments)
        if "MiniLM" not in model_name:
            clear_model_cache(model_name)

    # Spearman correlation
    dims = [r["dim"] for r in results]
    R_Cs = [r["R_C"] for r in results]
    R_Js = [r["R_J"] for r in results]
    R_totals = [r["R_total"] for r in results]

    if len(results) >= 3:
        rho, p = stats.spearmanr(dims, R_totals)
        print(f"\n  Spearman rho(dim, R_total) = {rho:.3f}, p = {p:.4f}")
        p1_supported = rho > 0 and p < 0.1
    else:
        rho, p = 0.0, 1.0
        p1_supported = False

    print(f"  P1 {'SUPPORTED' if p1_supported else 'NOT SUPPORTED'}")

    # Save
    p1_result = {
        "models": results,
        "rho": float(rho),
        "p": float(p),
        "supported": p1_supported,
    }

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, R_Cs, "o-", color="#2196F3", label=f"R_C", markersize=8)
    ax.plot(dims, R_Js, "s-", color="#FF5722", label=f"R_J", markersize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="R = 1")

    for i, r in enumerate(results):
        short = r["model"].split("/")[-1][:25]
        ax.annotate(f"{short}\n({r['params']})", (dims[i], R_Cs[i]),
                    textcoords="offset points", xytext=(5, 10), fontsize=7, alpha=0.7)

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Discriminability Ratio (R)")
    ax.set_title(f"P1: Scale-Convergence (n={len(results)} models, rho={rho:.2f}, p={p:.3f})")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "p1_scale_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: p1_scale_trend.png")

    return p1_result


def run_p3():
    """P3: Stratification separability — cross-lingual probing."""
    print("\n" + "=" * 60)
    print("P3: Stratification Separability (Probing)")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    ops = get_all_operations()
    categories = {op.id: op.category for op in ops}
    all_ids = [op.id for op in ops]

    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")

    # Embed
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}

    # --- Probe 1: "What is computed" (Z_sem) ---
    # Train on English, test on each other language
    print("\n  Probe 1: Category classification (Z_sem)")
    print("  Train on English → test on other languages")

    # English training data
    X_train = []
    y_train = []
    for op_id in all_ids:
        key = f"{op_id}_en"
        if key in embeddings:
            X_train.append(embeddings[key])
            y_train.append(1 if categories[op_id] == "computational" else 0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"    en (train): {train_acc:.3f}")

    cross_lingual_accs = {}
    for lang in LANGUAGES:
        X_test = []
        y_test = []
        for op_id in all_ids:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                X_test.append(embeddings[key])
                y_test.append(1 if categories[op_id] == "computational" else 0)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        acc = accuracy_score(y_test, clf.predict(X_test))
        cross_lingual_accs[lang] = float(acc)
        marker = "(train)" if lang == "en" else ""
        print(f"    {lang}: {acc:.3f} {marker}")

    mean_transfer = float(np.mean([v for k, v in cross_lingual_accs.items() if k != "en"]))
    print(f"    Mean transfer (non-en): {mean_transfer:.3f}")

    # --- Probe 2: Operation identity (finer-grained Z_sem) ---
    print("\n  Probe 2: Operation identity (100-way, finer Z_sem)")

    op_to_idx = {op_id: i for i, op_id in enumerate(all_ids)}
    X_train2, y_train2 = [], []
    for op_id in all_ids:
        key = f"{op_id}_en"
        if key in embeddings:
            X_train2.append(embeddings[key])
            y_train2.append(op_to_idx[op_id])
    X_train2 = np.array(X_train2)
    y_train2 = np.array(y_train2)

    clf2 = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf2.fit(X_train2, y_train2)

    op_transfer_accs = {}
    for lang in LANGUAGES:
        X_test, y_test = [], []
        for op_id in all_ids:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                X_test.append(embeddings[key])
                y_test.append(op_to_idx[op_id])
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        acc = accuracy_score(y_test, clf2.predict(X_test))
        op_transfer_accs[lang] = float(acc)
        print(f"    {lang}: {acc:.3f}")

    mean_op_transfer = float(np.mean([v for k, v in op_transfer_accs.items() if k != "en"]))
    print(f"    Mean transfer (non-en): {mean_op_transfer:.3f}")

    # P3 interpretation
    p3_supported = mean_transfer > 0.7  # category generalizes cross-lingually
    print(f"\n  P3: Category probe transfers cross-lingually ({mean_transfer:.3f})")
    print(f"  P3 {'SUPPORTED' if p3_supported else 'NOT SUPPORTED'}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Probe 1: category
    langs = sorted(LANGUAGES)
    accs1 = [cross_lingual_accs[l] for l in langs]
    colors1 = ["#4CAF50" if l == "en" else "#2196F3" for l in langs]
    axes[0].bar(langs, accs1, color=colors1)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Probe 1: Category (comp vs judg)\nTrained on English")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Probe 2: operation identity
    accs2 = [op_transfer_accs[l] for l in langs]
    colors2 = ["#4CAF50" if l == "en" else "#FF9800" for l in langs]
    axes[1].bar(langs, accs2, color=colors2)
    axes[1].axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="chance (1%)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Probe 2: Operation identity (100-way)\nTrained on English")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("P3: Cross-Lingual Probe Transfer (Z_sem Separability)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "p3_probing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: p3_probing.png")

    return {
        "category_probe": {
            "train_acc": train_acc,
            "cross_lingual_accs": cross_lingual_accs,
            "mean_transfer": mean_transfer,
        },
        "operation_probe": {
            "cross_lingual_accs": op_transfer_accs,
            "mean_transfer": mean_op_transfer,
        },
        "p3_supported": p3_supported,
    }


def main():
    p1 = run_p1()
    p3 = run_p3()

    # Save combined results
    combined = {"P1": p1, "P3": p3}
    out_path = RESULTS_DIR / "p1_p3_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nAll results saved: {out_path}")


if __name__ == "__main__":
    main()
