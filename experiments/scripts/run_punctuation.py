#!/usr/bin/env python3
"""P7 extension: punctuation/special character robustness.

Tests whether punctuation variants of the same sentence map to nearby Z.
E.g., "Sort the list" vs "Sort the list." vs "Sort the list?" vs "Sort the list!"
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"


def generate_punctuation_variants(text: str) -> dict[str, str]:
    """Generate punctuation/formatting variants of a sentence."""
    base = text.rstrip(".!?,:;")
    return {
        "bare": base,
        "period": base + ".",
        "question": base + "?",
        "exclamation": base + "!",
        "ellipsis": base + "...",
        "colon": base + ":",
        "lowercase": base.lower(),
        "uppercase": base.upper(),
        "extra_spaces": "  " + base.replace(" ", "  ") + "  ",
        "no_article": base.replace("the ", "").replace("The ", ""),
    }


def main():
    print("=" * 60)
    print("Punctuation & Special Character Robustness")
    print("=" * 60)

    ops = get_all_operations()
    comp_ops = [op for op in ops if op.category == "computational"]
    judg_ops = [op for op in ops if op.category == "judgment"]
    all_ops = comp_ops + judg_ops

    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")

    # Generate variants for English descriptions
    print(f"\nGenerating punctuation variants for {len(all_ops)} operations...")
    variant_texts = []
    variant_keys = []  # (op_id, variant_name)
    for op in all_ops:
        en_text = op.descriptions.get("en", "")
        if not en_text:
            continue
        variants = generate_punctuation_variants(en_text)
        for vname, vtext in variants.items():
            variant_texts.append(vtext)
            variant_keys.append((op.id, vname))

    print(f"  Total variants: {len(variant_texts)}")

    # Embed
    variant_array = cache.get_or_compute(model, variant_texts)

    # Organize embeddings
    op_variants = {}  # op_id -> {variant_name: vector}
    for i, (op_id, vname) in enumerate(variant_keys):
        if op_id not in op_variants:
            op_variants[op_id] = {}
        op_variants[op_id][vname] = variant_array[i]

    # Compute d_punctuation: distance between variants of same operation
    d_punct_list = []
    for op_id, variants in op_variants.items():
        vecs = list(variants.values())
        for a, b in combinations(vecs, 2):
            d_punct_list.append(float(cosine(a, b)))

    # Compute d_semantic: distance between different operations (bare form)
    bare_vecs = []
    bare_ids = []
    for op_id, variants in op_variants.items():
        if "bare" in variants:
            bare_vecs.append(variants["bare"])
            bare_ids.append(op_id)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(bare_vecs), size=(min(1000, len(bare_vecs) * 3), 2), replace=True)
    indices = indices[indices[:, 0] != indices[:, 1]]
    d_semantic_list = [float(cosine(bare_vecs[i], bare_vecs[j])) for i, j in indices]

    mean_d_punct = float(np.mean(d_punct_list))
    mean_d_semantic = float(np.mean(d_semantic_list))
    R_punct = mean_d_semantic / mean_d_punct if mean_d_punct > 1e-10 else float("inf")

    print(f"\n  d_punctuation (same meaning, diff punctuation) = {mean_d_punct:.4f}")
    print(f"  d_semantic (diff meaning, same punctuation)     = {mean_d_semantic:.4f}")
    print(f"  R_punctuation = {R_punct:.2f}")
    print(f"  P7-ext {'SUPPORTED' if R_punct > 1 else 'NOT SUPPORTED'}")

    # Per-variant-type analysis: which variants cause most drift?
    print(f"\n  Per-variant drift from 'bare' form:")
    variant_drifts = {}
    for op_id, variants in op_variants.items():
        if "bare" not in variants:
            continue
        bare = variants["bare"]
        for vname, vec in variants.items():
            if vname == "bare":
                continue
            d = float(cosine(bare, vec))
            if vname not in variant_drifts:
                variant_drifts[vname] = []
            variant_drifts[vname].append(d)

    sorted_drifts = sorted(variant_drifts.items(), key=lambda x: np.mean(x[1]))
    for vname, dists in sorted_drifts:
        m = float(np.mean(dists))
        print(f"    {vname:<16} mean_drift = {m:.4f}")

    # Save results
    result = {
        "model": model.name,
        "R_punctuation": R_punct,
        "mean_d_punctuation": mean_d_punct,
        "mean_d_semantic": mean_d_semantic,
        "n_ops": len(all_ops),
        "n_variants": len(variant_texts),
        "per_variant_drift": {k: float(np.mean(v)) for k, v in sorted_drifts},
    }

    out_path = RESULTS_DIR / "punctuation_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Visualization
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Bar chart: per-variant drift
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [vname for vname, _ in sorted_drifts]
    means = [float(np.mean(dists)) for _, dists in sorted_drifts]
    colors = ["#4CAF50" if m < 0.05 else "#FF9800" if m < 0.15 else "#F44336" for m in means]
    ax.barh(range(len(names)), means, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean cosine distance from bare form")
    ax.set_title(f"Punctuation Variant Drift — {model.name}\nR_punctuation = {R_punct:.2f}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "punctuation_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES_DIR / 'punctuation_drift.png'}")


if __name__ == "__main__":
    main()
