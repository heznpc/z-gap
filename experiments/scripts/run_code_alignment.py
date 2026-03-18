#!/usr/bin/env python3
"""NL-Code cross-modal alignment experiment.

Tests PRH for code: do NL descriptions and code converge in Z?
Uses UniXcoder (microsoft/unixcoder-base) which embeds both NL and code.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.code_alignment import CODE_EQUIVALENTS, compute_nl_code_alignment

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"


def main():
    print("=" * 60)
    print("NL-Code Cross-Modal Alignment Experiment")
    print("=" * 60)

    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    categories = {op.id: op.category for op in ops}

    cache = EmbeddingCache(CACHE_DIR)

    # UniXcoder embeds both NL and code into the same space
    print("\nLoading UniXcoder (microsoft/unixcoder-base)...")
    model = SentenceTransformerEmbedder("microsoft/unixcoder-base")
    print(f"  Model: {model.name}, dim={model.dimension}")

    # --- Embed NL descriptions ---
    print("\nStep 1: Embedding NL descriptions...")
    nl_texts, nl_keys = [], []
    for op in ops:
        if op.category != "computational":
            continue
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                nl_texts.append(desc)
                nl_keys.append(f"{op.id}_{lang}")

    nl_array = cache.get_or_compute(model, nl_texts)
    nl_embeddings = {k: nl_array[i] for i, k in enumerate(nl_keys)}
    print(f"  Embedded {len(nl_embeddings)} NL descriptions")

    # --- Embed code equivalents ---
    print("\nStep 2: Embedding code equivalents...")
    code_texts, code_keys = [], []
    for op_id in comp_ids:
        if op_id in CODE_EQUIVALENTS:
            code_texts.append(CODE_EQUIVALENTS[op_id])
            code_keys.append(op_id)

    code_array = cache.get_or_compute(model, code_texts)
    code_embeddings = {k: code_array[i] for i, k in enumerate(code_keys)}
    print(f"  Embedded {len(code_embeddings)} code snippets")

    # --- Compute alignment ---
    print("\nStep 3: Computing NL-Code alignment...")
    result = compute_nl_code_alignment(nl_embeddings, code_embeddings, comp_ids, LANGUAGES)

    print(f"\n  R_code = {result['R_code']:.3f}")
    print(f"  d_match (NL ↔ same code)  = {result['mean_d_match']:.4f} ± {result['d_match_std']:.4f}")
    print(f"  d_mismatch (NL ↔ diff code) = {result['mean_d_mismatch']:.4f}")
    print(f"  n_match = {result['n_match_pairs']}, n_mismatch = {result['n_mismatch_pairs']}")
    print(f"\n  Per-language d_match (NL → same code):")
    for lang, d in sorted(result["per_lang_d_match"].items()):
        print(f"    {lang}: {d:.4f}")

    supported = result["R_code"] > 1.0
    print(f"\n  PRH for code {'SUPPORTED' if supported else 'NOT SUPPORTED'}: "
          f"R_code = {result['R_code']:.3f} {'>' if supported else '<='} 1.0")

    # --- Also test with MiniLM for comparison ---
    print("\n" + "=" * 60)
    print("Comparison: MiniLM-L12 (NL-only model)")
    print("=" * 60)

    model2 = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")

    nl_array2 = cache.get_or_compute(model2, nl_texts)
    nl_embeddings2 = {k: nl_array2[i] for i, k in enumerate(nl_keys)}

    code_array2 = cache.get_or_compute(model2, code_texts)
    code_embeddings2 = {k: code_array2[i] for i, k in enumerate(code_keys)}

    result2 = compute_nl_code_alignment(nl_embeddings2, code_embeddings2, comp_ids, LANGUAGES)

    print(f"\n  R_code = {result2['R_code']:.3f}")
    print(f"  d_match = {result2['mean_d_match']:.4f}, d_mismatch = {result2['mean_d_mismatch']:.4f}")
    print(f"\n  Per-language d_match:")
    for lang, d in sorted(result2["per_lang_d_match"].items()):
        print(f"    {lang}: {d:.4f}")

    # --- Save results ---
    output = {
        "unixcoder": {
            "model": model.name, "dim": model.dimension,
            "R_code": result["R_code"],
            "d_match": result["mean_d_match"],
            "d_mismatch": result["mean_d_mismatch"],
            "per_lang_d_match": result["per_lang_d_match"],
            "supported": result["R_code"] > 1.0,
        },
        "minilm": {
            "model": model2.name, "dim": model2.dimension,
            "R_code": result2["R_code"],
            "d_match": result2["mean_d_match"],
            "d_mismatch": result2["mean_d_mismatch"],
            "per_lang_d_match": result2["per_lang_d_match"],
            "supported": result2["R_code"] > 1.0,
        },
    }

    out_path = RESULTS_DIR / "code_alignment_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # --- Visualization ---
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, res) in zip(axes, [("UniXcoder (code-trained)", result), ("MiniLM-L12 (NL-only)", result2)]):
        bars = ax.bar(["d_match\n(NL ↔ same code)", "d_mismatch\n(NL ↔ diff code)"],
                      [res["mean_d_match"], res["mean_d_mismatch"]],
                      color=["#4CAF50", "#F44336"], width=0.5)
        ax.set_ylabel("Mean Cosine Distance")
        ax.set_title(f"{name}\nR_code = {res['R_code']:.2f}")
        for bar, val in zip(bars, [res["mean_d_match"], res["mean_d_mismatch"]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)

    fig.suptitle("NL-Code Cross-Modal Alignment (PRH for Code)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "code_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES_DIR / 'code_alignment.png'}")

    # Per-language comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    langs = sorted(LANGUAGES)
    x = range(len(langs))
    w = 0.35
    ax.bar([i - w/2 for i in x], [result["per_lang_d_match"][l] for l in langs],
           w, label="UniXcoder", color="#2196F3")
    ax.bar([i + w/2 for i in x], [result2["per_lang_d_match"][l] for l in langs],
           w, label="MiniLM-L12", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("d_match (NL → same code)")
    ax.set_title("Per-Language NL-Code Distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "code_alignment_per_lang.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURES_DIR / 'code_alignment_per_lang.png'}")

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("SUMMARY: NL-Code Cross-Modal Alignment")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'R_code':>7} {'d_match':>8} {'d_mismatch':>11} {'PRH':>5}")
    print("-" * 70)
    for name, res in [("UniXcoder (code-trained)", result), ("MiniLM-L12 (NL-only)", result2)]:
        ok = "YES" if res["R_code"] > 1.0 else "NO"
        print(f"{name:<35} {res['R_code']:>7.3f} {res['mean_d_match']:>8.4f} {res['mean_d_mismatch']:>11.4f} {ok:>5}")


if __name__ == "__main__":
    main()
