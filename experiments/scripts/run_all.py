#!/usr/bin/env python3
"""Main experiment runner: generate stimuli, compute embeddings, test predictions, visualize."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, export_stimuli, LANGUAGES, get_spacing_variants
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.metrics import discriminability_ratio, spacing_robustness
from src.predictions import test_p2_cross_lingual_invariance, test_p7_spacing_robustness
from src.visualize import plot_embedding_space, plot_discriminability, plot_spacing_robustness

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "stimuli"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"


def run_model(model, ops, comp_ids, judg_ids, all_ids, categories, cache):
    """Run full pipeline for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {model.name} (dim={model.dimension})")
    print(f"{'='*60}")

    # --- Embed all ops × languages ---
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get_or_compute(model, texts)
    embeddings = {k: embeddings_array[i] for i, k in enumerate(keys)}
    print(f"  Embedded {len(embeddings)} descriptions")

    # --- Spacing variants (Korean) ---
    spacing_texts = []
    spacing_variants_map = {}
    for op in ops:
        ko_text = op.descriptions.get("ko", "")
        if not ko_text:
            continue
        variants = get_spacing_variants(ko_text)
        spacing_variants_map[op.id] = {}
        for vname, vtext in variants.items():
            spacing_variants_map[op.id][vname] = len(spacing_texts)
            spacing_texts.append(vtext)

    spacing_array = cache.get_or_compute(model, spacing_texts)
    embeddings_correct, embeddings_variants = {}, {}
    for op_id, variants in spacing_variants_map.items():
        embeddings_variants[op_id] = {}
        for vname, idx in variants.items():
            embeddings_variants[op_id][vname] = spacing_array[idx]
            if vname == "correct":
                embeddings_correct[op_id] = spacing_array[idx]
    print(f"  Embedded {len(spacing_texts)} spacing variants")

    # --- Test P2 ---
    p2 = test_p2_cross_lingual_invariance(embeddings, comp_ids, judg_ids, LANGUAGES)
    print(f"\n  P2: R_C={p2.details['R_C']:.3f}  R_J={p2.details['R_J']:.3f}  "
          f"diff={p2.effect_size:.3f}  p={p2.p_value:.4f}  {'OK' if p2.supported else 'FAIL'}")

    # --- Test P7 ---
    p7 = test_p7_spacing_robustness(embeddings_correct, embeddings_variants, all_ids)
    print(f"  P7: R_spacing={p7.details['R_spacing']:.3f}  "
          f"d_sp={p7.details['mean_d_spacing']:.4f}  d_sem={p7.details['mean_d_semantic']:.4f}  "
          f"{'OK' if p7.supported else 'FAIL'}")

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = model.name.replace("/", "_")

    plot_embedding_space(
        embeddings, all_ids, LANGUAGES, categories,
        FIGURES_DIR / f"embedding_space_{tag}.png",
        title=f"Embedding Space — {model.name}"
    )
    plot_discriminability(
        p2.details["R_C"], p2.details["R_J"], model.name,
        FIGURES_DIR / f"discriminability_{tag}.png"
    )
    plot_spacing_robustness(
        p7.details["R_spacing"], p7.details["mean_d_spacing"],
        p7.details["mean_d_semantic"], model.name,
        FIGURES_DIR / f"spacing_{tag}.png"
    )

    return {
        "model": model.name, "dim": model.dimension,
        "n_comp": len(comp_ids), "n_judg": len(judg_ids),
        "P2": {"R_C": p2.details["R_C"], "R_J": p2.details["R_J"],
               "supported": p2.supported, "p": p2.p_value},
        "P7": {"R_spacing": p7.details["R_spacing"], "supported": p7.supported,
               "d_spacing": p7.details["mean_d_spacing"],
               "d_semantic": p7.details["mean_d_semantic"]},
    }


def main():
    print("Step 1: Generating stimuli")
    export_stimuli(DATA_DIR)

    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    judg_ids = [op.id for op in ops if op.category == "judgment"]
    all_ids = comp_ids + judg_ids
    categories = {op.id: op.category for op in ops}
    print(f"Total: {len(comp_ids)} comp + {len(judg_ids)} judg = {len(all_ids)} ops × {len(LANGUAGES)} langs")

    cache = EmbeddingCache(CACHE_DIR)

    models = [
        SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2"),
        SentenceTransformerEmbedder("intfloat/multilingual-e5-large"),
    ]

    all_results = []
    for model in models:
        result = run_model(model, ops, comp_ids, judg_ids, all_ids, categories, cache)
        all_results.append(result)

    # Save
    results_path = RESULTS_DIR / "prediction_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'dim':>5} | {'R_C':>6} {'R_J':>6} {'P2':>4} | {'R_sp':>6} {'P7':>4}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['model']:<45} {r['dim']:>5} | "
              f"{r['P2']['R_C']:>6.2f} {r['P2']['R_J']:>6.2f} "
              f"{'OK' if r['P2']['supported'] else 'FAIL':>4} | "
              f"{r['P7']['R_spacing']:>6.2f} {'OK' if r['P7']['supported'] else 'FAIL':>4}")

    # P1 check: does R increase with scale?
    if len(all_results) >= 2:
        R_small = all_results[0]["P2"]["R_C"] + all_results[0]["P2"]["R_J"]
        R_large = all_results[1]["P2"]["R_C"] + all_results[1]["P2"]["R_J"]
        print(f"\n  P1 (Scale-Convergence): R_total small={R_small:.2f} → large={R_large:.2f}  "
              f"{'INCREASING (OK)' if R_large > R_small else 'NOT INCREASING'}")


if __name__ == "__main__":
    main()
