#!/usr/bin/env python3
"""Main experiment runner: generate stimuli, compute embeddings, test predictions, visualize."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, export_stimuli, LANGUAGES, DIALECTS, get_spacing_variants
from src.embeddings import SentenceTransformerEmbedder, MistralEmbedder, EmbeddingCache
from src.metrics import discriminability_ratio, spacing_robustness, compute_per_operation_detail, dialectal_continuum
from src.predictions import test_p2_cross_lingual_invariance, test_p2_dialectal, test_p7_spacing_robustness
from src.visualize import (
    plot_embedding_space, plot_discriminability, plot_spacing_robustness,
    plot_d_intra_distributions, plot_d_intra_vs_d_inter,
    plot_per_operation_d_intra, plot_p1_scale_trend, plot_cross_lingual_heatmap,
)
from src.analysis import diagnose_p2_failure, compute_p1_trend
from src.report import generate_report

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

    # --- Per-operation detail ---
    per_op_details = compute_per_operation_detail(embeddings, all_ids, LANGUAGES, categories)
    print(f"  Computed per-operation detail for {len(per_op_details)} operations")

    # --- Test P2 ---
    p2 = test_p2_cross_lingual_invariance(embeddings, comp_ids, judg_ids, LANGUAGES)
    print(f"\n  P2: R_C={p2.details['R_C']:.3f}  R_J={p2.details['R_J']:.3f}  "
          f"diff={p2.effect_size:.3f}  p={p2.p_value:.4f}  {'OK' if p2.supported else 'FAIL'}")

    # --- Test P7 ---
    p7 = test_p7_spacing_robustness(embeddings_correct, embeddings_variants, all_ids)
    print(f"  P7: R_spacing={p7.details['R_spacing']:.3f}  "
          f"d_sp={p7.details['mean_d_spacing']:.4f}  d_sem={p7.details['mean_d_semantic']:.4f}  "
          f"p={p7.p_value:.4f}  {'OK' if p7.supported else 'FAIL'}")

    # --- P2 Failure Diagnosis ---
    result_c = discriminability_ratio(embeddings, comp_ids, LANGUAGES)
    result_j = discriminability_ratio(embeddings, judg_ids, LANGUAGES)
    diagnosis = diagnose_p2_failure(result_c, result_j, per_op_details)
    print(f"  P2 diagnosis: primary driver = {diagnosis['primary_driver']}")

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

    # New diagnostic plots
    plot_d_intra_distributions(
        per_op_details, model.name,
        FIGURES_DIR / f"d_intra_dist_{tag}.png"
    )
    plot_d_intra_vs_d_inter(
        per_op_details, model.name,
        FIGURES_DIR / f"d_intra_vs_d_inter_{tag}.png"
    )
    plot_per_operation_d_intra(
        per_op_details, model.name,
        FIGURES_DIR / f"per_op_d_intra_{tag}.png"
    )
    plot_cross_lingual_heatmap(
        per_op_details, LANGUAGES, "computational", model.name,
        FIGURES_DIR / f"heatmap_comp_{tag}.png"
    )
    plot_cross_lingual_heatmap(
        per_op_details, LANGUAGES, "judgment", model.name,
        FIGURES_DIR / f"heatmap_judg_{tag}.png"
    )

    # --- Dialect continuum (P2-dialect) ---
    p2d_result = {"supported": False, "details": {}}
    # Check if any dialect embeddings exist
    dialect_keys = [k for k in embeddings if k.count("_") >= 3]  # op_lang_dialect format
    if dialect_keys:
        p2d = test_p2_dialectal(embeddings, comp_ids, judg_ids, LANGUAGES, DIALECTS)
        print(f"  P2-dialect: continuum={'YES' if p2d.supported else 'NO'}  "
              f"effect={p2d.effect_size:.3f}")
        p2d_result = {"supported": p2d.supported, "effect_size": p2d.effect_size, "details": p2d.details}
    else:
        print("  P2-dialect: skipped (no dialect stimuli loaded)")

    return {
        "model": model.name, "dim": model.dimension,
        "n_comp": len(comp_ids), "n_judg": len(judg_ids),
        "P2": {
            "R_C": p2.details["R_C"], "R_J": p2.details["R_J"],
            "supported": p2.supported, "p": p2.p_value,
            "ci_95": p2.details.get("ci_95"),
            "d_intra_C": result_c["mean_d_intra"],
            "d_intra_J": result_j["mean_d_intra"],
            "d_inter_C": result_c["mean_d_inter"],
            "d_inter_J": result_j["mean_d_inter"],
        },
        "P2_dialect": p2d_result,
        "P7": {
            "R_spacing": p7.details["R_spacing"], "supported": p7.supported,
            "d_spacing": p7.details["mean_d_spacing"],
            "d_semantic": p7.details["mean_d_semantic"],
            "p_value": p7.p_value,
            "ci_95": p7.details.get("ci_95"),
        },
        "diagnosis": diagnosis,
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

    # --- Full model list ---
    # Group 1: Existing baselines
    # Group 2: E5 family (P1 scale-convergence test — same architecture, different scales)
    # Group 3: New multilingual/code models
    models = [
        # Baselines
        SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2"),   # 384d
        # E5 family (P1: scale-convergence)
        SentenceTransformerEmbedder("intfloat/multilingual-e5-small"),           # 384d
        SentenceTransformerEmbedder("intfloat/multilingual-e5-base"),            # 768d
        SentenceTransformerEmbedder("intfloat/multilingual-e5-large"),           # 1024d
        # Cross-lingual retrieval (P2 retest)
        SentenceTransformerEmbedder("BAAI/bge-m3"),                              # 1024d
        # Multilingual (P2, P6)
        SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-8B"),                  # 4096d
        SentenceTransformerEmbedder("jinaai/jina-embeddings-v3"),                 # 1024d
        # Code-specialized (NL-code alignment)
        MistralEmbedder("codestral-embed-2505"),                                 # 1024d
        # TODO: OmniSONAR — requires fairseq2, add when available
        # MetaEmbedder("omnisonar"),
    ]

    all_results = []
    diagnoses = []
    for model in models:
        result = run_model(model, ops, comp_ids, judg_ids, all_ids, categories, cache)
        all_results.append(result)
        diagnoses.append(result["diagnosis"])

    # --- P1 Scale-Convergence Analysis ---
    p1_result = compute_p1_trend(all_results)
    print(f"\n  P1 (Scale-Convergence): rho_total={p1_result.get('rho_total', 'N/A')}, "
          f"p={p1_result.get('p_total', 'N/A')}  "
          f"{'Supported' if p1_result.get('supported') else 'Not supported'}")

    # P1 plot
    if p1_result.get("n_models", 0) >= 3:
        plot_p1_scale_trend(p1_result, FIGURES_DIR / "p1_scale_trend.png")

    # --- Save JSON results ---
    # Strip non-serializable diagnosis data for JSON
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k != "diagnosis"}
        # Serialize diagnosis summary (exclude numpy arrays)
        diag = r["diagnosis"]
        jr["diagnosis_summary"] = {
            "primary_driver": diag["primary_driver"],
            "d_intra_C": diag["d_intra_C"],
            "d_intra_J": diag["d_intra_J"],
            "d_inter_C": diag["d_inter_C"],
            "d_inter_J": diag["d_inter_J"],
            "mannwhitney_p": diag["mannwhitney_p"],
        }
        json_results.append(jr)

    results_path = RESULTS_DIR / "prediction_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # --- Generate Report ---
    generate_report(all_results, p1_result, diagnoses, RESULTS_DIR / "report.md")

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


if __name__ == "__main__":
    main()
