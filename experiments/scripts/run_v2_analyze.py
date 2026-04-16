#!/usr/bin/env python3
"""V2 Step 2: Analyze hidden states (runs on CPU, no GPU needed).

Usage:
    python scripts/run_v2_analyze.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_v2_analyze.py --model all
    python scripts/run_v2_analyze.py --model all --cka   # include cross-model CKA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hidden_states import load_hidden_states, PRIMARY_MODELS
from src.hidden_state_analysis import (
    layer_convergence_curve, cross_modal_alignment, linear_cka,
    cross_model_cka, p2_per_layer, rsa_analysis, tier_comparison,
)
from src.hidden_state_visualize import (
    plot_convergence_curve, plot_cross_modal_alignment,
    plot_convergence_and_code_overlay, plot_cka_heatmap,
    plot_p2_per_layer, plot_rsa_curves, plot_tier_comparison,
)
from src.stimuli import LANGUAGES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "hidden_states"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures" / "v2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "stimuli"


def load_states_for_model(model_name: str) -> dict:
    """Load all extracted states for a model, organized by tier and modality."""
    safe_name = model_name.split("/")[-1]
    model_dir = RESULTS_DIR / safe_name

    if not model_dir.exists():
        print(f"  No data found for {model_name}")
        return {}

    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  No metadata for {model_name}")
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    # Load op_ids from metadata
    op_ids_by_tier = {}
    for fname, info in meta.get("files", {}).items():
        tier = info.get("tier")
        if "op_ids" in info:
            op_ids_by_tier[tier] = info["op_ids"]

    # Load NL states: {op_id: {lang: array(n_layers+1, dim)}}
    nl_states = {}
    for tier in [1, 2, 3]:
        for lang in LANGUAGES:
            try:
                states = load_hidden_states(RESULTS_DIR, model_name, tier, "nl", lang)
                op_ids = op_ids_by_tier.get(tier, [])
                for i, op_id in enumerate(op_ids):
                    if i < len(states):
                        if op_id not in nl_states:
                            nl_states[op_id] = {}
                        nl_states[op_id][lang] = states[i]
            except FileNotFoundError:
                pass

    # Load code states: {op_id: array(n_layers+1, dim)}
    code_states = {}
    for tier in [1, 2, 3]:
        try:
            states = load_hidden_states(RESULTS_DIR, model_name, tier, "code")
            op_ids = op_ids_by_tier.get(tier, [])
            for i, op_id in enumerate(op_ids):
                if i < len(states):
                    code_states[op_id] = states[i]
        except FileNotFoundError:
            pass

    n_layers = meta.get("n_layers", 32)
    return {
        "nl": nl_states,
        "code": code_states,
        "n_layers": n_layers,
        "meta": meta,
        "op_ids_by_tier": op_ids_by_tier,
    }


def analyze_single_model(model_name: str, do_p2: bool = True, n_perm: int = 1000):
    """Run all analyses for one model."""
    safe_name = model_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    data = load_states_for_model(model_name)
    if not data or not data["nl"]:
        print("  No states loaded, skipping.")
        return {}

    nl_states = data["nl"]
    code_states = data["code"]
    n_layers = data["n_layers"]

    fig_dir = FIGURES_DIR / safe_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {"model": model_name}

    # 4.1 Layer-wise convergence curve
    print("  [1/5] Layer convergence curve...")
    conv = layer_convergence_curve(nl_states, LANGUAGES, n_layers)
    results["convergence"] = {
        "R": conv["R"].tolist(),
        "peak_layer": int(np.argmax(conv["R"])),
        "peak_R": float(np.max(conv["R"])),
    }
    plot_convergence_curve(conv, safe_name, str(fig_dir / "convergence_curve.png"))

    # 4.2 Cross-modal alignment
    if code_states:
        print("  [2/5] Cross-modal alignment...")
        cma = cross_modal_alignment(nl_states, code_states, LANGUAGES, n_layers)
        results["cross_modal"] = {
            "R_code": cma["R_code"].tolist(),
            "peak_layer": int(np.argmax(cma["R_code"])),
            "peak_R_code": float(np.max(cma["R_code"])),
        }
        plot_cross_modal_alignment(cma, safe_name,
                                   str(fig_dir / "cross_modal_alignment.png"))
        plot_convergence_and_code_overlay(
            conv, cma, safe_name, str(fig_dir / "convergence_overlay.png"))
    else:
        print("  [2/5] No code states, skipping cross-modal.")

    # 4.4 P2 per-layer
    if do_p2:
        print(f"  [3/5] P2 per-layer (n_perm={n_perm})...")
        # Split ops into comp/judg by ID prefix
        comp_ids = [oid for oid in nl_states if oid.startswith("comp")]
        judg_ids = [oid for oid in nl_states if oid.startswith("judg")]
        if comp_ids and judg_ids:
            p2 = p2_per_layer(nl_states, comp_ids, judg_ids, LANGUAGES,
                              n_layers, n_perm=n_perm)
            results["p2"] = {
                "R_C": p2["R_C"].tolist(),
                "R_J": p2["R_J"].tolist(),
                "p_values": p2["p_values"].tolist(),
                "p2_holds_layers": [int(l) for l in range(n_layers + 1)
                                    if p2["R_C"][l] > p2["R_J"][l] and p2["p_values"][l] < 0.05],
            }
            plot_p2_per_layer(p2, safe_name, str(fig_dir / "p2_per_layer.png"))
        else:
            print("    No judgment ops found, skipping P2.")
    else:
        print("  [3/5] P2 skipped.")

    # 4.5 RSA
    print("  [4/5] RSA analysis...")
    all_op_ids = list(nl_states.keys())
    rsa = rsa_analysis(nl_states, code_states or None, all_op_ids, LANGUAGES, n_layers)
    results["rsa"] = {
        "cross_lingual": {f"{a}-{b}": rho.tolist()
                          for (a, b), rho in rsa["cross_lingual"].items()},
    }
    if rsa.get("nl_code"):
        results["rsa"]["nl_code"] = {lang: rho.tolist()
                                     for lang, rho in rsa["nl_code"].items()}
    plot_rsa_curves(rsa, safe_name, str(fig_dir / "rsa_curves.png"))

    # Tier comparison
    print("  [5/5] Tier comparison...")
    tier_ids = data.get("op_ids_by_tier", {})
    if len(tier_ids) > 1:
        tc = tier_comparison(nl_states, tier_ids, LANGUAGES, n_layers)
        results["tier_comparison"] = {
            tier: {"peak_layer": int(np.argmax(r["R"])), "peak_R": float(np.max(r["R"]))}
            for tier, r in tc.items()
        }
        plot_tier_comparison(tc, safe_name, str(fig_dir / "tier_comparison.png"))

    return results


def run_cka_pairs(models: list[str]):
    """Run CKA comparison between model pairs."""
    from itertools import combinations

    print("\n--- Cross-model CKA ---")
    model_data = {}
    for m in models:
        d = load_states_for_model(m)
        if d and d["nl"]:
            model_data[m] = d

    for (ma, mb) in combinations(model_data.keys(), 2):
        safe_a = ma.split("/")[-1]
        safe_b = mb.split("/")[-1]
        print(f"  CKA: {safe_a} vs {safe_b}")

        data_a = model_data[ma]
        data_b = model_data[mb]
        common_ops = [op for op in data_a["nl"] if op in data_b["nl"]]

        if len(common_ops) < 5:
            print(f"    Only {len(common_ops)} common ops, skipping.")
            continue

        for lang in ["en"]:  # Primary CKA on English
            cka = cross_model_cka(
                data_a["nl"], data_b["nl"], common_ops, lang,
                data_a["n_layers"], data_b["n_layers"]
            )
            fig_dir = FIGURES_DIR / "cka"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_cka_heatmap(
                cka, safe_a, safe_b, lang,
                str(fig_dir / f"cka_{safe_a}_vs_{safe_b}_{lang}.png")
            )


def main():
    parser = argparse.ArgumentParser(description="V2: Analyze hidden states")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--cka", action="store_true", help="Run cross-model CKA")
    parser.add_argument("--no-p2", action="store_true", help="Skip P2 analysis")
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = PRIMARY_MODELS
    else:
        models = [args.model]

    all_results = {}
    for model_name in models:
        result = analyze_single_model(
            model_name, do_p2=not args.no_p2, n_perm=args.n_perm
        )
        if result:
            all_results[model_name] = result

    # Save combined results
    out_path = RESULTS_DIR / "v2_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # CKA
    if args.cka and len(models) > 1:
        run_cka_pairs(models)


if __name__ == "__main__":
    main()
