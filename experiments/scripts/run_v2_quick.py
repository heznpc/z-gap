#!/usr/bin/env python3
"""V2 Quick Run: Validate pipeline on M4 Mac with small model.

Extracts hidden states from Qwen2.5-Coder-1.5B-Instruct (Tier 1 only),
runs layer-wise convergence analysis, and produces figures.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "stimuli"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "hidden_states"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures" / "v2"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LANGUAGES = ["en", "ko", "zh", "ar", "es"]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.spatial.distance import cosine
    from itertools import combinations

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load stimuli ──
    print("Loading Tier 1 stimuli...")
    with open(DATA_DIR / "computational.json") as f:
        comp_ops = json.load(f)
    with open(DATA_DIR / "judgment.json") as f:
        judg_ops = json.load(f)

    # Code equivalents
    from src.code_alignment import CODE_EQUIVALENTS

    # ── Load model ──
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        output_hidden_states=True,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded in {time.time()-t0:.1f}s: {n_layers} layers, {hidden_dim}d")

    # ── Extract function ──
    def extract_single(text: str) -> np.ndarray:
        """Returns (n_layers+1, hidden_dim)."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        layers = []
        for h in outputs.hidden_states:
            vec = h[0, -1, :]  # last token
            layers.append(vec.cpu().float().numpy())
        return np.stack(layers)

    # ── Extract NL states ──
    print("\nExtracting NL hidden states...")
    nl_states = {}  # {op_id: {lang: array(n_layers+1, dim)}}
    all_ops = comp_ops + judg_ops
    total = len(all_ops) * len(LANGUAGES)
    done = 0

    for op in all_ops:
        op_id = op["id"]
        nl_states[op_id] = {}
        for lang in LANGUAGES:
            desc = op.get("descriptions", {}).get(lang)
            if desc:
                nl_states[op_id][lang] = extract_single(desc)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} ({done*100//total}%)")

    print(f"  Done: {len(nl_states)} operations")

    # ── Extract code states ──
    print("Extracting code hidden states...")
    code_states = {}  # {op_id: array(n_layers+1, dim)}
    for op in comp_ops:
        op_id = op["id"]
        code = CODE_EQUIVALENTS.get(op_id)
        if code:
            code_states[op_id] = extract_single(code)
    print(f"  Done: {len(code_states)} code snippets")

    # ── Free GPU memory ──
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # ── Analysis: Layer-wise convergence ──
    print("\nAnalyzing layer-wise convergence...")
    comp_ids = [op["id"] for op in comp_ops]
    judg_ids = [op["id"] for op in judg_ops]

    R_all = np.zeros(n_layers + 1)
    R_C = np.zeros(n_layers + 1)
    R_J = np.zeros(n_layers + 1)
    R_code = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        # Cross-lingual convergence (all ops)
        d_intra, d_inter = [], []
        for op_id in nl_states:
            vecs = [nl_states[op_id][l][layer] for l in LANGUAGES if l in nl_states[op_id]]
            for a, b in combinations(vecs, 2):
                d_intra.append(cosine(a, b))
        for lang in LANGUAGES:
            lang_vecs = [nl_states[op_id][lang][layer]
                         for op_id in nl_states if lang in nl_states[op_id]]
            for a, b in combinations(lang_vecs, 2):
                d_inter.append(cosine(a, b))
        R_all[layer] = np.mean(d_inter) / max(np.mean(d_intra), 1e-10)

        # R_C (computational only)
        d_intra_c, d_inter_c = [], []
        for op_id in comp_ids:
            vecs = [nl_states[op_id][l][layer] for l in LANGUAGES if l in nl_states.get(op_id, {})]
            for a, b in combinations(vecs, 2):
                d_intra_c.append(cosine(a, b))
        for lang in LANGUAGES:
            lang_vecs = [nl_states[op_id][lang][layer]
                         for op_id in comp_ids if lang in nl_states.get(op_id, {})]
            for a, b in combinations(lang_vecs, 2):
                d_inter_c.append(cosine(a, b))
        R_C[layer] = np.mean(d_inter_c) / max(np.mean(d_intra_c), 1e-10) if d_intra_c else 0

        # R_J (judgment only)
        d_intra_j, d_inter_j = [], []
        for op_id in judg_ids:
            vecs = [nl_states[op_id][l][layer] for l in LANGUAGES if l in nl_states.get(op_id, {})]
            for a, b in combinations(vecs, 2):
                d_intra_j.append(cosine(a, b))
        for lang in LANGUAGES:
            lang_vecs = [nl_states[op_id][lang][layer]
                         for op_id in judg_ids if lang in nl_states.get(op_id, {})]
            for a, b in combinations(lang_vecs, 2):
                d_inter_j.append(cosine(a, b))
        R_J[layer] = np.mean(d_inter_j) / max(np.mean(d_intra_j), 1e-10) if d_intra_j else 0

        # R_code (NL-code alignment)
        d_match, d_mismatch = [], []
        for op_id in comp_ids:
            if op_id not in code_states:
                continue
            code_vec = code_states[op_id][layer]
            for lang in LANGUAGES:
                if lang not in nl_states.get(op_id, {}):
                    continue
                nl_vec = nl_states[op_id][lang][layer]
                d_match.append(cosine(nl_vec, code_vec))
                others = [oid for oid in comp_ids if oid != op_id and oid in code_states]
                for oid in others[:10]:
                    d_mismatch.append(cosine(nl_vec, code_states[oid][layer]))
        R_code[layer] = np.mean(d_mismatch) / max(np.mean(d_match), 1e-10) if d_match else 0

        if layer % 5 == 0:
            print(f"  Layer {layer}/{n_layers}: R={R_all[layer]:.3f}, R_C={R_C[layer]:.3f}, "
                  f"R_J={R_J[layer]:.3f}, R_code={R_code[layer]:.3f}")

    # ── Save results ──
    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "device": DEVICE,
        "R_all": R_all.tolist(),
        "R_C": R_C.tolist(),
        "R_J": R_J.tolist(),
        "R_code": R_code.tolist(),
        "peak_R_all": {"layer": int(np.argmax(R_all)), "value": float(np.max(R_all))},
        "peak_R_code": {"layer": int(np.argmax(R_code)), "value": float(np.max(R_code))},
        "p2_layers_where_RC_gt_RJ": [int(l) for l in range(n_layers+1) if R_C[l] > R_J[l]],
    }

    out_json = RESULTS_DIR / "v2_quick_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(n_layers + 1)

    # Fig 1: Convergence curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, R_all, "o-", color="#1f77b4", linewidth=2, markersize=4, label="R (all ops)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    peak = np.argmax(R_all)
    ax.annotate(f"peak: layer {peak}\nR={R_all[peak]:.2f}",
                xy=(peak, R_all[peak]), xytext=(peak+1, R_all[peak]+0.1),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red")
    ax.set_xlabel("Layer"); ax.set_ylabel("R = d_inter / d_intra")
    ax.set_title(f"Cross-lingual Convergence — {MODEL_NAME.split('/')[-1]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "v2_quick_convergence.png"), dpi=150)
    plt.close(fig)

    # Fig 2: P2 per layer (R_C vs R_J)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, R_C, "o-", color="#2ca02c", linewidth=2, markersize=3, label="R_C (computational)")
    ax.plot(layers, R_J, "s-", color="#d62728", linewidth=2, markersize=3, label="R_J (judgment)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(layers, R_C, R_J, where=R_C > R_J, alpha=0.15, color="green", label="P2 holds")
    ax.fill_between(layers, R_C, R_J, where=R_C <= R_J, alpha=0.15, color="red", label="P2 fails")
    ax.set_xlabel("Layer"); ax.set_ylabel("R ratio")
    ax.set_title(f"P2 per Layer (R_C vs R_J) — {MODEL_NAME.split('/')[-1]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "v2_quick_p2.png"), dpi=150)
    plt.close(fig)

    # Fig 3: Convergence + R_code overlay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, R_all, "o-", color="#1f77b4", linewidth=2, markersize=3, label="R (cross-lingual)")
    ax.plot(layers, R_code, "s-", color="#8c564b", linewidth=2, markersize=3, label="R_code (NL↔code)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("R ratio")
    ax.set_title(f"Cross-lingual vs NL-Code Convergence — {MODEL_NAME.split('/')[-1]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "v2_quick_overlay.png"), dpi=150)
    plt.close(fig)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS: {MODEL_NAME.split('/')[-1]}")
    print(f"{'='*60}")
    print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Peak R (cross-lingual): layer {results['peak_R_all']['layer']}, "
          f"R={results['peak_R_all']['value']:.3f}")
    print(f"Peak R_code (NL-code): layer {results['peak_R_code']['layer']}, "
          f"R_code={results['peak_R_code']['value']:.3f}")
    p2_layers = results['p2_layers_where_RC_gt_RJ']
    print(f"P2 holds at layers: {p2_layers if p2_layers else 'NONE'}")
    print(f"\nFigures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
