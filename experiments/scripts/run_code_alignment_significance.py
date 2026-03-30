#!/usr/bin/env python3
"""Permutation test and bootstrap CI for NL-Code alignment R_code.

Loads cached embeddings, recomputes d_match / d_mismatch,
then runs:
  - Permutation test (10,000 permutations) for R_code > 1
  - Bootstrap 95% CI for R_code
  - Cohen's d effect size
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.code_alignment import CODE_EQUIVALENTS

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "embeddings"

N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
RNG_SEED = 42


def load_embeddings(model_name: str):
    """Load cached NL and code embeddings for a given model."""
    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]

    model = SentenceTransformerEmbedder(model_name)
    cache = EmbeddingCache(CACHE_DIR)

    # NL texts - must be in same order as original to hit cache
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

    # Code texts
    code_texts, code_keys = [], []
    for op_id in comp_ids:
        if op_id in CODE_EQUIVALENTS:
            code_texts.append(CODE_EQUIVALENTS[op_id])
            code_keys.append(op_id)

    code_array = cache.get_or_compute(model, code_texts)
    code_embeddings = {k: code_array[i] for i, k in enumerate(code_keys)}

    return nl_embeddings, code_embeddings, comp_ids


def compute_pairwise_distances(nl_embeddings, code_embeddings, comp_ids, pairing=None):
    """Compute d_match and d_mismatch lists.

    pairing: if provided, a dict mapping op_id -> op_id (shuffled assignment).
             When None, uses identity (correct pairing).

    Returns (d_match_array, d_mismatch_array) as numpy arrays.
    """
    if pairing is None:
        pairing = {op_id: op_id for op_id in comp_ids}

    d_match_list = []
    d_mismatch_list = []

    for op_id in comp_ids:
        # The code this op is "paired" to under the current pairing
        paired_code_id = pairing[op_id]
        if paired_code_id not in code_embeddings:
            continue

        code_vec = code_embeddings[paired_code_id]

        # d_match: NL descriptions of this op -> code of its paired op
        for lang in LANGUAGES:
            nl_key = f"{op_id}_{lang}"
            if nl_key in nl_embeddings:
                d_match_list.append(cosine(nl_embeddings[nl_key], code_vec))

        # d_mismatch: NL descriptions of this op -> code of OTHER (non-paired) ops
        other_ids = [oid for oid in comp_ids if oid != op_id and oid in code_embeddings]
        for other_id in other_ids[:10]:
            for lang in LANGUAGES[:2]:  # en + ko, matching original
                nl_key = f"{op_id}_{lang}"
                if nl_key in nl_embeddings:
                    d_mismatch_list.append(cosine(nl_embeddings[nl_key], code_embeddings[other_id]))

    return np.array(d_match_list), np.array(d_mismatch_list)


def compute_R_code(d_match, d_mismatch):
    """R_code = mean(d_mismatch) / mean(d_match)."""
    m = np.mean(d_match)
    if m < 1e-10:
        return float("inf")
    return float(np.mean(d_mismatch) / m)


def permutation_test(nl_embeddings, code_embeddings, comp_ids, observed_R, n_perm, rng):
    """Permutation test: shuffle NL-code pairings, recompute R_code.

    H0: NL-code pairing is arbitrary (R_code ~ 1).
    p-value: proportion of permuted R_code >= observed R_code.
    """
    valid_ids = [op_id for op_id in comp_ids if op_id in code_embeddings]
    perm_R_values = np.zeros(n_perm)

    for i in range(n_perm):
        # Randomly permute which code snippet each NL op is "matched" to
        shuffled = rng.permutation(valid_ids)
        pairing = {op_id: shuffled[j] for j, op_id in enumerate(valid_ids)}
        d_match_perm, d_mismatch_perm = compute_pairwise_distances(
            nl_embeddings, code_embeddings, comp_ids, pairing
        )
        perm_R_values[i] = compute_R_code(d_match_perm, d_mismatch_perm)

        if (i + 1) % 1000 == 0:
            print(f"    Permutation {i+1}/{n_perm}...")

    # One-sided: how often does permuted R >= observed R?
    p_value = float(np.mean(perm_R_values >= observed_R))
    return p_value, perm_R_values


def bootstrap_ci(d_match, d_mismatch, n_boot, rng, ci=0.95):
    """Bootstrap 95% CI for R_code by resampling match/mismatch pairs."""
    boot_R = np.zeros(n_boot)
    n_match = len(d_match)
    n_mismatch = len(d_mismatch)

    for i in range(n_boot):
        idx_m = rng.integers(0, n_match, size=n_match)
        idx_mm = rng.integers(0, n_mismatch, size=n_mismatch)
        boot_R[i] = compute_R_code(d_match[idx_m], d_mismatch[idx_mm])

    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_R, 100 * alpha))
    hi = float(np.percentile(boot_R, 100 * (1 - alpha)))
    return lo, hi, boot_R


def cohens_d(d_match, d_mismatch):
    """Cohen's d for the difference between d_mismatch and d_match distributions."""
    n1, n2 = len(d_match), len(d_mismatch)
    s1, s2 = np.std(d_match, ddof=1), np.std(d_mismatch, ddof=1)
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-10:
        return float("inf")
    return float((np.mean(d_mismatch) - np.mean(d_match)) / s_pooled)


def run_significance_for_model(model_name, label, rng):
    """Run full significance analysis for one model."""
    print(f"\n{'='*60}")
    print(f"  {label} ({model_name})")
    print(f"{'='*60}")

    print("  Loading cached embeddings...")
    nl_emb, code_emb, comp_ids = load_embeddings(model_name)
    print(f"  NL embeddings: {len(nl_emb)}, Code embeddings: {len(code_emb)}")

    # Observed statistics
    print("  Computing observed d_match / d_mismatch...")
    d_match, d_mismatch = compute_pairwise_distances(nl_emb, code_emb, comp_ids)
    observed_R = compute_R_code(d_match, d_mismatch)
    print(f"  Observed R_code = {observed_R:.4f}")
    print(f"  d_match: mean={np.mean(d_match):.4f}, std={np.std(d_match):.4f}, n={len(d_match)}")
    print(f"  d_mismatch: mean={np.mean(d_mismatch):.4f}, std={np.std(d_mismatch):.4f}, n={len(d_mismatch)}")

    # Cohen's d
    d_effect = cohens_d(d_match, d_mismatch)
    print(f"  Cohen's d = {d_effect:.4f}")

    # Permutation test
    print(f"\n  Running permutation test ({N_PERMUTATIONS} permutations)...")
    p_value, perm_R_values = permutation_test(
        nl_emb, code_emb, comp_ids, observed_R, N_PERMUTATIONS, rng
    )
    print(f"  p-value = {p_value:.6f}")
    print(f"  Permuted R_code: mean={np.mean(perm_R_values):.4f}, "
          f"std={np.std(perm_R_values):.4f}, "
          f"range=[{np.min(perm_R_values):.4f}, {np.max(perm_R_values):.4f}]")

    # Bootstrap CI
    print(f"\n  Computing bootstrap 95% CI ({N_BOOTSTRAP} resamples)...")
    ci_lo, ci_hi, boot_R = bootstrap_ci(d_match, d_mismatch, N_BOOTSTRAP, rng)
    print(f"  95% CI for R_code: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "model": model_name,
        "R_code": observed_R,
        "d_match_mean": float(np.mean(d_match)),
        "d_match_std": float(np.std(d_match)),
        "d_mismatch_mean": float(np.mean(d_mismatch)),
        "d_mismatch_std": float(np.std(d_mismatch)),
        "n_match_pairs": len(d_match),
        "n_mismatch_pairs": len(d_mismatch),
        "cohens_d": d_effect,
        "permutation_test": {
            "n_permutations": N_PERMUTATIONS,
            "p_value": p_value,
            "null_R_mean": float(np.mean(perm_R_values)),
            "null_R_std": float(np.std(perm_R_values)),
        },
        "bootstrap_ci": {
            "n_bootstrap": N_BOOTSTRAP,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
        },
    }


def main():
    print("=" * 60)
    print("NL-Code Alignment: Significance Testing")
    print(f"  Permutations: {N_PERMUTATIONS}, Bootstrap: {N_BOOTSTRAP}")
    print(f"  Seed: {RNG_SEED}")
    print("=" * 60)

    rng = np.random.default_rng(RNG_SEED)

    results = {}

    # UniXcoder
    results["unixcoder"] = run_significance_for_model(
        "microsoft/unixcoder-base", "UniXcoder", rng
    )

    # MiniLM
    results["minilm"] = run_significance_for_model(
        "paraphrase-multilingual-MiniLM-L12-v2", "MiniLM-L12", rng
    )

    # Save results
    out_path = RESULTS_DIR / "code_alignment_significance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Final summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'R_code':>7} {'p-value':>10} {'95% CI':>20} {'Cohen d':>9}")
    print("-" * 70)
    for name, r in results.items():
        ci = f"[{r['bootstrap_ci']['ci_95_lower']:.3f}, {r['bootstrap_ci']['ci_95_upper']:.3f}]"
        pv = r["permutation_test"]["p_value"]
        pv_str = f"{pv:.4f}" if pv > 0 else f"<{1/N_PERMUTATIONS:.4f}"
        print(f"{name:<20} {r['R_code']:>7.3f} {pv_str:>10} {ci:>20} {r['cohens_d']:>9.3f}")


if __name__ == "__main__":
    main()
