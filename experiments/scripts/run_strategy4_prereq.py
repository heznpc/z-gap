#!/usr/bin/env python3
"""Strategy 4 prerequisite check: is k-NN redundant with distance ratio R?

Audit finding: if d_intra_J < d_intra_C (the known P2 result), then
kNN_J > kNN_C may follow automatically -- the "reanalysis" would just
measure the same thing twice.

This script runs a quick (<5 min) diagnostic to determine whether to
proceed with the full Strategy 4 k-NN analysis.

Decision tree
-------------
1.  Compute Spearman rho between per-operation d_intra and per-operation
    kNN accuracy@5.
    - |rho| > 0.7  =>  metrics are redundant  =>  STOP
    - |rho| <= 0.7 =>  enough independence   =>  proceed with CSLS

2.  If proceeding, compute CSLS-corrected k-NN accuracy and compare
    with raw k-NN accuracy.  CSLS normalizes for local density (hubness).

3.  Decompose k-NN first-hits by source language to check whether a
    single language pair dominates (pair proximity vs operation convergence).

4.  Final verdict:
    - rho > 0.7  =>  report correlation, do not pursue kNN separately
    - rho <= 0.7  AND  CSLS flips comp/judg ordering  =>  report both
    - rho <= 0.7  AND  CSLS confirms raw  =>  report CSLS as robust version
"""

import json
import sys
import gc
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import EmbeddingCache
from src.metrics import compute_d_intra, knn_cross_lingual_accuracy

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "embeddings"

# Use the three models from the existing P1/P2 pipeline
MODELS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", 384, "118M"),
    ("paraphrase-multilingual-mpnet-base-v2", 768, "278M"),
    ("intfloat/multilingual-e5-large", 1024, "560M"),
]

PRIMARY_K = 5


# ------------------------------------------------------------------ #
#  CSLS (Cross-domain Similarity Local Scaling)                       #
# ------------------------------------------------------------------ #
#                                                                      #
#  For each point x_i, define:                                         #
#    r_k(x_i) = mean cosine similarity to x_i's k nearest neighbors   #
#                                                                      #
#  CSLS(x_i, x_j) = 2 * sim(x_i, x_j) - r_k(x_i) - r_k(x_j)       #
#                                                                      #
#  This penalizes hub points (high r_k) and rewards isolated points,  #
#  correcting for the hubness problem in high-dimensional spaces.      #
# ------------------------------------------------------------------ #

def _build_all_vectors(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build aligned arrays of vectors, keys, op_ids, langs."""
    keys, vecs, ops, langs = [], [], [], []
    for op_id in operation_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                keys.append(key)
                vecs.append(embeddings[key])
                ops.append(op_id)
                langs.append(lang)
    if not vecs:
        return np.empty((0, 0)), [], [], []
    return np.array(vecs), keys, ops, langs


def compute_csls_knn_accuracy(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    k_csls: int = 10,
    k_eval: int = 5,
) -> dict:
    """k-NN cross-lingual accuracy using CSLS-reranked neighbors.

    Parameters
    ----------
    k_csls : int
        Number of neighbors for computing the CSLS penalty term r_k.
    k_eval : int
        Number of neighbors to evaluate accuracy at (accuracy@k_eval).

    Returns
    -------
    dict with per-query results, overall accuracy, and per-category accuracy.
    """
    X, keys, op_list, lang_list = _build_all_vectors(
        embeddings, operation_ids, languages
    )
    N = len(keys)
    if N == 0:
        return {"accuracy": 0.0, "per_query": []}

    # Cosine similarity matrix (embeddings are L2-normalized, so dot product)
    sim_matrix = X @ X.T  # (N, N)

    # Step 1: compute r_k(x_i) = mean similarity to k_csls nearest neighbors
    # (excluding self)
    r_k = np.zeros(N)
    for i in range(N):
        sims_i = sim_matrix[i].copy()
        sims_i[i] = -np.inf  # exclude self
        top_k_idx = np.argpartition(sims_i, -k_csls)[-k_csls:]
        r_k[i] = np.mean(sims_i[top_k_idx])

    # Step 2: CSLS score matrix
    # CSLS(i, j) = 2 * sim(i, j) - r_k(i) - r_k(j)
    csls_matrix = 2 * sim_matrix - r_k[:, None] - r_k[None, :]
    np.fill_diagonal(csls_matrix, -np.inf)  # exclude self

    # Step 3: for each query, find top-k_eval by CSLS score and check hits
    per_query = []
    for i in range(N):
        query_op = op_list[i]
        query_lang = lang_list[i]

        # Targets: same operation, different language
        targets = set()
        for j in range(N):
            if op_list[j] == query_op and lang_list[j] != query_lang:
                targets.add(j)

        # Top-k by CSLS
        csls_row = csls_matrix[i]
        top_k_idx = np.argpartition(csls_row, -k_eval)[-k_eval:]
        top_k_set = set(top_k_idx.tolist())

        hit = len(top_k_set & targets) > 0

        # First-hit rank (by CSLS)
        sorted_by_csls = np.argsort(csls_row)[::-1]  # descending
        first_rank = None
        for rank_idx, j in enumerate(sorted_by_csls[:N]):
            if j in targets:
                first_rank = rank_idx + 1
                break

        per_query.append({
            "key": keys[i],
            "op_id": query_op,
            "lang": query_lang,
            "hit": hit,
            "first_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        })

    accuracy = float(np.mean([q["hit"] for q in per_query]))
    mrr = float(np.mean([q["reciprocal_rank"] for q in per_query]))

    return {
        "accuracy": accuracy,
        "mrr": mrr,
        "k_csls": k_csls,
        "k_eval": k_eval,
        "n_queries": N,
        "per_query": per_query,
    }


def compute_language_diversity(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    categories: dict[str, str],
) -> dict:
    """For each query, report which language the first k-NN hit came from.

    Returns per-category breakdown and a diversity score.
    A diversity score near 1.0 = hits spread evenly across languages.
    A diversity score near 0.0 = one language dominates first hits.
    """
    X, keys, op_list, lang_list = _build_all_vectors(
        embeddings, operation_ids, languages
    )
    N = len(keys)
    if N == 0:
        return {}

    # Cosine distance matrix
    dist_matrix = cdist(X, X, metric="cosine")
    np.fill_diagonal(dist_matrix, np.inf)
    sorted_indices = np.argsort(dist_matrix, axis=1)

    # For each query: find the first neighbor that is the same op, different lang
    first_hit_lang: dict[str, list[str]] = {"computational": [], "judgment": []}

    for i in range(N):
        query_op = op_list[i]
        query_lang = lang_list[i]
        cat = categories.get(query_op, "unknown")
        if cat not in first_hit_lang:
            continue

        for j in sorted_indices[i]:
            if op_list[j] == query_op and lang_list[j] != query_lang:
                first_hit_lang[cat].append(lang_list[j])
                break

    result = {}
    for cat, hit_langs in first_hit_lang.items():
        if not hit_langs:
            result[cat] = {
                "first_hit_distribution": {},
                "dominant_lang": None,
                "dominant_fraction": 0.0,
                "diversity_score": 0.0,
            }
            continue

        counter = Counter(hit_langs)
        total = len(hit_langs)
        distribution = {lang: count / total for lang, count in counter.most_common()}

        # Diversity: normalized entropy
        # H = -sum(p * log(p)) / log(n_langs)
        probs = np.array(list(distribution.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(len(languages))
        diversity = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        dominant_lang, dominant_count = counter.most_common(1)[0]

        result[cat] = {
            "first_hit_distribution": distribution,
            "dominant_lang": dominant_lang,
            "dominant_fraction": dominant_count / total,
            "diversity_score": diversity,
            "n_queries": total,
        }

    return result


def run_prereq_for_model(
    model_name: str, dim: int, params: str
) -> dict:
    """Run the full prerequisite check for one model."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name} ({params} params, {dim}d)")
    print(f"{'=' * 60}")

    # --- Load embeddings from cache ---
    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    judg_ids = [op.id for op in ops if op.category == "judgment"]
    all_ids = comp_ids + judg_ids
    categories = {oid: "computational" for oid in comp_ids}
    categories.update({oid: "judgment" for oid in judg_ids})

    cache = EmbeddingCache(CACHE_DIR)

    # Build the same text list used by the embedding pipeline so the
    # cache hash matches.  We use cache.get() directly with the
    # canonical model name to avoid importing sentence_transformers.
    canonical_name = f"st_{model_name.split('/')[-1]}"

    texts, emb_keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                emb_keys.append(f"{op.id}_{lang}")

    embeddings_array = cache.get(canonical_name, texts)
    if embeddings_array is None:
        raise RuntimeError(
            f"No cached embeddings for {canonical_name}. "
            f"Run the main experiment pipeline first to populate the cache."
        )
    print(f"  Cache hit for {canonical_name} ({len(texts)} texts)")
    embeddings = {k: embeddings_array[i] for i, k in enumerate(emb_keys)}

    del embeddings_array
    gc.collect()

    # ============================================================
    # CHECK 1: Spearman correlation between d_intra and kNN@5
    # ============================================================
    print("\n  [1/3] Metric independence: d_intra vs kNN accuracy@5")

    # Per-operation d_intra
    d_intra_all = compute_d_intra(embeddings, all_ids, LANGUAGES)

    # Per-operation kNN accuracy@5
    knn_full = knn_cross_lingual_accuracy(
        embeddings, all_ids, LANGUAGES, k_values=[PRIMARY_K]
    )

    # Aggregate per-operation: mean accuracy@5 across languages for each op
    op_knn_acc: dict[str, float] = {}
    for q in knn_full["per_query"]:
        op_knn_acc.setdefault(q["op_id"], []).append(
            float(q["per_k_hit"][PRIMARY_K])
        )
    op_knn_acc = {op: float(np.mean(vals)) for op, vals in op_knn_acc.items()}

    # Align: only operations present in both
    shared_ops = sorted(set(d_intra_all.keys()) & set(op_knn_acc.keys()))
    d_intra_vec = np.array([d_intra_all[op] for op in shared_ops])
    knn_acc_vec = np.array([op_knn_acc[op] for op in shared_ops])

    # Note: we expect NEGATIVE correlation (lower d_intra = tighter cluster
    # = easier for kNN to find cross-lingual neighbors)
    rho, rho_p = stats.spearmanr(d_intra_vec, knn_acc_vec)

    print(f"    n_operations = {len(shared_ops)}")
    print(f"    Spearman rho = {rho:+.4f}  (p = {rho_p:.2e})")
    print(f"    |rho| = {abs(rho):.4f}")

    # Also compute per-category to verify direction
    comp_d = [d_intra_all[op] for op in shared_ops if categories[op] == "computational"]
    comp_k = [op_knn_acc[op] for op in shared_ops if categories[op] == "computational"]
    judg_d = [d_intra_all[op] for op in shared_ops if categories[op] == "judgment"]
    judg_k = [op_knn_acc[op] for op in shared_ops if categories[op] == "judgment"]

    print(f"    Computational: mean d_intra={np.mean(comp_d):.4f}, mean kNN@5={np.mean(comp_k):.4f}")
    print(f"    Judgment:      mean d_intra={np.mean(judg_d):.4f}, mean kNN@5={np.mean(judg_k):.4f}")

    redundant = abs(rho) > 0.7
    if redundant:
        print(f"\n    ** REDUNDANT: |rho| = {abs(rho):.4f} > 0.7 **")
        print(f"    kNN accuracy is largely determined by d_intra.")
        print(f"    Recommendation: do NOT pursue kNN as a separate strategy.")
    else:
        print(f"\n    INDEPENDENT: |rho| = {abs(rho):.4f} <= 0.7")
        print(f"    kNN captures information beyond d_intra. Proceeding to CSLS check.")

    # ============================================================
    # CHECK 2: CSLS-corrected kNN vs raw kNN
    # ============================================================
    print(f"\n  [2/3] CSLS-corrected kNN accuracy@{PRIMARY_K}")

    csls_result = compute_csls_knn_accuracy(
        embeddings, all_ids, LANGUAGES, k_csls=10, k_eval=PRIMARY_K
    )

    # Per-category CSLS accuracy
    csls_by_cat: dict[str, list[bool]] = {}
    for q in csls_result["per_query"]:
        cat = categories.get(q["op_id"], "unknown")
        csls_by_cat.setdefault(cat, []).append(q["hit"])

    csls_acc_c = float(np.mean(csls_by_cat.get("computational", [0])))
    csls_acc_j = float(np.mean(csls_by_cat.get("judgment", [0])))

    # Raw kNN per-category for comparison
    raw_by_cat: dict[str, list[bool]] = {}
    for q in knn_full["per_query"]:
        cat = categories.get(q["op_id"], "unknown")
        raw_by_cat.setdefault(cat, []).append(q["per_k_hit"][PRIMARY_K])

    raw_acc_c = float(np.mean(raw_by_cat.get("computational", [0])))
    raw_acc_j = float(np.mean(raw_by_cat.get("judgment", [0])))

    raw_ordering = "C > J" if raw_acc_c > raw_acc_j else "J >= C"
    csls_ordering = "C > J" if csls_acc_c > csls_acc_j else "J >= C"
    ordering_flipped = raw_ordering != csls_ordering

    print(f"    Raw  kNN@{PRIMARY_K}:  Comp={raw_acc_c:.4f}  Judg={raw_acc_j:.4f}  [{raw_ordering}]")
    print(f"    CSLS kNN@{PRIMARY_K}:  Comp={csls_acc_c:.4f}  Judg={csls_acc_j:.4f}  [{csls_ordering}]")
    print(f"    CSLS MRR:     {csls_result['mrr']:.4f}")

    if ordering_flipped:
        print(f"\n    ** CSLS FLIPS the comp/judg ordering **")
        print(f"    Raw hubness was distorting the raw kNN result.")
    else:
        print(f"\n    CSLS CONFIRMS the raw ordering ({csls_ordering}).")
        print(f"    Raw kNN result is robust to hubness correction.")

    # ============================================================
    # CHECK 3: Language-pair decomposition
    # ============================================================
    print(f"\n  [3/3] Language diversity of first kNN hits")

    lang_div = compute_language_diversity(
        embeddings, all_ids, LANGUAGES, categories
    )

    for cat in ["computational", "judgment"]:
        info = lang_div.get(cat, {})
        dist = info.get("first_hit_distribution", {})
        dominant = info.get("dominant_lang", "?")
        dom_frac = info.get("dominant_fraction", 0.0)
        diversity = info.get("diversity_score", 0.0)

        print(f"    {cat.capitalize()}:")
        print(f"      Diversity score (normalized entropy): {diversity:.4f}")
        print(f"      Dominant language: {dominant} ({dom_frac:.1%})")
        for lang, frac in sorted(dist.items(), key=lambda x: -x[1]):
            bar = "#" * int(frac * 40)
            print(f"        {lang}: {frac:.1%} {bar}")

        if dom_frac > 0.6:
            print(f"      ** WARNING: {dominant} dominates >60% of first hits **")
            print(f"      kNN may be measuring pair proximity, not operation convergence.")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n  {'=' * 50}")
    print(f"  VERDICT for {model_name.split('/')[-1]}")
    print(f"  {'=' * 50}")

    if redundant:
        verdict = "STOP"
        print(f"  |rho| = {abs(rho):.4f} > 0.7  =>  kNN is redundant with R")
        print(f"  Do NOT pursue Strategy 4 as a separate analysis.")
        print(f"  Report the correlation as evidence of redundancy.")
    elif ordering_flipped:
        verdict = "PROCEED_REPORT_BOTH"
        print(f"  |rho| = {abs(rho):.4f} <= 0.7  =>  kNN adds information")
        print(f"  CSLS FLIPS the raw ordering  =>  report both raw and CSLS")
        print(f"  The hubness correction changes the qualitative conclusion.")
    else:
        verdict = "PROCEED_CSLS"
        print(f"  |rho| = {abs(rho):.4f} <= 0.7  =>  kNN adds information")
        print(f"  CSLS CONFIRMS raw ordering   =>  report CSLS as robust version")

    return {
        "model": model_name,
        "dim": dim,
        "params": params,
        "check1_independence": {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "abs_rho": float(abs(rho)),
            "redundant": redundant,
            "n_operations": len(shared_ops),
            "comp_mean_d_intra": float(np.mean(comp_d)),
            "judg_mean_d_intra": float(np.mean(judg_d)),
            "comp_mean_knn5": float(np.mean(comp_k)),
            "judg_mean_knn5": float(np.mean(judg_k)),
        },
        "check2_csls": {
            "raw_acc_C": raw_acc_c,
            "raw_acc_J": raw_acc_j,
            "raw_ordering": raw_ordering,
            "csls_acc_C": csls_acc_c,
            "csls_acc_J": csls_acc_j,
            "csls_ordering": csls_ordering,
            "csls_mrr": csls_result["mrr"],
            "ordering_flipped": ordering_flipped,
        },
        "check3_language_diversity": lang_div,
        "verdict": verdict,
    }


def main():
    print("=" * 60)
    print("Strategy 4 Prerequisite Check")
    print("Is k-NN redundant with distance ratio R?")
    print("=" * 60)

    all_results = []
    for model_name, dim, params in MODELS:
        result = run_prereq_for_model(model_name, dim, params)
        all_results.append(result)

    # ============================================================
    # Cross-model summary
    # ============================================================
    print(f"\n\n{'=' * 60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n{'Model':<35s}  {'|rho|':>6s}  {'Raw':>8s}  {'CSLS':>8s}  {'Verdict':>12s}")
    print(f"{'_' * 75}")
    for r in all_results:
        m = r["model"].split("/")[-1][:30]
        rho = r["check1_independence"]["abs_rho"]
        raw = r["check2_csls"]["raw_ordering"]
        csls = r["check2_csls"]["csls_ordering"]
        v = r["verdict"]
        print(f"{m:<35s}  {rho:>6.4f}  {raw:>8s}  {csls:>8s}  {v:>12s}")

    # Consensus verdict
    verdicts = [r["verdict"] for r in all_results]
    if all(v == "STOP" for v in verdicts):
        consensus = "STOP"
        print(f"\n  CONSENSUS: kNN is redundant across ALL models.")
        print(f"  Recommendation: do NOT pursue Strategy 4.")
        print(f"  Report Spearman correlations as evidence of redundancy.")
    elif any(v == "STOP" for v in verdicts):
        consensus = "MIXED"
        print(f"\n  MIXED: kNN is redundant for some models but not others.")
        print(f"  Recommendation: investigate model-by-model; proceed with caution.")
    elif any(v == "PROCEED_REPORT_BOTH" for v in verdicts):
        consensus = "PROCEED_REPORT_BOTH"
        print(f"\n  CONSENSUS: Proceed with Strategy 4, report both raw and CSLS.")
        print(f"  CSLS flips the ordering for at least one model.")
    else:
        consensus = "PROCEED_CSLS"
        print(f"\n  CONSENSUS: Proceed with Strategy 4, use CSLS as robust metric.")
        print(f"  Raw kNN ordering is confirmed by CSLS across all models.")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "strategy4_prereq_results.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    output = {
        "consensus_verdict": consensus,
        "models": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_convert)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
