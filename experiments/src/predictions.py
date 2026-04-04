"""Test harness for falsifiable predictions P1, P2, P2-kNN, P2-dialect, P6, P7."""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from .metrics import (
    discriminability_ratio, spacing_robustness, dialectal_continuum,
    knn_accuracy_by_category, compute_topology_suite,
)


@dataclass
class PredictionResult:
    prediction_id: str
    supported: bool
    effect_size: float
    p_value: float
    details: dict


def test_p2_cross_lingual_invariance(
    embeddings: dict[str, np.ndarray],
    comp_ids: list[str],
    judg_ids: list[str],
    languages: list[str],
) -> PredictionResult:
    """P2: R_C > R_J — discriminability ratio is higher for computational than judgment ops."""
    result_c = discriminability_ratio(embeddings, comp_ids, languages)
    result_j = discriminability_ratio(embeddings, judg_ids, languages)
    R_C = result_c["R"]
    R_J = result_j["R"]

    # Bootstrap confidence interval for R_C - R_J
    n_boot = 10000
    rng = np.random.default_rng(42)
    d_intra_c = list(result_c["d_intra_per_op"].values())
    d_intra_j = list(result_j["d_intra_per_op"].values())

    boot_diffs = []
    for _ in range(n_boot):
        bc = rng.choice(d_intra_c, size=len(d_intra_c), replace=True)
        bj = rng.choice(d_intra_j, size=len(d_intra_j), replace=True)
        mean_bc = np.mean(bc)
        mean_bj = np.mean(bj)
        r_c = result_c["mean_d_inter"] / mean_bc if mean_bc > 1e-10 else 0
        r_j = result_j["mean_d_inter"] / mean_bj if mean_bj > 1e-10 else 0
        boot_diffs.append(r_c - r_j)

    boot_diffs = np.array(boot_diffs)
    p_value = float(np.mean(boot_diffs <= 0))  # one-sided: P(R_C <= R_J)
    effect_size = float(R_C - R_J)

    return PredictionResult(
        prediction_id="P2",
        supported=R_C > R_J and p_value < 0.05,
        effect_size=effect_size,
        p_value=p_value,
        details={
            "R_C": R_C, "R_J": R_J,
            "R_C_d_intra": result_c["mean_d_intra"],
            "R_J_d_intra": result_j["mean_d_intra"],
            "ci_95": (float(np.percentile(boot_diffs, 2.5)),
                      float(np.percentile(boot_diffs, 97.5))),
        }
    )


def test_p2_dialectal(
    embeddings: dict[str, np.ndarray],
    comp_ids: list[str],
    judg_ids: list[str],
    languages: list[str],
    dialects: dict[str, list[str]],
) -> PredictionResult:
    """P2-dialect: R_cross_dialect > R_cross_lingual (continuum prediction).

    Tests that the communicability gap is continuous — cross-dialectal R
    falls between within-dialect and cross-lingual R.
    """
    result_comp = dialectal_continuum(embeddings, comp_ids, languages, dialects)
    result_judg = dialectal_continuum(embeddings, judg_ids, languages, dialects)

    continuum_holds = result_comp["continuum_holds"] and result_judg["continuum_holds"]

    # Bootstrap CI for effect size
    n_boot = 10000
    rng = np.random.default_rng(42)
    boot_effects = []
    for _ in range(n_boot):
        # Resample operation ids with replacement
        boot_ids = rng.choice(comp_ids, size=len(comp_ids), replace=True).tolist()
        boot_comp = dialectal_continuum(embeddings, boot_ids, languages, dialects)
        boot_effects.append(boot_comp["R_cross_dialect"] - boot_comp["R_cross_lingual"])
    boot_effects = np.array(boot_effects)
    p_value = float(np.mean(boot_effects <= 0))

    return PredictionResult(
        prediction_id="P2-dialect",
        supported=continuum_holds,
        effect_size=result_comp["R_cross_dialect"] - result_comp["R_cross_lingual"],
        p_value=p_value,
        details={
            "comp": result_comp,
            "judg": result_judg,
            "continuum_holds_comp": result_comp["continuum_holds"],
            "continuum_holds_judg": result_judg["continuum_holds"],
            "bootstrap_ci_95": (float(np.percentile(boot_effects, 2.5)),
                                float(np.percentile(boot_effects, 97.5))),
        },
    )


def test_p7_spacing_robustness(
    embeddings_correct: dict[str, np.ndarray],
    embeddings_variants: dict[str, dict[str, np.ndarray]],
    operation_ids: list[str],
    n_boot: int = 10000,
) -> PredictionResult:
    """P7: R_spacing > 1 — spacing variation << semantic variation."""
    from .metrics import cosine_distance
    result = spacing_robustness(embeddings_correct, embeddings_variants, operation_ids)
    R_sp = result["R_spacing"]

    # Collect raw distance arrays for bootstrap
    d_spacing_list = []
    for op_id in operation_ids:
        if op_id not in embeddings_correct or op_id not in embeddings_variants:
            continue
        correct = embeddings_correct[op_id]
        for vname, vvec in embeddings_variants[op_id].items():
            if vname == "correct":
                continue
            d_spacing_list.append(cosine_distance(correct, vvec))

    vecs = list(embeddings_correct.values())
    rng = np.random.default_rng(42)
    target = min(500, len(vecs) * 2)
    indices = np.empty((0, 2), dtype=int)
    while len(indices) < target:
        new = rng.choice(len(vecs), size=(target - len(indices) + 50, 2), replace=True)
        new = new[new[:, 0] != new[:, 1]]
        indices = np.vstack([indices, new]) if len(indices) > 0 else new
    indices = indices[:target]
    d_semantic_list = [cosine_distance(vecs[i], vecs[j]) for i, j in indices]

    d_spacing_arr = np.array(d_spacing_list)
    d_semantic_arr = np.array(d_semantic_list)

    # Bootstrap: resample both distributions, compute R_spacing
    boot_Rs = []
    for _ in range(n_boot):
        bs = rng.choice(d_spacing_arr, size=len(d_spacing_arr), replace=True)
        bd = rng.choice(d_semantic_arr, size=len(d_semantic_arr), replace=True)
        mean_bs = np.mean(bs)
        mean_bd = np.mean(bd)
        r = mean_bd / mean_bs if mean_bs > 1e-10 else 0.0
        boot_Rs.append(r)

    boot_Rs = np.array(boot_Rs)
    p_value = float(np.mean(boot_Rs <= 1.0))
    ci_95 = (float(np.percentile(boot_Rs, 2.5)), float(np.percentile(boot_Rs, 97.5)))

    result["ci_95"] = ci_95
    result["p_value"] = p_value

    return PredictionResult(
        prediction_id="P7",
        supported=R_sp > 1.0 and p_value < 0.05,
        effect_size=R_sp,
        p_value=p_value,
        details=result,
    )


def test_p2_knn(
    embeddings: dict[str, np.ndarray],
    comp_ids: list[str],
    judg_ids: list[str],
    languages: list[str],
    k_values: list[int] = None,
    n_permutations: int = 10000,
) -> PredictionResult:
    """P2-kNN: kNN_accuracy_C > kNN_accuracy_J at the local topology level.

    Strategy 4 (Aristotelian reanalysis): if P2 fails at the distance
    level (R_C < R_J), it may still hold at the topology level.
    Gröger et al. (2026) showed global distance metrics are confounded
    but local neighborhood structure persists.

    Test: for each (op, lang) query, check if the same operation in
    another language appears among the k nearest neighbors.  Compare
    hit rates between computational and judgment operations.

    Statistical test: permutation test on the per-operation accuracy
    difference.  Under H0, category labels are exchangeable.

    Returns PredictionResult with primary_k=5 as the headline result.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    primary_k = 5  # headline k for supported/effect_size/p_value

    all_ids = comp_ids + judg_ids
    categories = {oid: "computational" for oid in comp_ids}
    categories.update({oid: "judgment" for oid in judg_ids})

    # Run the full topology suite
    topo = compute_topology_suite(
        embeddings, all_ids, languages, categories, k_values
    )

    knn = topo["knn"]
    cat_c = knn["by_category"].get("computational", {})
    cat_j = knn["by_category"].get("judgment", {})

    # --- Permutation test on per-operation accuracy@k ---
    # Collect per-operation accuracy (averaged across languages) for primary_k
    op_accs = {}  # op_id -> mean accuracy@primary_k across its language queries
    for q in knn["knn"]["per_query"] if "knn" in knn else []:
        op_accs.setdefault(q["op_id"], []).append(q["per_k_hit"][primary_k])

    # The knn result is nested; get per_query from the right place
    # knn_result structure: topo["knn"]["by_category"][cat]["per_operation"][op_id]
    comp_op_accs = []
    judg_op_accs = []
    for op_id in comp_ids:
        per_op = cat_c.get("per_operation", {}).get(op_id, {})
        acc = per_op.get("accuracy", {}).get(primary_k, 0.0)
        comp_op_accs.append(acc)
    for op_id in judg_ids:
        per_op = cat_j.get("per_operation", {}).get(op_id, {})
        acc = per_op.get("accuracy", {}).get(primary_k, 0.0)
        judg_op_accs.append(acc)

    comp_op_accs = np.array(comp_op_accs)
    judg_op_accs = np.array(judg_op_accs)
    observed_diff = float(np.mean(comp_op_accs) - np.mean(judg_op_accs))

    # Permutation test: shuffle category labels among operations
    all_op_accs = np.concatenate([comp_op_accs, judg_op_accs])
    n_comp = len(comp_op_accs)
    rng = np.random.default_rng(42)
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(all_op_accs)
        perm_diffs[i] = np.mean(perm[:n_comp]) - np.mean(perm[n_comp:])

    # One-sided p-value: P(perm_diff >= observed_diff) under H0
    p_value = float(np.mean(perm_diffs >= observed_diff))

    # Per-k summary
    per_k_details = {}
    for k in k_values:
        acc_c = cat_c.get("accuracy", {}).get(k, 0.0)
        acc_j = cat_j.get("accuracy", {}).get(k, 0.0)
        rec_c = cat_c.get("recall", {}).get(k, 0.0)
        rec_j = cat_j.get("recall", {}).get(k, 0.0)
        per_k_details[k] = {
            "accuracy_C": acc_c,
            "accuracy_J": acc_j,
            "delta_accuracy": acc_c - acc_j,
            "recall_C": rec_c,
            "recall_J": rec_j,
            "delta_recall": rec_c - rec_j,
            "supported": acc_c > acc_j,
        }

    acc_c_primary = cat_c.get("accuracy", {}).get(primary_k, 0.0)
    acc_j_primary = cat_j.get("accuracy", {}).get(primary_k, 0.0)
    mrr_c = cat_c.get("mrr", 0.0)
    mrr_j = cat_j.get("mrr", 0.0)

    supported = acc_c_primary > acc_j_primary and p_value < 0.05

    return PredictionResult(
        prediction_id="P2-kNN",
        supported=supported,
        effect_size=observed_diff,
        p_value=p_value,
        details={
            "primary_k": primary_k,
            "per_k": per_k_details,
            "mrr_C": mrr_c,
            "mrr_J": mrr_j,
            "delta_mrr": mrr_c - mrr_j,
            "neighborhood_overlap_C": topo["neighborhood_overlap"]["by_category"].get("computational", 0.0),
            "neighborhood_overlap_J": topo["neighborhood_overlap"]["by_category"].get("judgment", 0.0),
            "hubness_skewness": topo["hubness"].get("skewness", 0.0),
            "hubness_detected": topo["hubness"].get("hubness_detected", False),
            "random_baseline": topo["random_baseline"],
            "permutation_ci_95": (
                float(np.percentile(perm_diffs, 2.5)),
                float(np.percentile(perm_diffs, 97.5)),
            ),
            "topology_suite": topo,
        },
    )
