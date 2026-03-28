"""Test harness for falsifiable predictions P1, P2, P2-dialect, P6, P7."""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from .metrics import discriminability_ratio, spacing_robustness, dialectal_continuum


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

    return PredictionResult(
        prediction_id="P2-dialect",
        supported=continuum_holds,
        effect_size=result_comp["R_cross_dialect"] - result_comp["R_cross_lingual"],
        p_value=0.0,  # TODO: bootstrap CI
        details={
            "comp": result_comp,
            "judg": result_judg,
            "continuum_holds_comp": result_comp["continuum_holds"],
            "continuum_holds_judg": result_judg["continuum_holds"],
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
