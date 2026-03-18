"""Test harness for falsifiable predictions P1, P2, P6, P7."""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from .metrics import discriminability_ratio, spacing_robustness


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


def test_p7_spacing_robustness(
    embeddings_correct: dict[str, np.ndarray],
    embeddings_variants: dict[str, dict[str, np.ndarray]],
    operation_ids: list[str],
) -> PredictionResult:
    """P7: R_spacing > 1 — spacing variation << semantic variation."""
    result = spacing_robustness(embeddings_correct, embeddings_variants, operation_ids)
    R_sp = result["R_spacing"]

    return PredictionResult(
        prediction_id="P7",
        supported=R_sp > 1.0,
        effect_size=R_sp,
        p_value=0.0,  # placeholder — proper bootstrap in full version
        details=result,
    )
