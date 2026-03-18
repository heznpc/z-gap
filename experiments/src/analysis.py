"""P2 failure diagnosis and P1 scale-convergence trend analysis."""

import numpy as np
from scipy import stats


def diagnose_p2_failure(
    result_c: dict,
    result_j: dict,
    per_op_details: list[dict],
) -> dict:
    """Decompose R_C < R_J into d_intra and d_inter contributions.

    R = d_inter / d_intra, so R_J > R_C can be caused by:
      (a) d_intra_J < d_intra_C  (judgment more cross-lingually invariant)
      (b) d_inter_J > d_inter_C  (judgment ops more semantically spread)
      (c) both
    """
    d_intra_C = result_c["mean_d_intra"]
    d_intra_J = result_j["mean_d_intra"]
    d_inter_C = result_c["mean_d_inter"]
    d_inter_J = result_j["mean_d_inter"]
    R_C = result_c["R"]
    R_J = result_j["R"]

    # Per-operation d_intra distributions
    comp_d_intras = [d["d_intra"] for d in per_op_details if d["category"] == "computational"]
    judg_d_intras = [d["d_intra"] for d in per_op_details if d["category"] == "judgment"]

    # Mann-Whitney U test: are the distributions significantly different?
    if comp_d_intras and judg_d_intras:
        u_stat, u_p = stats.mannwhitneyu(comp_d_intras, judg_d_intras, alternative="two-sided")
    else:
        u_stat, u_p = 0.0, 1.0

    # Per-operation d_inter (mean distance to others)
    comp_d_inters = [d["mean_d_inter"] for d in per_op_details if d["category"] == "computational"]
    judg_d_inters = [d["mean_d_inter"] for d in per_op_details if d["category"] == "judgment"]

    # Contribution analysis
    delta_d_intra = d_intra_C - d_intra_J  # positive = comp less invariant
    delta_d_inter = d_inter_C - d_inter_J  # negative = comp less spread

    # Which factor dominates?
    # R_C/R_J = (d_inter_C / d_intra_C) / (d_inter_J / d_intra_J)
    #         = (d_inter_C / d_inter_J) * (d_intra_J / d_intra_C)
    inter_ratio = d_inter_C / d_inter_J if d_inter_J > 1e-10 else 1.0
    intra_ratio = d_intra_J / d_intra_C if d_intra_C > 1e-10 else 1.0

    # Find outlier operations (highest and lowest d_intra)
    sorted_ops = sorted(per_op_details, key=lambda d: d["d_intra"])
    most_invariant = sorted_ops[:5]
    least_invariant = sorted_ops[-5:]

    return {
        "R_C": R_C,
        "R_J": R_J,
        "d_intra_C": d_intra_C,
        "d_intra_J": d_intra_J,
        "d_inter_C": d_inter_C,
        "d_inter_J": d_inter_J,
        "delta_d_intra": delta_d_intra,
        "delta_d_inter": delta_d_inter,
        "inter_ratio": inter_ratio,
        "intra_ratio": intra_ratio,
        "primary_driver": (
            "d_intra (judgment more invariant)" if abs(np.log(intra_ratio)) > abs(np.log(inter_ratio))
            else "d_inter (judgment more spread)"
        ),
        "mannwhitney_u": u_stat,
        "mannwhitney_p": u_p,
        "comp_d_intra_stats": {
            "mean": float(np.mean(comp_d_intras)),
            "std": float(np.std(comp_d_intras)),
            "median": float(np.median(comp_d_intras)),
        },
        "judg_d_intra_stats": {
            "mean": float(np.mean(judg_d_intras)),
            "std": float(np.std(judg_d_intras)),
            "median": float(np.median(judg_d_intras)),
        },
        "comp_d_inter_stats": {
            "mean": float(np.mean(comp_d_inters)),
            "std": float(np.std(comp_d_inters)),
        },
        "judg_d_inter_stats": {
            "mean": float(np.mean(judg_d_inters)),
            "std": float(np.std(judg_d_inters)),
        },
        "most_invariant_ops": [{"op_id": d["op_id"], "category": d["category"], "d_intra": d["d_intra"]} for d in most_invariant],
        "least_invariant_ops": [{"op_id": d["op_id"], "category": d["category"], "d_intra": d["d_intra"]} for d in least_invariant],
    }


def compute_p1_trend(all_results: list[dict]) -> dict:
    """P1: R increases monotonically with model scale (dimension).

    Uses Spearman rank correlation between model dimension and R.
    """
    if len(all_results) < 3:
        return {
            "supported": False,
            "reason": f"Need >= 3 models for trend analysis, got {len(all_results)}",
            "n_models": len(all_results),
        }

    sorted_results = sorted(all_results, key=lambda r: r["dim"])
    dims = [r["dim"] for r in sorted_results]
    R_Cs = [r["P2"]["R_C"] for r in sorted_results]
    R_Js = [r["P2"]["R_J"] for r in sorted_results]
    R_totals = [rc + rj for rc, rj in zip(R_Cs, R_Js)]

    rho_C, p_C = stats.spearmanr(dims, R_Cs)
    rho_J, p_J = stats.spearmanr(dims, R_Js)
    rho_total, p_total = stats.spearmanr(dims, R_totals)

    return {
        "supported": rho_total > 0 and p_total < 0.1,
        "n_models": len(all_results),
        "dims": dims,
        "R_Cs": R_Cs,
        "R_Js": R_Js,
        "R_totals": R_totals,
        "model_names": [r["model"] for r in sorted_results],
        "rho_C": float(rho_C),
        "p_C": float(p_C),
        "rho_J": float(rho_J),
        "p_J": float(p_J),
        "rho_total": float(rho_total),
        "p_total": float(p_total),
    }
