"""V2 Visualization: Layer-wise convergence curves, CKA heatmaps, RSA plots."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

COLORS = {
    "en": "#1f77b4", "ko": "#ff7f0e", "zh": "#2ca02c",
    "ar": "#d62728", "es": "#9467bd", "code": "#8c564b",
}


def plot_convergence_curve(results: dict, model_name: str, out_path: str):
    """Plot R(l) across layers — the core Z_sem localization figure."""
    R = results["R"]
    layers = np.arange(len(R))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(layers, R, "o-", color="#1f77b4", linewidth=2, markersize=4)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="R=1 (chance)")

    # Mark peak
    peak_layer = np.argmax(R)
    ax.annotate(f"peak: layer {peak_layer}\nR={R[peak_layer]:.2f}",
                xy=(peak_layer, R[peak_layer]),
                xytext=(peak_layer + 2, R[peak_layer] + 0.1),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R = d_inter / d_intra", fontsize=12)
    ax.set_title(f"Cross-lingual Convergence Curve — {model_name}", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_modal_alignment(results: dict, model_name: str, out_path: str):
    """Plot per-layer R_code(l) — NL-code alignment across layers."""
    R = results["R_code"]
    layers = np.arange(len(R))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(layers, R, "s-", color="#8c564b", linewidth=2, markersize=4, label="R_code")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    peak = np.argmax(R)
    ax.annotate(f"peak: layer {peak}\nR_code={R[peak]:.2f}",
                xy=(peak, R[peak]),
                xytext=(peak + 2, R[peak] + 0.05),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R_code = d_mismatch / d_match", fontsize=12)
    ax.set_title(f"NL-Code Alignment per Layer — {model_name}", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_and_code_overlay(
    conv_results: dict, code_results: dict, model_name: str, out_path: str
):
    """Overlay cross-lingual R and R_code on same plot — key comparison."""
    R_cross = conv_results["R"]
    R_code = code_results["R_code"]
    layers = np.arange(len(R_cross))

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.plot(layers, R_cross, "o-", color="#1f77b4", linewidth=2,
             markersize=3, label="R (cross-lingual)")
    ax1.plot(layers, R_code, "s-", color="#8c564b", linewidth=2,
             markersize=3, label="R_code (NL↔code)")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    peak_cross = np.argmax(R_cross)
    peak_code = np.argmax(R_code)
    ax1.axvline(x=peak_cross, color="#1f77b4", linestyle=":", alpha=0.5)
    ax1.axvline(x=peak_code, color="#8c564b", linestyle=":", alpha=0.5)

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("R ratio", fontsize=12)
    ax1.set_title(f"Cross-lingual vs NL-Code Convergence — {model_name}", fontsize=13)
    ax1.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cka_heatmap(cka_matrix: np.ndarray, model_a: str, model_b: str,
                     lang: str, out_path: str):
    """Plot CKA heatmap between two models."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(cka_matrix, cmap="viridis", aspect="auto",
                   vmin=0, vmax=1, origin="lower")
    ax.set_xlabel(f"{model_b} layer", fontsize=11)
    ax.set_ylabel(f"{model_a} layer", fontsize=11)
    ax.set_title(f"Linear CKA ({lang}) — {model_a} vs {model_b}", fontsize=12)
    plt.colorbar(im, ax=ax, label="CKA")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_p2_per_layer(results: dict, model_name: str, out_path: str):
    """Plot R_C and R_J across layers for P2 re-test."""
    R_C = results["R_C"]
    R_J = results["R_J"]
    p_values = results["p_values"]
    layers = np.arange(len(R_C))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1],
                                    sharex=True)

    ax1.plot(layers, R_C, "o-", color="#2ca02c", linewidth=2, markersize=3,
             label="R_C (computational)")
    ax1.plot(layers, R_J, "s-", color="#d62728", linewidth=2, markersize=3,
             label="R_J (judgment)")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.fill_between(layers, R_C, R_J, where=R_C > R_J,
                     alpha=0.15, color="green", label="P2 holds")
    ax1.fill_between(layers, R_C, R_J, where=R_C <= R_J,
                     alpha=0.15, color="red", label="P2 fails")
    ax1.set_ylabel("R ratio", fontsize=12)
    ax1.set_title(f"P2 Re-test per Layer — {model_name}", fontsize=13)
    ax1.legend(fontsize=9)

    # p-value panel
    sig = p_values < 0.05
    ax2.bar(layers[sig], -np.log10(p_values[sig] + 1e-10), color="green", alpha=0.6)
    ax2.bar(layers[~sig], -np.log10(p_values[~sig] + 1e-10), color="gray", alpha=0.3)
    ax2.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.5,
                label="p=0.05")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("-log10(p)", fontsize=10)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rsa_curves(rsa_results: dict, model_name: str, out_path: str):
    """Plot RSA Spearman rho across layers — cross-lingual and NL-code."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cross-lingual RSA
    for (la, lb), rho_arr in rsa_results["cross_lingual"].items():
        layers = np.arange(len(rho_arr))
        ax1.plot(layers, rho_arr, "-", linewidth=1.5, alpha=0.7, label=f"{la}-{lb}")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Spearman ρ", fontsize=11)
    ax1.set_title(f"Cross-lingual RSA — {model_name}", fontsize=12)
    ax1.legend(fontsize=8, ncol=2)

    # NL-Code RSA
    if rsa_results.get("nl_code"):
        for lang, rho_arr in rsa_results["nl_code"].items():
            layers = np.arange(len(rho_arr))
            ax2.plot(layers, rho_arr, "o-", color=COLORS.get(lang, "gray"),
                     linewidth=1.5, markersize=3, label=lang)
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Spearman ρ (NL vs Code RDM)", fontsize=11)
        ax2.set_title(f"NL-Code RSA — {model_name}", fontsize=12)
        ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tier_comparison(tier_results: dict, model_name: str, out_path: str):
    """Compare convergence curves across complexity tiers."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    tier_colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    tier_labels = {1: "Tier 1 (one-liner)", 2: "Tier 2 (multi-step)",
                   3: "Tier 3 (compositional)"}

    for tier, result in sorted(tier_results.items()):
        R = result["R"]
        layers = np.arange(len(R))
        ax.plot(layers, R, "o-", color=tier_colors.get(tier, "gray"),
                linewidth=2, markersize=3, label=tier_labels.get(tier, f"Tier {tier}"))

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R = d_inter / d_intra", fontsize=12)
    ax.set_title(f"Convergence by Complexity Tier — {model_name}", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
