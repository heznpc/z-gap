#!/usr/bin/env python3
"""DEFERRED: Sparse Language Dimensions — requires power analysis before execution.

Audit finding: removing 4/384 dimensions produces angular deviation ~5.7°,
which may be below measurement noise. Run angular_deviation power analysis
before executing this experiment. Strategy B (run_strategy2_langpair.py)
answers a related question more cleanly via language-pair decomposition.

Original description:
---
Strategy 2: Sparse Language Dimensions — P2 failure reanalysis.

Hypothesis (Zhong et al., 2025): language identity is encoded in a small set of
sparse dimensions orthogonal to semantic content. If we project embeddings onto
the non-language subspace (removing those dimensions), R_C may increase —
revealing that the P2 failure (R_C < R_J) was driven by language-specific
encoding rather than genuine semantic divergence.

Pipeline:
  1. Load cached embeddings for each model
  2. Identify language-encoding dimensions (3 methods, prefer Method A)
  3. Project embeddings onto the orthogonal complement (non-language subspace)
  4. Recompute R_C, R_J on projected embeddings
  5. Controls: random dimension removal, semantic dimension removal
  6. Statistical testing via bootstrap

Reference: Zhong et al. (2025) "Language Lives in Sparse Dimensions" arXiv:2510.07213
"""

import json
import sys
import gc
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, MistralEmbedder, EmbeddingCache
from src.metrics import discriminability_ratio
from src.predictions import test_p2_cross_lingual_invariance

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"


# ---------------------------------------------------------------------------
# Step 1: Load embeddings (reuses existing cache)
# ---------------------------------------------------------------------------

def load_embeddings_for_model(model, ops, cache):
    """Load cached embeddings, returning {f"{op_id}_{lang}": vector} dict."""
    texts, keys = [], []
    for op in ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                texts.append(desc)
                keys.append(f"{op.id}_{lang}")
    embeddings_array = cache.get_or_compute(model, texts)
    return {k: embeddings_array[i] for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# Step 2: Identify language-encoding dimensions
# ---------------------------------------------------------------------------

def identify_language_dims_classifier(embeddings, op_ids, languages, n_dims=None):
    """Method A: Logistic regression language classifier.

    Train a one-vs-rest logistic regression to predict language from embedding.
    The weight matrix W has shape (n_languages, d). The row space of W spans
    the "language subspace." We take the top-k singular vectors of W as the
    language directions.

    Why Method A is preferred:
      - Directly optimizes for language-predictive directions (discriminative)
      - Naturally handles non-orthogonal language structure
      - Aligns with Zhong et al.'s finding that language is linearly decodable
      - SVD of weight matrix gives directions ranked by discriminative power,
        providing a principled ordering for the number-of-dims-to-remove sweep

    Returns:
        lang_dirs: np.ndarray of shape (k, d) — orthonormal language directions
        clf_accuracy: float — cross-val accuracy of the classifier
        singular_values: np.ndarray — singular values of W (for elbow criterion)
    """
    X, y = [], []
    for op_id in op_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                X.append(embeddings[key])
                y.append(lang)
    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train classifier with L2 regularization
    # C=1.0 is moderate regularization — prevents overfitting while allowing
    # the model to find the language subspace
    clf = LogisticRegression(
        max_iter=2000, random_state=42, C=1.0,
        multi_class="multinomial", solver="lbfgs",
    )

    # Cross-val accuracy tells us how much language signal exists
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    clf_accuracy = float(np.mean(cv_scores))

    # Fit on all data to get weight matrix
    clf.fit(X, y_enc)
    W = clf.coef_  # shape: (n_classes, d) or (1, d) for binary

    # SVD of weight matrix: U @ diag(s) @ Vt
    # The rows of Vt (or columns of V) are the principal directions
    # along which the classifier discriminates language
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    # Vt has shape (min(n_classes, d), d)
    # Each row is a direction in embedding space, ordered by importance

    # Determine number of dimensions to remove
    if n_dims is None:
        # Principled criterion: keep directions explaining > 5% of total
        # discriminative variance (squared singular values)
        s_sq = s ** 2
        s_sq_frac = s_sq / s_sq.sum()
        n_dims = int(np.sum(s_sq_frac > 0.05))
        n_dims = max(n_dims, 1)  # at least 1

    lang_dirs = Vt[:n_dims]  # shape: (n_dims, d)

    # Orthonormalize (SVD already gives orthonormal rows, but be safe)
    lang_dirs, _ = np.linalg.qr(lang_dirs.T)  # shape: (d, n_dims)
    lang_dirs = lang_dirs.T  # shape: (n_dims, d)

    return lang_dirs, clf_accuracy, s


def identify_language_dims_pca(embeddings, op_ids, languages, n_dims=None):
    """Method B: PCA on per-language mean embeddings.

    Compute the mean embedding for each language, then PCA on the (n_lang, d)
    matrix of means. Top-k components = language dimensions.

    Simpler than Method A but less discriminative — captures variance in means,
    which may include semantic variance if language-category distributions differ.

    Returns:
        lang_dirs: np.ndarray of shape (k, d) — orthonormal language directions
        explained_var_ratio: np.ndarray — explained variance ratios
    """
    # Compute per-language means
    lang_vecs = {lang: [] for lang in languages}
    for op_id in op_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                lang_vecs[lang].append(embeddings[key])

    means = np.array([np.mean(lang_vecs[lang], axis=0) for lang in languages])
    # Center
    grand_mean = means.mean(axis=0)
    centered = means - grand_mean

    # SVD
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = s ** 2
    explained_var_ratio = explained_var / explained_var.sum()

    if n_dims is None:
        # Keep components explaining > 10% of between-language variance
        n_dims = int(np.sum(explained_var_ratio > 0.10))
        n_dims = max(n_dims, 1)

    lang_dirs = Vt[:n_dims]
    return lang_dirs, explained_var_ratio


def identify_language_dims_difference(embeddings, op_ids, languages):
    """Method C: Difference vectors (mean_lang - grand_mean).

    Simplest approach: each language direction is the deviation of that
    language's mean embedding from the grand mean. Orthonormalize the
    resulting set.

    Returns:
        lang_dirs: np.ndarray of shape (k, d) — orthonormal directions
    """
    lang_vecs = {lang: [] for lang in languages}
    for op_id in op_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                lang_vecs[lang].append(embeddings[key])

    means = np.array([np.mean(lang_vecs[lang], axis=0) for lang in languages])
    grand_mean = means.mean(axis=0)
    diffs = means - grand_mean  # shape: (n_lang, d)

    # Orthonormalize via QR
    Q, R = np.linalg.qr(diffs.T)  # Q shape: (d, n_lang)
    # Keep columns where R diagonal is non-negligible
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    lang_dirs = Q[:, :rank].T  # shape: (rank, d)
    return lang_dirs


# ---------------------------------------------------------------------------
# Step 3: Project onto non-language subspace
# ---------------------------------------------------------------------------

def project_out_directions(embeddings, directions):
    """Project embeddings onto the orthogonal complement of the given directions.

    Given orthonormal directions D of shape (k, d), the projection matrix
    onto the orthogonal complement is:
        P_perp = I - D^T @ D

    For each embedding v, the projected embedding is:
        v_proj = v - D^T @ (D @ v)

    This removes exactly the language-encoding components while preserving
    all other structure (semantic, procedural, pragmatic).

    Args:
        embeddings: dict {key: np.ndarray of shape (d,)}
        directions: np.ndarray of shape (k, d), orthonormal rows

    Returns:
        projected: dict {key: np.ndarray of shape (d,)}
    """
    D = np.array(directions, dtype=np.float64)  # (k, d)
    projected = {}
    for key, vec in embeddings.items():
        v = vec.astype(np.float64)
        # Project out each direction: v_proj = v - sum_i (v . d_i) d_i
        coeffs = D @ v          # (k,)
        removal = D.T @ coeffs  # (d,)
        v_proj = v - removal
        projected[key] = v_proj.astype(np.float32)
    return projected


# ---------------------------------------------------------------------------
# Step 4: Recompute R_C and R_J after projection
# ---------------------------------------------------------------------------

def compute_R_with_bootstrap(embeddings, comp_ids, judg_ids, languages, n_boot=10000):
    """Compute R_C, R_J, and bootstrap CI for their difference.

    Returns dict with R_C, R_J, delta, p_value, ci_95, and component distances.
    """
    result_c = discriminability_ratio(embeddings, comp_ids, languages)
    result_j = discriminability_ratio(embeddings, judg_ids, languages)
    R_C = result_c["R"]
    R_J = result_j["R"]

    # Bootstrap for R_C - R_J
    rng = np.random.default_rng(42)
    d_intra_c = np.array(list(result_c["d_intra_per_op"].values()))
    d_intra_j = np.array(list(result_j["d_intra_per_op"].values()))

    boot_diffs = []
    for _ in range(n_boot):
        bc = rng.choice(d_intra_c, size=len(d_intra_c), replace=True)
        bj = rng.choice(d_intra_j, size=len(d_intra_j), replace=True)
        r_c = result_c["mean_d_inter"] / np.mean(bc) if np.mean(bc) > 1e-10 else 0
        r_j = result_j["mean_d_inter"] / np.mean(bj) if np.mean(bj) > 1e-10 else 0
        boot_diffs.append(r_c - r_j)

    boot_diffs = np.array(boot_diffs)
    p_value = float(np.mean(boot_diffs <= 0))

    return {
        "R_C": R_C,
        "R_J": R_J,
        "delta": R_C - R_J,
        "p_value": p_value,
        "ci_95": (float(np.percentile(boot_diffs, 2.5)),
                  float(np.percentile(boot_diffs, 97.5))),
        "d_intra_C": result_c["mean_d_intra"],
        "d_intra_J": result_j["mean_d_intra"],
        "d_inter_C": result_c["mean_d_inter"],
        "d_inter_J": result_j["mean_d_inter"],
    }


# ---------------------------------------------------------------------------
# Step 5: Controls
# ---------------------------------------------------------------------------

def random_direction_control(embeddings, comp_ids, judg_ids, languages,
                             n_dims, n_trials=50, seed=42):
    """Control 1: Remove the same number of RANDOM dimensions.

    If removing n_dims random directions produces a similar increase in R_C,
    then the effect is not specific to language dimensions — it is just
    a dimensionality reduction artifact.

    Returns distribution of R_C values under random removal.
    """
    rng = np.random.default_rng(seed)
    # Get embedding dimensionality from first vector
    d = next(iter(embeddings.values())).shape[0]

    R_C_random = []
    R_J_random = []
    for trial in range(n_trials):
        # Generate random orthonormal directions
        random_vecs = rng.standard_normal((n_dims, d))
        Q, _ = np.linalg.qr(random_vecs.T)
        random_dirs = Q[:, :n_dims].T  # (n_dims, d)

        projected = project_out_directions(embeddings, random_dirs)
        result_c = discriminability_ratio(projected, comp_ids, languages)
        result_j = discriminability_ratio(projected, judg_ids, languages)
        R_C_random.append(result_c["R"])
        R_J_random.append(result_j["R"])

    return {
        "R_C_random": R_C_random,
        "R_J_random": R_J_random,
        "R_C_mean": float(np.mean(R_C_random)),
        "R_C_std": float(np.std(R_C_random)),
        "R_J_mean": float(np.mean(R_J_random)),
        "R_J_std": float(np.std(R_J_random)),
    }


def semantic_direction_control(embeddings, comp_ids, judg_ids, languages):
    """Control 2: Remove the semantic (comp-vs-judg) direction.

    Train a classifier to separate computational from judgment embeddings.
    The discriminative direction IS the semantic direction. Removing it should
    destroy R (collapse toward 1.0), confirming R is measuring real semantic
    structure, not noise.

    This is the negative control: if removing language dims helps but removing
    semantic dims hurts, the two subspaces are genuinely orthogonal.
    """
    X, y = [], []
    for op_id in comp_ids + judg_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                X.append(embeddings[key])
                y.append("computational" if op_id in comp_ids else "judgment")
    X = np.array(X, dtype=np.float64)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    clf.fit(X, y_enc)

    # Weight vector = semantic direction (binary classifier: single direction)
    w = clf.coef_[0]  # shape: (d,)
    w_norm = w / np.linalg.norm(w)
    semantic_dir = w_norm.reshape(1, -1)  # (1, d)

    projected = project_out_directions(embeddings, semantic_dir)
    result_c = discriminability_ratio(projected, comp_ids, languages)
    result_j = discriminability_ratio(projected, judg_ids, languages)

    return {
        "R_C_after": result_c["R"],
        "R_J_after": result_j["R"],
        "d_intra_C": result_c["mean_d_intra"],
        "d_intra_J": result_j["mean_d_intra"],
        "d_inter_C": result_c["mean_d_inter"],
        "d_inter_J": result_j["mean_d_inter"],
    }


# ---------------------------------------------------------------------------
# Step 6: Dimension sweep — how many to remove?
# ---------------------------------------------------------------------------

def dimension_sweep(embeddings, comp_ids, judg_ids, languages, lang_dirs, max_k=None):
    """Sweep over k = 0, 1, ..., max_k language dimensions removed.

    For each k, project out the top-k language directions and recompute R_C, R_J.
    This produces a curve that should show:
      - R_C increasing as k increases (removing language noise from comp ops)
      - R_C plateauing once all language dims are removed
      - R_J relatively stable (judgment ops already language-invariant)

    The elbow point in R_C is the effective dimensionality of language encoding.
    """
    if max_k is None:
        max_k = lang_dirs.shape[0]
    max_k = min(max_k, lang_dirs.shape[0])

    sweep_results = []
    for k in range(max_k + 1):
        if k == 0:
            result_c = discriminability_ratio(embeddings, comp_ids, languages)
            result_j = discriminability_ratio(embeddings, judg_ids, languages)
        else:
            projected = project_out_directions(embeddings, lang_dirs[:k])
            result_c = discriminability_ratio(projected, comp_ids, languages)
            result_j = discriminability_ratio(projected, judg_ids, languages)

        sweep_results.append({
            "k": k,
            "R_C": result_c["R"],
            "R_J": result_j["R"],
            "d_intra_C": result_c["mean_d_intra"],
            "d_intra_J": result_j["mean_d_intra"],
            "d_inter_C": result_c["mean_d_inter"],
            "d_inter_J": result_j["mean_d_inter"],
        })

    return sweep_results


# ---------------------------------------------------------------------------
# Step 7: Visualizations
# ---------------------------------------------------------------------------

def plot_projection_comparison(baseline, projected, model_name, n_removed, output_path):
    """Bar chart: R_C and R_J before and after projection."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    before = [baseline["R_C"], baseline["R_J"]]
    after = [projected["R_C"], projected["R_J"]]

    bars1 = ax.bar(x - width/2, before, width, label="Original", color=["#2196F3", "#FF5722"], alpha=0.6)
    bars2 = ax.bar(x + width/2, after, width, label=f"Projected (k={n_removed})", color=["#2196F3", "#FF5722"], alpha=1.0)

    ax.set_ylabel("Discriminability Ratio (R)")
    ax.set_title(f"Strategy 2: Language Dimension Removal — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(["R_C (computational)", "R_J (judgment)"])
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    # Annotate values
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_sweep(sweep_results, model_name, output_path):
    """Line plot: R_C and R_J as a function of k (dims removed)."""
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in sweep_results]
    R_Cs = [r["R_C"] for r in sweep_results]
    R_Js = [r["R_J"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, R_Cs, "o-", color="#2196F3", label="R_C (computational)", markersize=6)
    ax.plot(ks, R_Js, "s-", color="#FF5722", label="R_J (judgment)", markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="R = 1")

    ax.set_xlabel("Number of language dimensions removed (k)")
    ax.set_ylabel("Discriminability Ratio (R)")
    ax.set_title(f"Dimension Sweep — {model_name}")
    ax.legend()
    ax.set_xticks(ks)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_random_control(baseline_R_C, projected_R_C, random_R_Cs,
                        model_name, n_removed, output_path):
    """Histogram of R_C under random removal, with baseline and projected marked."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(random_R_Cs, bins=20, alpha=0.6, color="gray", label="Random removal")
    ax.axvline(baseline_R_C, color="#2196F3", linestyle="--", linewidth=2, label=f"Original R_C = {baseline_R_C:.2f}")
    ax.axvline(projected_R_C, color="#4CAF50", linestyle="-", linewidth=2, label=f"Language-projected R_C = {projected_R_C:.2f}")

    ax.set_xlabel("R_C after removing k dimensions")
    ax.set_ylabel("Count")
    ax.set_title(f"Control: Random vs Language Dim Removal (k={n_removed}) — {model_name}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_component_decomposition(baseline, projected, model_name, output_path):
    """Show d_intra and d_inter before/after for both C and J."""
    import matplotlib.pyplot as plt

    labels = ["d_intra_C", "d_intra_J", "d_inter_C", "d_inter_J"]
    before = [baseline["d_intra_C"], baseline["d_intra_J"],
              baseline["d_inter_C"], baseline["d_inter_J"]]
    after = [projected["d_intra_C"], projected["d_intra_J"],
             projected["d_inter_C"], projected["d_inter_J"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, before, width, label="Original", alpha=0.6, color="#9E9E9E")
    ax.bar(x + width/2, after, width, label="After projection", alpha=0.9, color="#4CAF50")

    ax.set_ylabel("Mean cosine distance")
    ax.set_title(f"Component Decomposition — {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8: Interpretation logic
# ---------------------------------------------------------------------------

def interpret_results(baseline, projected, random_ctrl, semantic_ctrl, clf_accuracy):
    """Determine what the results tell us about the P2 failure.

    Decision tree:

    1. Is language decodable from embeddings? (clf_accuracy >> 0.2)
       NO  -> language signal is weak; sparse-dims hypothesis does not apply
       YES -> proceed

    2. Does R_C increase after language dimension removal?
       NO  -> the P2 failure is NOT driven by language encoding;
              the divergence is genuine semantic divergence
       YES -> proceed

    3. Is the R_C increase larger than under random removal?
       NO  -> the effect is a dimensionality-reduction artifact, not specific
              to language dimensions
       YES -> proceed

    4. Does removing semantic dimensions hurt R? (negative control)
       NO  -> R is not measuring real semantic structure (problematic)
       YES -> CONCLUSION: P2 failure is a measurement artifact.
              At the Z_sem level, computational operations DO converge
              across languages. The divergence is confined to the
              language-encoding subspace.
    """
    R_C_orig = baseline["R_C"]
    R_C_proj = projected["R_C"]
    R_C_random_mean = random_ctrl["R_C_mean"]
    R_C_random_std = random_ctrl["R_C_std"]
    R_C_semantic = semantic_ctrl["R_C_after"]

    findings = []

    # 1. Language decodability
    lang_decodable = clf_accuracy > 0.40  # well above chance (0.20 for 5 langs)
    findings.append(f"Language classifier accuracy: {clf_accuracy:.3f} "
                    f"({'strong signal' if lang_decodable else 'weak signal'})")

    # 2. R_C increase
    R_C_increased = R_C_proj > R_C_orig
    delta = R_C_proj - R_C_orig
    pct_change = delta / R_C_orig * 100 if R_C_orig > 0 else 0
    findings.append(f"R_C change: {R_C_orig:.3f} -> {R_C_proj:.3f} "
                    f"(delta={delta:+.3f}, {pct_change:+.1f}%)")

    # 3. Specificity vs random
    z_score = (R_C_proj - R_C_random_mean) / R_C_random_std if R_C_random_std > 1e-10 else 0
    specific = z_score > 2.0
    findings.append(f"Specificity: projected R_C z-score vs random = {z_score:.2f} "
                    f"({'specific' if specific else 'not specific'})")

    # 4. Negative control
    R_destroyed = R_C_semantic < R_C_orig * 0.5  # semantic removal cuts R by >50%
    findings.append(f"Semantic control: R_C {R_C_orig:.3f} -> {R_C_semantic:.3f} "
                    f"({'R destroyed as expected' if R_destroyed else 'R surprisingly robust'})")

    # Overall interpretation
    if not lang_decodable:
        conclusion = "INCONCLUSIVE"
        interpretation = (
            "Language signal is weak in these embeddings. The sparse-dimensions "
            "hypothesis does not strongly apply. The P2 failure may be driven by "
            "other factors (e.g., surface-form variation, cultural framing)."
        )
    elif R_C_increased and specific and R_destroyed:
        conclusion = "P2_FAILURE_IS_ARTIFACT"
        interpretation = (
            "The P2 failure is a measurement artifact. Removing language-encoding "
            "dimensions specifically increases R_C (more than random removal), and "
            "removing semantic dimensions destroys R. At the Z_sem level, "
            "computational operations DO converge across languages. The original "
            "divergence is confined to the language-encoding subspace."
        )
    elif R_C_increased and not specific:
        conclusion = "DIMENSIONALITY_ARTIFACT"
        interpretation = (
            "R_C increases after projection, but the effect is not specific to "
            "language dimensions (random removal produces similar gains). This "
            "suggests the P2 failure may partly reflect high-dimensional geometry "
            "artifacts, not language encoding per se."
        )
    elif not R_C_increased:
        conclusion = "GENUINE_DIVERGENCE"
        interpretation = (
            "R_C does NOT increase after removing language dimensions. The P2 "
            "failure reflects genuine semantic divergence in how computational "
            "operations are described across languages — not a language-encoding "
            "artifact. This is a real Z_sem-level phenomenon."
        )
    else:
        conclusion = "MIXED"
        interpretation = (
            "Results are mixed: R_C increases with language dim removal and the "
            "effect is specific, but the semantic control did not behave as "
            "expected. Further investigation needed."
        )

    return {
        "conclusion": conclusion,
        "interpretation": interpretation,
        "findings": findings,
        "metrics": {
            "clf_accuracy": clf_accuracy,
            "R_C_orig": R_C_orig,
            "R_C_projected": R_C_proj,
            "R_C_delta": delta,
            "R_C_pct_change": pct_change,
            "R_C_random_mean": R_C_random_mean,
            "R_C_random_std": R_C_random_std,
            "z_score_vs_random": z_score,
            "R_C_after_semantic_removal": R_C_semantic,
        },
    }


# ---------------------------------------------------------------------------
# Step 9: Per-category analysis (comp subcategories)
# ---------------------------------------------------------------------------

def per_subcategory_analysis(embeddings, projected, ops, languages):
    """Check if the projection effect varies by operation subcategory.

    Some computational operations may be more "universal" (math) while others
    more "culturally mediated" (string ops with language-specific tokens).
    The projection should help more for the latter.
    """
    # Group operations by subcategory inferred from the operation ID.
    # IDs follow the pattern comp_NN_name, where the name hints at domain.
    # We use the dialect_descriptions field which actually holds the tag list
    # due to constructor positional ordering in stimuli.py.
    subcats = {}
    for op in ops:
        if op.category != "computational":
            continue
        # dialect_descriptions holds the [domain, action] tags due to
        # constructor field ordering
        tags = op.dialect_descriptions
        if isinstance(tags, list) and len(tags) >= 1:
            tag = tags[0]
        else:
            # Fallback: extract domain from op.id (e.g., comp_26_transpose -> matrix guess)
            tag = "other"
        if tag not in subcats:
            subcats[tag] = []
        subcats[tag].append(op.id)

    results = {}
    for tag, op_ids in subcats.items():
        if len(op_ids) < 3:
            continue
        r_orig = discriminability_ratio(embeddings, op_ids, languages)
        r_proj = discriminability_ratio(projected, op_ids, languages)
        results[tag] = {
            "n_ops": len(op_ids),
            "R_orig": r_orig["R"],
            "R_proj": r_proj["R"],
            "delta": r_proj["R"] - r_orig["R"],
            "d_intra_orig": r_orig["mean_d_intra"],
            "d_intra_proj": r_proj["mean_d_intra"],
        }

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_strategy2_for_model(model, ops, comp_ids, judg_ids, all_ids, categories, cache):
    """Full Strategy 2 pipeline for one model."""
    model_name = model.name
    tag = model_name.replace("/", "_")
    print(f"\n{'='*60}")
    print(f"Strategy 2: {model_name} (dim={model.dimension})")
    print(f"{'='*60}")

    # --- Load embeddings ---
    embeddings = load_embeddings_for_model(model, ops, cache)
    print(f"  Loaded {len(embeddings)} embeddings (dim={model.dimension})")

    # --- Baseline R ---
    baseline = compute_R_with_bootstrap(embeddings, comp_ids, judg_ids, LANGUAGES)
    print(f"  Baseline: R_C={baseline['R_C']:.3f}, R_J={baseline['R_J']:.3f}, "
          f"delta={baseline['delta']:+.3f}")

    # --- Method A: Classifier-based language dimension identification ---
    print(f"\n  Identifying language dimensions (Method A: classifier)...")
    lang_dirs_A, clf_acc, singular_values = identify_language_dims_classifier(
        embeddings, all_ids, LANGUAGES
    )
    n_lang_dims = lang_dirs_A.shape[0]
    print(f"    Classifier accuracy: {clf_acc:.3f} (chance=0.20)")
    print(f"    Language dimensions identified: {n_lang_dims}")
    print(f"    Singular values: {singular_values[:6]}")

    # --- Method B: PCA (for comparison) ---
    lang_dirs_B, exp_var = identify_language_dims_pca(embeddings, all_ids, LANGUAGES)
    print(f"    PCA explained variance: {exp_var}")

    # --- Method C: Difference vectors (for comparison) ---
    lang_dirs_C = identify_language_dims_difference(embeddings, all_ids, LANGUAGES)
    print(f"    Difference vectors rank: {lang_dirs_C.shape[0]}")

    # --- Dimension sweep ---
    print(f"\n  Running dimension sweep (k=0..{n_lang_dims})...")
    sweep = dimension_sweep(embeddings, comp_ids, judg_ids, LANGUAGES, lang_dirs_A)
    for s in sweep:
        print(f"    k={s['k']}: R_C={s['R_C']:.3f}, R_J={s['R_J']:.3f}")

    # --- Project with optimal k (use all identified dims) ---
    projected_embs = project_out_directions(embeddings, lang_dirs_A)
    projected = compute_R_with_bootstrap(projected_embs, comp_ids, judg_ids, LANGUAGES)
    print(f"\n  Projected (k={n_lang_dims}): R_C={projected['R_C']:.3f}, "
          f"R_J={projected['R_J']:.3f}, delta={projected['delta']:+.3f}")

    # --- Control 1: Random dimension removal ---
    print(f"\n  Control 1: Random dimension removal (k={n_lang_dims}, 50 trials)...")
    random_ctrl = random_direction_control(
        embeddings, comp_ids, judg_ids, LANGUAGES, n_lang_dims
    )
    print(f"    Random R_C: {random_ctrl['R_C_mean']:.3f} +/- {random_ctrl['R_C_std']:.3f}")

    # --- Control 2: Semantic dimension removal ---
    print(f"  Control 2: Semantic dimension removal...")
    semantic_ctrl = semantic_direction_control(embeddings, comp_ids, judg_ids, LANGUAGES)
    print(f"    R_C after semantic removal: {semantic_ctrl['R_C_after']:.3f}")
    print(f"    R_J after semantic removal: {semantic_ctrl['R_J_after']:.3f}")

    # --- Per-subcategory analysis ---
    print(f"\n  Per-subcategory analysis:")
    subcat = per_subcategory_analysis(embeddings, projected_embs, ops, LANGUAGES)
    for tag, res in sorted(subcat.items(), key=lambda x: -abs(x[1]["delta"])):
        print(f"    {tag:12s} (n={res['n_ops']:2d}): "
              f"R {res['R_orig']:.3f} -> {res['R_proj']:.3f} "
              f"(delta={res['delta']:+.3f})")

    # --- Interpret ---
    interp = interpret_results(baseline, projected, random_ctrl, semantic_ctrl, clf_acc)
    print(f"\n  CONCLUSION: {interp['conclusion']}")
    print(f"  {interp['interpretation']}")

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_projection_comparison(
        baseline, projected, model_name, n_lang_dims,
        FIGURES_DIR / f"strategy2_comparison_{tag}.png"
    )
    plot_dimension_sweep(
        sweep, model_name,
        FIGURES_DIR / f"strategy2_sweep_{tag}.png"
    )
    plot_random_control(
        baseline["R_C"], projected["R_C"], random_ctrl["R_C_random"],
        model_name, n_lang_dims,
        FIGURES_DIR / f"strategy2_random_ctrl_{tag}.png"
    )
    plot_component_decomposition(
        baseline, projected, model_name,
        FIGURES_DIR / f"strategy2_components_{tag}.png"
    )

    return {
        "model": model_name,
        "dim": model.dimension,
        "baseline": {k: v for k, v in baseline.items() if not isinstance(v, tuple)},
        "baseline_ci_95": baseline["ci_95"],
        "projected": {k: v for k, v in projected.items() if not isinstance(v, tuple)},
        "projected_ci_95": projected["ci_95"],
        "n_lang_dims": n_lang_dims,
        "clf_accuracy": clf_acc,
        "singular_values": singular_values.tolist(),
        "sweep": sweep,
        "random_control": {
            "R_C_mean": random_ctrl["R_C_mean"],
            "R_C_std": random_ctrl["R_C_std"],
            "R_J_mean": random_ctrl["R_J_mean"],
            "R_J_std": random_ctrl["R_J_std"],
        },
        "semantic_control": semantic_ctrl,
        "subcategory_analysis": subcat,
        "interpretation": interp,
    }


def main():
    print("Strategy 2: Sparse Language Dimensions — P2 Failure Reanalysis")
    print("=" * 70)

    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    judg_ids = [op.id for op in ops if op.category == "judgment"]
    all_ids = comp_ids + judg_ids
    categories = {op.id: op.category for op in ops}
    print(f"Stimuli: {len(comp_ids)} comp + {len(judg_ids)} judg = "
          f"{len(all_ids)} ops x {len(LANGUAGES)} langs")

    cache = EmbeddingCache(CACHE_DIR)

    # Run on all models with cached embeddings
    models = [
        SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2"),   # 384d
        SentenceTransformerEmbedder("intfloat/multilingual-e5-small"),           # 384d
        SentenceTransformerEmbedder("intfloat/multilingual-e5-base"),            # 768d
        SentenceTransformerEmbedder("intfloat/multilingual-e5-large"),           # 1024d
        SentenceTransformerEmbedder("BAAI/bge-m3"),                              # 1024d
    ]

    all_results = []
    for model in models:
        try:
            result = run_strategy2_for_model(
                model, ops, comp_ids, judg_ids, all_ids, categories, cache
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model
            gc.collect()

    # --- Cross-model summary ---
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<40} {'dim':>5} | {'R_C':>6} {'R_C*':>6} {'delta':>7} | "
          f"{'rand':>6} {'z':>5} | {'Conclusion'}")
    print("-" * 100)
    for r in all_results:
        i = r["interpretation"]
        print(f"{r['model']:<40} {r['dim']:>5} | "
              f"{i['metrics']['R_C_orig']:>6.3f} {i['metrics']['R_C_projected']:>6.3f} "
              f"{i['metrics']['R_C_delta']:>+7.3f} | "
              f"{i['metrics']['R_C_random_mean']:>6.3f} "
              f"{i['metrics']['z_score_vs_random']:>5.2f} | "
              f"{i['conclusion']}")

    # --- Save results ---
    out_path = RESULTS_DIR / "strategy2_sparse_dims.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
