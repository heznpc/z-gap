"""V2 Analysis: Layer-wise convergence, CKA, RSA, cross-modal alignment.

All analyses operate on pre-extracted hidden states (numpy arrays).
No GPU required — runs on CPU after extraction.
"""

import numpy as np
from itertools import combinations
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


# ────────────────────────────────────────────────────────────────
# 4.1  Layer-wise Convergence Curve
# ────────────────────────────────────────────────────────────────

def layer_convergence_curve(
    nl_states: dict[str, dict[str, np.ndarray]],
    languages: list[str],
    n_layers: int,
) -> dict:
    """Compute cross-lingual invariance R(l) at each layer.

    Args:
        nl_states: {op_id: {lang: array(n_layers+1, dim)}}
        languages: list of language codes
        n_layers: number of transformer layers

    Returns:
        {"R": array(n_layers+1), "d_intra": ..., "d_inter": ...}
    """
    op_ids = list(nl_states.keys())
    R = np.zeros(n_layers + 1)
    d_intra_all = np.zeros(n_layers + 1)
    d_inter_all = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        d_intra = []  # same op, different languages
        d_inter = []  # different ops, same language

        # d_intra: cross-lingual distances for same operation
        for op_id in op_ids:
            vecs = []
            for lang in languages:
                if lang in nl_states[op_id]:
                    vecs.append(nl_states[op_id][lang][layer])
            for a, b in combinations(vecs, 2):
                d_intra.append(cosine(a, b))

        # d_inter: cross-operation distances within same language
        for lang in languages:
            lang_vecs = []
            for op_id in op_ids:
                if lang in nl_states[op_id]:
                    lang_vecs.append(nl_states[op_id][lang][layer])
            for a, b in combinations(lang_vecs, 2):
                d_inter.append(cosine(a, b))

        mean_intra = np.mean(d_intra) if d_intra else 1e-10
        mean_inter = np.mean(d_inter) if d_inter else 0.0
        R[layer] = mean_inter / max(mean_intra, 1e-10)
        d_intra_all[layer] = mean_intra
        d_inter_all[layer] = mean_inter

    return {"R": R, "d_intra": d_intra_all, "d_inter": d_inter_all}


# ────────────────────────────────────────────────────────────────
# 4.2  Cross-modal Layer Alignment (NL ↔ Code)
# ────────────────────────────────────────────────────────────────

def cross_modal_alignment(
    nl_states: dict[str, dict[str, np.ndarray]],
    code_states: dict[str, np.ndarray],
    languages: list[str],
    n_layers: int,
    n_mismatch_sample: int = 10,
    seed: int = 42,
) -> dict:
    """Compute per-layer R_code(l) = d_mismatch(l) / d_match(l).

    Args:
        nl_states: {op_id: {lang: array(n_layers+1, dim)}}
        code_states: {op_id: array(n_layers+1, dim)}
        languages: language codes
        n_layers: number of layers
    """
    rng = np.random.default_rng(seed)
    comp_ids = [oid for oid in nl_states if oid in code_states]

    R_code = np.zeros(n_layers + 1)
    d_match_per_layer = np.zeros(n_layers + 1)
    d_mismatch_per_layer = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        d_match = []
        d_mismatch = []

        for op_id in comp_ids:
            code_vec = code_states[op_id][layer]
            for lang in languages:
                if lang not in nl_states[op_id]:
                    continue
                nl_vec = nl_states[op_id][lang][layer]
                d_match.append(cosine(nl_vec, code_vec))

                # Sample mismatches
                others = [oid for oid in comp_ids if oid != op_id]
                sampled = rng.choice(
                    others, size=min(n_mismatch_sample, len(others)), replace=False
                )
                for other_id in sampled:
                    d_mismatch.append(cosine(nl_vec, code_states[other_id][layer]))

        mean_match = np.mean(d_match) if d_match else 1e-10
        mean_mismatch = np.mean(d_mismatch) if d_mismatch else 0.0
        R_code[layer] = mean_mismatch / max(mean_match, 1e-10)
        d_match_per_layer[layer] = mean_match
        d_mismatch_per_layer[layer] = mean_mismatch

    return {
        "R_code": R_code,
        "d_match": d_match_per_layer,
        "d_mismatch": d_mismatch_per_layer,
    }


# ────────────────────────────────────────────────────────────────
# 4.3  CKA (Centered Kernel Alignment)
# ────────────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n_samples, dim_x)
        Y: (n_samples, dim_y)

    Returns: CKA similarity in [0, 1].
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices (linear kernel)
    # HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = np.linalg.norm(XtY, "fro") ** 2
    hsic_xx = np.linalg.norm(XtX, "fro") ** 2
    hsic_yy = np.linalg.norm(YtY, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


def cross_model_cka(
    states_a: dict[str, dict[str, np.ndarray]],
    states_b: dict[str, dict[str, np.ndarray]],
    op_ids: list[str],
    lang: str,
    n_layers_a: int,
    n_layers_b: int,
) -> np.ndarray:
    """Compute CKA matrix between two models' layer representations.

    Args:
        states_a, states_b: {op_id: {lang: array(n_layers+1, dim)}}
        op_ids: operations to compare
        lang: language to use
        n_layers_a, n_layers_b: number of layers per model

    Returns: CKA matrix of shape (n_layers_a+1, n_layers_b+1)
    """
    cka_matrix = np.zeros((n_layers_a + 1, n_layers_b + 1))

    for la in range(n_layers_a + 1):
        X = np.stack([states_a[op][lang][la] for op in op_ids
                      if op in states_a and lang in states_a[op]])
        for lb in range(n_layers_b + 1):
            Y = np.stack([states_b[op][lang][lb] for op in op_ids
                          if op in states_b and lang in states_b[op]])
            if len(X) == len(Y) and len(X) > 1:
                cka_matrix[la, lb] = linear_cka(X, Y)

    return cka_matrix


# ────────────────────────────────────────────────────────────────
# 4.4  P2 per-layer retest
# ────────────────────────────────────────────────────────────────

def discriminability_ratio_at_layer(
    nl_states: dict[str, dict[str, np.ndarray]],
    op_ids: list[str],
    languages: list[str],
    layer: int,
) -> float:
    """Compute R = d_inter / d_intra at a specific layer for given ops."""
    d_intra, d_inter = [], []

    for op_id in op_ids:
        vecs = []
        for lang in languages:
            if op_id in nl_states and lang in nl_states[op_id]:
                vecs.append(nl_states[op_id][lang][layer])
        for a, b in combinations(vecs, 2):
            d_intra.append(cosine(a, b))

    for lang in languages:
        lang_vecs = []
        for op_id in op_ids:
            if op_id in nl_states and lang in nl_states[op_id]:
                lang_vecs.append(nl_states[op_id][lang][layer])
        for a, b in combinations(lang_vecs, 2):
            d_inter.append(cosine(a, b))

    mean_intra = np.mean(d_intra) if d_intra else 1e-10
    mean_inter = np.mean(d_inter) if d_inter else 0.0
    return float(mean_inter / max(mean_intra, 1e-10))


def p2_per_layer(
    nl_states: dict[str, dict[str, np.ndarray]],
    comp_ids: list[str],
    judg_ids: list[str],
    languages: list[str],
    n_layers: int,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """Test P2 (R_C > R_J) at each layer with permutation test.

    Returns: {"R_C": array, "R_J": array, "p_values": array}
    """
    rng = np.random.default_rng(seed)
    all_ids = comp_ids + judg_ids
    n_comp = len(comp_ids)

    R_C = np.zeros(n_layers + 1)
    R_J = np.zeros(n_layers + 1)
    p_values = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        R_C[layer] = discriminability_ratio_at_layer(nl_states, comp_ids, languages, layer)
        R_J[layer] = discriminability_ratio_at_layer(nl_states, judg_ids, languages, layer)
        observed_diff = R_C[layer] - R_J[layer]

        # Permutation test: shuffle comp/judg labels
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(all_ids)
            perm_comp = list(perm[:n_comp])
            perm_judg = list(perm[n_comp:])
            r_c = discriminability_ratio_at_layer(nl_states, perm_comp, languages, layer)
            r_j = discriminability_ratio_at_layer(nl_states, perm_judg, languages, layer)
            if (r_c - r_j) >= observed_diff:
                count += 1
        p_values[layer] = count / n_perm

    return {"R_C": R_C, "R_J": R_J, "p_values": p_values}


# ────────────────────────────────────────────────────────────────
# 4.5  RSA (Representational Similarity Analysis)
# ────────────────────────────────────────────────────────────────

def compute_rdm(vectors: np.ndarray) -> np.ndarray:
    """Compute Representational Dissimilarity Matrix.

    Args: vectors of shape (n_items, dim)
    Returns: RDM of shape (n_items, n_items)
    """
    n = len(vectors)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine(vectors[i], vectors[j])
            rdm[i, j] = d
            rdm[j, i] = d
    return rdm


def upper_tri(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangular elements (excluding diagonal)."""
    idx = np.triu_indices(len(matrix), k=1)
    return matrix[idx]


def rsa_analysis(
    nl_states: dict[str, dict[str, np.ndarray]],
    code_states: dict[str, np.ndarray] | None,
    op_ids: list[str],
    languages: list[str],
    n_layers: int,
) -> dict:
    """RSA: Spearman correlation between RDMs across layers.

    Returns:
        {
            "cross_lingual": {(lang_a, lang_b): array(n_layers+1)},
            "nl_code": {lang: array(n_layers+1)} (if code_states provided),
        }
    """
    results = {"cross_lingual": {}, "nl_code": {}}

    for la, lb in combinations(languages, 2):
        results["cross_lingual"][(la, lb)] = np.zeros(n_layers + 1)
    if code_states:
        for lang in languages:
            results["nl_code"][lang] = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        # Build RDMs per language
        rdms = {}
        for lang in languages:
            vecs = []
            valid_ops = []
            for op_id in op_ids:
                if op_id in nl_states and lang in nl_states[op_id]:
                    vecs.append(nl_states[op_id][lang][layer])
                    valid_ops.append(op_id)
            if len(vecs) > 2:
                rdms[lang] = (compute_rdm(np.stack(vecs)), valid_ops)

        # Code RDM
        if code_states:
            code_vecs = []
            code_ops = []
            for op_id in op_ids:
                if op_id in code_states:
                    code_vecs.append(code_states[op_id][layer])
                    code_ops.append(op_id)
            if len(code_vecs) > 2:
                rdm_code = compute_rdm(np.stack(code_vecs))

        # Cross-lingual RSA
        for la, lb in combinations(languages, 2):
            if la in rdms and lb in rdms:
                rdm_a, ops_a = rdms[la]
                rdm_b, ops_b = rdms[lb]
                # Use intersection of ops
                common = [op for op in ops_a if op in ops_b]
                if len(common) > 2:
                    idx_a = [ops_a.index(op) for op in common]
                    idx_b = [ops_b.index(op) for op in common]
                    sub_a = rdm_a[np.ix_(idx_a, idx_a)]
                    sub_b = rdm_b[np.ix_(idx_b, idx_b)]
                    rho, _ = spearmanr(upper_tri(sub_a), upper_tri(sub_b))
                    results["cross_lingual"][(la, lb)][layer] = rho

        # NL-Code RSA
        if code_states and len(code_vecs) > 2:
            for lang in languages:
                if lang in rdms:
                    rdm_nl, ops_nl = rdms[lang]
                    common = [op for op in ops_nl if op in code_ops]
                    if len(common) > 2:
                        idx_nl = [ops_nl.index(op) for op in common]
                        idx_code = [code_ops.index(op) for op in common]
                        sub_nl = rdm_nl[np.ix_(idx_nl, idx_nl)]
                        sub_code = rdm_code[np.ix_(idx_code, idx_code)]
                        rho, _ = spearmanr(upper_tri(sub_nl), upper_tri(sub_code))
                        results["nl_code"][lang][layer] = rho

    return results


# ────────────────────────────────────────────────────────────────
# Tier-wise complexity analysis
# ────────────────────────────────────────────────────────────────

def tier_comparison(
    nl_states: dict[str, dict[str, np.ndarray]],
    tier_ids: dict[int, list[str]],
    languages: list[str],
    n_layers: int,
) -> dict:
    """Compare convergence curves across complexity tiers.

    Args:
        tier_ids: {1: [op_ids...], 2: [...], 3: [...]}

    Returns: {tier: {"R": array(n_layers+1)}}
    """
    results = {}
    for tier, op_ids in tier_ids.items():
        tier_states = {op: nl_states[op] for op in op_ids if op in nl_states}
        results[tier] = layer_convergence_curve(tier_states, languages, n_layers)
    return results
