"""Core metrics: d_intra, d_inter, discriminability ratio R, spacing robustness, dialectal continuum.

Topology metrics (Strategy 4 / Aristotelian k-NN reanalysis):
  - k-NN cross-lingual accuracy
  - Mean Reciprocal Rank (MRR)
  - Neighborhood overlap (Jaccard)
  - Hubness analysis
"""

import numpy as np
from scipy.spatial.distance import cosine, cdist
from itertools import combinations
from collections import Counter


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine(a, b))


def compute_d_intra(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
) -> dict[str, float]:
    """For each operation, compute mean pairwise cosine distance across languages.

    embeddings: {f"{op_id}_{lang}": vector}
    Returns: {op_id: d_intra}
    """
    result = {}
    for op_id in operation_ids:
        vecs = []
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                vecs.append(embeddings[key])
        if len(vecs) < 2:
            continue
        dists = [cosine_distance(a, b) for a, b in combinations(vecs, 2)]
        result[op_id] = float(np.mean(dists))
    return result


def compute_d_inter(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    lang: str,
    sample_size: int = 500,
) -> float:
    """For a language, compute mean pairwise cosine distance across operations.

    Uses random sampling if too many pairs.
    """
    vecs = []
    for op_id in operation_ids:
        key = f"{op_id}_{lang}"
        if key in embeddings:
            vecs.append(embeddings[key])
    if len(vecs) < 2:
        return 0.0

    n_pairs = len(vecs) * (len(vecs) - 1) // 2
    if n_pairs <= sample_size:
        dists = [cosine_distance(a, b) for a, b in combinations(vecs, 2)]
    else:
        rng = np.random.default_rng(42)
        indices = np.empty((0, 2), dtype=int)
        while len(indices) < sample_size:
            new = rng.choice(len(vecs), size=(sample_size - len(indices) + 50, 2), replace=True)
            new = new[new[:, 0] != new[:, 1]]
            indices = np.vstack([indices, new]) if len(indices) > 0 else new
        indices = indices[:sample_size]
        dists = [cosine_distance(vecs[i], vecs[j]) for i, j in indices]

    return float(np.mean(dists))


def discriminability_ratio(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
) -> dict:
    """Compute R = mean(d_inter) / mean(d_intra).

    R > 1 means cross-lingual same-operation similarity > within-language
    different-operation similarity.
    """
    d_intra = compute_d_intra(embeddings, operation_ids, languages)
    mean_d_intra = float(np.mean(list(d_intra.values()))) if d_intra else 0.0

    d_inter_per_lang = {}
    for lang in languages:
        d_inter_per_lang[lang] = compute_d_inter(embeddings, operation_ids, lang)
    mean_d_inter = float(np.mean(list(d_inter_per_lang.values())))

    R = mean_d_inter / mean_d_intra if mean_d_intra > 1e-10 else float("inf")

    return {
        "R": R,
        "mean_d_intra": mean_d_intra,
        "mean_d_inter": mean_d_inter,
        "d_intra_per_op": d_intra,
        "d_inter_per_lang": d_inter_per_lang,
    }


def compute_per_operation_detail(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    categories: dict[str, str],
) -> list[dict]:
    """Per-operation breakdown: d_intra, per-language-pair distances, mean distance to others.

    Returns list of dicts with:
      op_id, category, d_intra, lang_pair_distances, mean_dist_to_others
    """
    lang_pairs = list(combinations(languages, 2))
    results = []

    # Pre-collect per-language vectors for d_inter contribution
    lang_op_vecs: dict[str, list[tuple[str, np.ndarray]]] = {l: [] for l in languages}
    for op_id in operation_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                lang_op_vecs[lang].append((op_id, embeddings[key]))

    for op_id in operation_ids:
        # d_intra: mean pairwise distance across languages for this operation
        vecs_by_lang = {}
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                vecs_by_lang[lang] = embeddings[key]

        # Per-language-pair distances
        pair_dists = {}
        for l1, l2 in lang_pairs:
            if l1 in vecs_by_lang and l2 in vecs_by_lang:
                pair_dists[f"{l1}-{l2}"] = cosine_distance(vecs_by_lang[l1], vecs_by_lang[l2])

        d_intra = float(np.mean(list(pair_dists.values()))) if pair_dists else 0.0

        # Mean distance to other operations (per language)
        mean_dist_to_others = {}
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key not in embeddings:
                continue
            vec = embeddings[key]
            dists = [cosine_distance(vec, v) for oid, v in lang_op_vecs[lang] if oid != op_id]
            mean_dist_to_others[lang] = float(np.mean(dists)) if dists else 0.0

        results.append({
            "op_id": op_id,
            "category": categories.get(op_id, "unknown"),
            "d_intra": d_intra,
            "lang_pair_distances": pair_dists,
            "mean_dist_to_others": mean_dist_to_others,
            "mean_d_inter": float(np.mean(list(mean_dist_to_others.values()))) if mean_dist_to_others else 0.0,
        })

    return results


def spacing_robustness(
    embeddings_correct: dict[str, np.ndarray],
    embeddings_variants: dict[str, dict[str, np.ndarray]],
    operation_ids: list[str],
) -> dict:
    """Compute R_spacing = d_semantic / d_spacing.

    embeddings_correct: {op_id: vector} (correct spacing)
    embeddings_variants: {op_id: {variant_name: vector}}
    """
    d_spacing_list = []
    for op_id in operation_ids:
        if op_id not in embeddings_correct or op_id not in embeddings_variants:
            continue
        correct = embeddings_correct[op_id]
        for variant_name, variant_vec in embeddings_variants[op_id].items():
            if variant_name == "correct":
                continue
            d_spacing_list.append(cosine_distance(correct, variant_vec))

    # d_semantic: distances between different operations (correct spacing)
    vecs = list(embeddings_correct.values())
    if len(vecs) < 2:
        return {"R_spacing": 0.0}
    rng = np.random.default_rng(42)
    target = min(500, len(vecs) * 2)
    indices = np.empty((0, 2), dtype=int)
    while len(indices) < target:
        new = rng.choice(len(vecs), size=(target - len(indices) + 50, 2), replace=True)
        new = new[new[:, 0] != new[:, 1]]
        indices = np.vstack([indices, new]) if len(indices) > 0 else new
    indices = indices[:target]
    d_semantic_list = [cosine_distance(vecs[i], vecs[j]) for i, j in indices]

    mean_d_spacing = float(np.mean(d_spacing_list)) if d_spacing_list else 0.0
    mean_d_semantic = float(np.mean(d_semantic_list)) if d_semantic_list else 0.0
    R_spacing = mean_d_semantic / mean_d_spacing if mean_d_spacing > 1e-10 else float("inf")

    return {
        "R_spacing": R_spacing,
        "mean_d_spacing": mean_d_spacing,
        "mean_d_semantic": mean_d_semantic,
        "n_spacing_pairs": len(d_spacing_list),
        "n_semantic_pairs": len(d_semantic_list),
    }


def dialectal_continuum(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    dialects: dict[str, list[str]],
) -> dict:
    """Compute R at three levels: within-dialect, cross-dialect, cross-lingual.

    Embedding keys: "{op_id}_{lang}_{dialect}" for dialect variants,
                    "{op_id}_{lang}" for standard (backward-compatible).

    Returns: {level: R, d_intra, d_inter} for each level.
    """
    def _mean_dist(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        if not pairs:
            return 0.0
        return float(np.mean([cosine_distance(a, b) for a, b in pairs]))

    # 1. Cross-lingual d_intra: same op, different languages (standard dialects)
    cross_lingual_pairs = []
    for op_id in operation_ids:
        vecs = []
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                vecs.append(embeddings[key])
        cross_lingual_pairs.extend(combinations(vecs, 2))

    # 2. Cross-dialect d_intra: same op, same language, different dialects
    cross_dialect_pairs = []
    for op_id in operation_ids:
        for lang in languages:
            if lang not in dialects:
                continue
            vecs = []
            for dialect in dialects[lang]:
                key = f"{op_id}_{lang}_{dialect}"
                if key in embeddings:
                    vecs.append(embeddings[key])
            cross_dialect_pairs.extend(combinations(vecs, 2))

    # 3. Within-dialect d_intra: same op, same dialect — standard vs paraphrase
    #    or distance between the standard description embedding and itself
    #    (placeholder: will be populated by paraphrase data in Strategy 6-R)
    within_dialect_pairs = []
    # For now, within-dialect pairs are same-op same-language standard embeddings
    # When paraphrase data is available, this measures d_paraphrase

    d_cross_lingual = _mean_dist(cross_lingual_pairs)
    d_cross_dialect = _mean_dist(cross_dialect_pairs)
    d_within_dialect = _mean_dist(within_dialect_pairs)

    # Continuum prediction: d_within_dialect < d_cross_dialect < d_cross_lingual
    # Note: R metrics are kept for backward compatibility but the continuum
    # is now tested via the ordering of raw distances
    R_cross_lingual = d_within_dialect / d_cross_lingual if d_cross_lingual > 1e-10 else float("inf")
    R_cross_dialect = d_within_dialect / d_cross_dialect if d_cross_dialect > 1e-10 else float("inf")

    continuum_holds = (
        d_cross_dialect < d_cross_lingual
        if d_cross_dialect > 0 and d_cross_lingual > 0
        else False
    )

    return {
        "d_cross_lingual": d_cross_lingual,
        "d_cross_dialect": d_cross_dialect,
        "d_within_dialect": d_within_dialect,
        "R_cross_lingual": R_cross_lingual,
        "R_cross_dialect": R_cross_dialect,
        "n_cross_lingual_pairs": len(cross_lingual_pairs),
        "n_cross_dialect_pairs": len(cross_dialect_pairs),
        "n_within_dialect_pairs": len(within_dialect_pairs),
        "continuum_holds": continuum_holds,
    }


# ============================================================
# Strategy 4: Aristotelian k-NN topology metrics
# ============================================================
#
# Motivation (Gröger et al., 2026): global distance metrics like CKA and
# discriminability ratio R can be confounded by linear transforms and
# scaling.  LOCAL neighborhood topology — which points are nearest to
# which — is a stricter, more reliable test of representational
# convergence.  If P2 fails at the R level but succeeds at the k-NN
# level, the failure is a metric artifact, not a genuine divergence.
# ============================================================


def _build_distance_matrix(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build a full pairwise cosine distance matrix for all (op, lang) pairs.

    Returns:
        dist_matrix: (N, N) cosine distance matrix
        keys: list of "{op_id}_{lang}" keys in row/col order
        op_ids: list of operation ids corresponding to each row
        langs: list of language codes corresponding to each row
    """
    keys, vecs, op_list, lang_list = [], [], [], []
    for op_id in operation_ids:
        for lang in languages:
            key = f"{op_id}_{lang}"
            if key in embeddings:
                keys.append(key)
                vecs.append(embeddings[key])
                op_list.append(op_id)
                lang_list.append(lang)

    if len(vecs) == 0:
        return np.empty((0, 0)), [], [], []

    X = np.array(vecs)  # (N, d)
    dist_matrix = cdist(X, X, metric="cosine")  # (N, N)
    # Zero out diagonal (cdist can leave small float artifacts)
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix, keys, op_list, lang_list


def knn_cross_lingual_accuracy(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    k_values: list[int] = None,
) -> dict:
    """Compute k-NN cross-lingual retrieval accuracy.

    For each query point (op_id, lang_q), find its k nearest neighbors
    among ALL other points.  A "hit" occurs when at least one of the k
    neighbors is the SAME operation in a DIFFERENT language.

    This metric tests LOCAL topology: is the correct cross-lingual
    equivalent nearby, regardless of absolute distance scale?

    Returns dict with per-k accuracy, per-operation results, and
    Recall@k (fraction of cross-lingual matches found in top k).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    dist_matrix, keys, op_list, lang_list = _build_distance_matrix(
        embeddings, operation_ids, languages
    )
    N = len(keys)
    if N == 0:
        return {"k_values": k_values, "accuracy": {}, "recall": {}}

    max_k = max(k_values)

    # For each query, get sorted neighbor indices (excluding self)
    # argsort row-wise; col 0 is self (distance 0), so skip it
    sorted_indices = np.argsort(dist_matrix, axis=1)  # (N, N)

    # Build per-query results
    per_query = []  # list of dicts
    for i in range(N):
        query_op = op_list[i]
        query_lang = lang_list[i]

        # Neighbors (skip self at position 0)
        neighbors = sorted_indices[i, 1:]  # (N-1,)

        # Target set: indices of same operation, different language
        targets = set()
        for j in range(N):
            if op_list[j] == query_op and lang_list[j] != query_lang:
                targets.add(j)

        n_targets = len(targets)  # should be len(languages) - 1

        # For each k, check hit and compute recall
        per_k_hit = {}
        per_k_recall = {}
        per_k_reciprocal_rank = None

        # Reciprocal rank: rank of first correct match
        first_correct_rank = None
        for rank_idx, nb in enumerate(neighbors[:max(max_k, N - 1)]):
            if nb in targets:
                if first_correct_rank is None:
                    first_correct_rank = rank_idx + 1  # 1-based rank
                    break

        for k in k_values:
            top_k_set = set(neighbors[:k].tolist())
            hits_in_k = len(top_k_set & targets)
            per_k_hit[k] = hits_in_k > 0  # at-least-one hit
            per_k_recall[k] = hits_in_k / n_targets if n_targets > 0 else 0.0

        per_query.append({
            "key": keys[i],
            "op_id": query_op,
            "lang": query_lang,
            "per_k_hit": per_k_hit,
            "per_k_recall": per_k_recall,
            "reciprocal_rank": 1.0 / first_correct_rank if first_correct_rank else 0.0,
            "first_correct_rank": first_correct_rank,
        })

    # Aggregate: accuracy@k = fraction of queries with a hit
    accuracy = {}
    recall = {}
    mrr_all = float(np.mean([q["reciprocal_rank"] for q in per_query]))
    for k in k_values:
        accuracy[k] = float(np.mean([q["per_k_hit"][k] for q in per_query]))
        recall[k] = float(np.mean([q["per_k_recall"][k] for q in per_query]))

    return {
        "k_values": k_values,
        "accuracy": accuracy,       # Accuracy@k (at-least-one hit)
        "recall": recall,            # Recall@k (fraction of targets found)
        "mrr": mrr_all,              # Mean Reciprocal Rank
        "n_queries": N,
        "per_query": per_query,
    }


def knn_accuracy_by_category(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    categories: dict[str, str],
    k_values: list[int] = None,
) -> dict:
    """Compute k-NN cross-lingual accuracy separately per category.

    This is the core P2-kNN test: compare kNN_C vs kNN_J.

    Returns per-category accuracy, recall, MRR, and per-operation detail.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    full = knn_cross_lingual_accuracy(embeddings, operation_ids, languages, k_values)

    # Split per_query by category
    by_cat: dict[str, list] = {}
    for q in full["per_query"]:
        cat = categories.get(q["op_id"], "unknown")
        by_cat.setdefault(cat, []).append(q)

    result_by_cat = {}
    for cat, queries in by_cat.items():
        acc = {}
        rec = {}
        for k in k_values:
            acc[k] = float(np.mean([q["per_k_hit"][k] for q in queries]))
            rec[k] = float(np.mean([q["per_k_recall"][k] for q in queries]))
        mrr = float(np.mean([q["reciprocal_rank"] for q in queries]))

        # Per-operation aggregation (average across languages for each op)
        op_acc = {}
        for q in queries:
            op_acc.setdefault(q["op_id"], []).append(q)
        per_op = {}
        for op_id, op_queries in op_acc.items():
            per_op[op_id] = {
                "accuracy": {
                    k: float(np.mean([q["per_k_hit"][k] for q in op_queries]))
                    for k in k_values
                },
                "mrr": float(np.mean([q["reciprocal_rank"] for q in op_queries])),
                "mean_first_rank": float(np.mean([
                    q["first_correct_rank"] for q in op_queries
                    if q["first_correct_rank"] is not None
                ])) if any(q["first_correct_rank"] is not None for q in op_queries) else float("inf"),
            }

        result_by_cat[cat] = {
            "accuracy": acc,
            "recall": rec,
            "mrr": mrr,
            "n_queries": len(queries),
            "per_operation": per_op,
        }

    return {
        "k_values": k_values,
        "overall": {
            "accuracy": full["accuracy"],
            "recall": full["recall"],
            "mrr": full["mrr"],
        },
        "by_category": result_by_cat,
    }


def neighborhood_overlap(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    k: int = 10,
) -> dict:
    """Compute neighborhood overlap (Jaccard) for same-operation across languages.

    For each pair of languages (L1, L2) and each operation, compute:
        J = |N_k(op, L1) ∩ N_k(op, L2)| / |N_k(op, L1) ∪ N_k(op, L2)|

    where N_k(op, L) is the set of operation-ids among the k nearest
    neighbors of (op, L).  High Jaccard means the two language
    representations of the same operation "see" the same neighborhood,
    indicating topological alignment.

    Returns per-operation Jaccard and category aggregates.
    """
    dist_matrix, keys, op_list, lang_list = _build_distance_matrix(
        embeddings, operation_ids, languages
    )
    N = len(keys)
    if N == 0:
        return {}

    sorted_indices = np.argsort(dist_matrix, axis=1)

    # Build index: (op_id, lang) -> row index
    idx_map = {}
    for i in range(N):
        idx_map[(op_list[i], lang_list[i])] = i

    # For each point, its k-nearest neighbor operation-ids
    def _neighbor_ops(row_idx: int) -> set[str]:
        """Return the set of operation-ids among the k nearest neighbors."""
        neighbors = sorted_indices[row_idx, 1:k + 1]
        return {op_list[j] for j in neighbors}

    lang_pairs = list(combinations(languages, 2))
    per_op_jaccard = {}

    for op_id in operation_ids:
        pair_jaccards = []
        for l1, l2 in lang_pairs:
            if (op_id, l1) not in idx_map or (op_id, l2) not in idx_map:
                continue
            n1 = _neighbor_ops(idx_map[(op_id, l1)])
            n2 = _neighbor_ops(idx_map[(op_id, l2)])
            union = n1 | n2
            if len(union) == 0:
                continue
            jaccard = len(n1 & n2) / len(union)
            pair_jaccards.append(jaccard)
        if pair_jaccards:
            per_op_jaccard[op_id] = float(np.mean(pair_jaccards))

    return {
        "k": k,
        "per_operation": per_op_jaccard,
        "mean_jaccard": float(np.mean(list(per_op_jaccard.values()))) if per_op_jaccard else 0.0,
    }


def hubness_analysis(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    k: int = 5,
) -> dict:
    """Analyze hubness: how often each point appears as a k-nearest neighbor.

    In high-dimensional spaces, some points become "hubs" that are
    nearest neighbors to many others, distorting distance-based metrics
    while leaving neighborhood topology relatively unaffected.

    Returns:
        hub_counts: {key: count} — how many times each point is a k-NN of another
        skewness: skewness of the hub count distribution (>0 indicates hubness)
        top_hubs: the most frequent hubs
    """
    dist_matrix, keys, op_list, lang_list = _build_distance_matrix(
        embeddings, operation_ids, languages
    )
    N = len(keys)
    if N == 0:
        return {}

    sorted_indices = np.argsort(dist_matrix, axis=1)

    # Count how many times each point j appears in the top-k of any other point i
    hub_counter = Counter()
    for i in range(N):
        for j in sorted_indices[i, 1:k + 1]:
            hub_counter[keys[j]] += 1

    counts = np.array([hub_counter.get(k, 0) for k in keys], dtype=float)
    mean_count = float(np.mean(counts))
    std_count = float(np.std(counts))

    # Skewness of N_k distribution (Robin & Bhattacharya hubness indicator)
    if std_count > 1e-10:
        skewness = float(np.mean(((counts - mean_count) / std_count) ** 3))
    else:
        skewness = 0.0

    # Top hubs
    top_hubs = hub_counter.most_common(10)

    # Per-category mean hub count
    cat_counts: dict[str, list] = {}
    for i in range(N):
        cat_counts.setdefault(op_list[i], [])  # use op_id as proxy; caller categorizes
    # Actually aggregate by key
    per_key_count = {keys[i]: counts[i] for i in range(N)}

    return {
        "k": k,
        "mean_hub_count": mean_count,
        "std_hub_count": std_count,
        "skewness": skewness,
        "hubness_detected": skewness > 1.0,  # conventional threshold
        "top_hubs": [{"key": h, "count": c} for h, c in top_hubs],
        "per_key_count": per_key_count,
    }


def compute_topology_suite(
    embeddings: dict[str, np.ndarray],
    operation_ids: list[str],
    languages: list[str],
    categories: dict[str, str],
    k_values: list[int] = None,
) -> dict:
    """Run the full Strategy 4 topology analysis suite.

    Combines k-NN accuracy (per category), neighborhood overlap,
    hubness analysis, and random baseline into a single result dict.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    n_ops = len(operation_ids)
    n_langs = len(languages)
    N = n_ops * n_langs  # total points in the pool

    # --- k-NN accuracy by category (the core P2-kNN test) ---
    knn_result = knn_accuracy_by_category(
        embeddings, operation_ids, languages, categories, k_values
    )

    # --- Neighborhood overlap (Jaccard) by category ---
    overlap_full = neighborhood_overlap(embeddings, operation_ids, languages, k=10)
    overlap_by_cat = {}
    for cat_name in ["computational", "judgment"]:
        cat_ops = [oid for oid in operation_ids if categories.get(oid) == cat_name]
        vals = [overlap_full["per_operation"][oid] for oid in cat_ops
                if oid in overlap_full["per_operation"]]
        overlap_by_cat[cat_name] = float(np.mean(vals)) if vals else 0.0

    # --- Hubness analysis ---
    hubness = hubness_analysis(embeddings, operation_ids, languages, k=5)

    # --- Random baseline for k-NN accuracy ---
    # Each query has N-1 candidate neighbors.  The number of correct
    # targets is (n_langs - 1) (same op, different lang).
    # P(at-least-one hit in k draws without replacement from N-1 items
    #   containing n_langs-1 targets):
    # 1 - C(N-1-(n_langs-1), k) / C(N-1, k)
    from math import comb
    baseline = {}
    n_targets = n_langs - 1
    pool = N - 1
    for k in k_values:
        if k >= pool:
            baseline[k] = 1.0
        else:
            # Hypergeometric: P(X >= 1) = 1 - P(X = 0)
            p_zero = comb(pool - n_targets, k) / comb(pool, k) if pool >= k else 0.0
            baseline[k] = 1.0 - p_zero

    # --- Assemble summary ---
    summary = {
        "p2_knn_supported": {},  # per-k verdict
        "delta": {},             # kNN_C - kNN_J per k
    }
    cat_c = knn_result["by_category"].get("computational", {})
    cat_j = knn_result["by_category"].get("judgment", {})
    for k in k_values:
        acc_c = cat_c.get("accuracy", {}).get(k, 0.0)
        acc_j = cat_j.get("accuracy", {}).get(k, 0.0)
        summary["delta"][k] = acc_c - acc_j
        summary["p2_knn_supported"][k] = acc_c > acc_j

    return {
        "knn": knn_result,
        "neighborhood_overlap": {
            "full": overlap_full,
            "by_category": overlap_by_cat,
        },
        "hubness": hubness,
        "random_baseline": baseline,
        "summary": summary,
    }
