"""Core metrics: d_intra, d_inter, discriminability ratio R, spacing robustness."""

import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations


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
        indices = rng.choice(len(vecs), size=(sample_size, 2), replace=True)
        indices = indices[indices[:, 0] != indices[:, 1]]
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
    indices = rng.choice(len(vecs), size=(min(500, len(vecs) * 2), 2), replace=True)
    indices = indices[indices[:, 0] != indices[:, 1]]
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
