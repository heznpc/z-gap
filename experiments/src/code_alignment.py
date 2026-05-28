"""NL-Code cross-modal alignment experiment.

Tests whether NL descriptions and their code equivalents converge in Z,
directly testing PRH for code as a modality.
"""

import numpy as np
from scipy.spatial.distance import cosine

# 50 computational operations with Python code equivalents
CODE_EQUIVALENTS = {
    "comp_01_sort_asc": "sorted(lst)",
    "comp_02_find_max": "max(lst)",
    "comp_03_filter_pos": "[x for x in lst if x > 0]",
    "comp_04_reverse": "lst[::-1]",
    "comp_05_count": "len(lst)",
    "comp_06_sum": "sum(lst)",
    "comp_07_deduplicate": "list(set(lst))",
    "comp_08_top3": "sorted(lst, reverse=True)[:3]",
    "comp_09_mean": "sum(lst) / len(lst)",
    "comp_10_sort_desc": "sorted(lst, reverse=True)",
    "comp_11_concat": "s1 + s2",
    "comp_12_uppercase": "s.upper()",
    "comp_13_split": "s.split()",
    "comp_14_replace": "s.replace('a', 'b')",
    "comp_15_length": "len(s)",
    "comp_16_abs": "abs(n)",
    "comp_17_power": "n ** 3",
    "comp_18_modulo": "n % 7",
    "comp_19_sqrt": "n ** 0.5",
    "comp_20_gcd": "math.gcd(a, b)",
    "comp_21_union": "s1 | s2",
    "comp_22_intersect": "s1 & s2",
    "comp_23_difference": "s1 - s2",
    "comp_24_keys": "list(d.keys())",
    "comp_25_merge": "{**d1, **d2}",
    "comp_26_transpose": "list(zip(*matrix))",
    "comp_27_flatten": "[x for sub in lst for x in sub]",
    "comp_28_zip": "list(zip(lst1, lst2))",
    "comp_29_index": "lst.index(target)",
    "comp_30_contains": "target in lst",
    "comp_31_all_pos": "all(x > 0 for x in lst)",
    "comp_32_any_neg": "any(x < 0 for x in lst)",
    "comp_33_int_to_str": "str(n)",
    "comp_34_round": "round(n, 2)",
    "comp_35_min": "min(lst)",
    "comp_36_median": "sorted(lst)[len(lst)//2]",
    "comp_37_slice": "lst[2:5]",
    "comp_38_even": "[x for x in lst if x % 2 == 0]",
    "comp_39_freq": "collections.Counter(lst)",
    "comp_40_map_double": "[x * 2 for x in lst]",
    "comp_41_depth": "def depth(t): return 1 + max(depth(c) for c in t.children) if t.children else 0",
    "comp_42_is_palindrome": "s == s[::-1]",
    "comp_43_binary": "bin(n)",
    "comp_44_cumsum": "[sum(lst[:i+1]) for i in range(len(lst))]",
    "comp_45_prod": "functools.reduce(lambda a, b: a * b, lst)",
    "comp_46_range": "list(range(1, 11))",
    "comp_47_char_count": "len(s)",
    "comp_48_prime": "all(n % i != 0 for i in range(2, int(n**0.5)+1)) and n > 1",
    "comp_49_sort_by_len": "sorted(lst, key=len)",
    "comp_50_matrix_mult": "[[sum(a*b for a,b in zip(row,col)) for col in zip(*B)] for row in A]",
}


def compute_nl_code_alignment(
    nl_embeddings: dict[str, np.ndarray],
    code_embeddings: dict[str, np.ndarray],
    comp_ids: list[str],
    languages: list[str],
) -> dict:
    """Compute NL-code cross-modal alignment.

    For each operation, measure cosine similarity between NL description
    (in each language) and its code equivalent. Compare:
    - d_match: distance between NL and its corresponding code
    - d_mismatch: distance between NL and a different operation's code

    If PRH holds for code: d_match << d_mismatch (R_code > 1)
    """
    d_match_list = []
    d_mismatch_list = []

    for op_id in comp_ids:
        if op_id not in code_embeddings:
            continue
        code_vec = code_embeddings[op_id]

        # d_match: NL descriptions of THIS operation → code of THIS operation
        for lang in languages:
            nl_key = f"{op_id}_{lang}"
            if nl_key in nl_embeddings:
                d_match_list.append(float(cosine(nl_embeddings[nl_key], code_vec)))

        # d_mismatch: NL descriptions of THIS operation → code of OTHER operations
        other_ids = [oid for oid in comp_ids if oid != op_id and oid in code_embeddings]
        for other_id in other_ids[:10]:  # sample 10 for efficiency
            for lang in languages[:2]:  # en + one other
                nl_key = f"{op_id}_{lang}"
                if nl_key in nl_embeddings:
                    d_mismatch_list.append(float(cosine(nl_embeddings[nl_key], code_embeddings[other_id])))

    mean_d_match = float(np.mean(d_match_list)) if d_match_list else 0.0
    mean_d_mismatch = float(np.mean(d_mismatch_list)) if d_mismatch_list else 0.0
    R_code = mean_d_mismatch / mean_d_match if mean_d_match > 1e-10 else float("inf")

    # Per-language d_match
    per_lang_d_match = {}
    for lang in languages:
        dists = []
        for op_id in comp_ids:
            nl_key = f"{op_id}_{lang}"
            if nl_key in nl_embeddings and op_id in code_embeddings:
                dists.append(float(cosine(nl_embeddings[nl_key], code_embeddings[op_id])))
        per_lang_d_match[lang] = float(np.mean(dists)) if dists else 0.0

    return {
        "R_code": R_code,
        "mean_d_match": mean_d_match,
        "mean_d_mismatch": mean_d_mismatch,
        "n_match_pairs": len(d_match_list),
        "n_mismatch_pairs": len(d_mismatch_list),
        "per_lang_d_match": per_lang_d_match,
        "d_match_std": float(np.std(d_match_list)) if d_match_list else 0.0,
    }


def compute_per_language_R_code(
    nl_embeddings: dict[str, np.ndarray],
    code_embeddings: dict[str, np.ndarray],
    comp_ids: list[str],
    languages: list[str],
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Per-language R_code with permutation test and bootstrap CI.

    For each language, compute R_code = mean(d_mismatch) / mean(d_match)
    using ALL mismatched operations (not sampled).

    Returns: {lang: {R_code, p_value, ci_95, cohens_d, ...}}
    """
    rng = np.random.default_rng(seed)
    valid_ids = [oid for oid in comp_ids if oid in code_embeddings]

    results = {}
    for lang in languages:
        d_match = []
        d_mismatch = []

        for op_id in valid_ids:
            nl_key = f"{op_id}_{lang}"
            if nl_key not in nl_embeddings:
                continue
            nl_vec = nl_embeddings[nl_key]
            code_vec = code_embeddings[op_id]

            # d_match: this NL → this code
            d_match.append(float(cosine(nl_vec, code_vec)))

            # d_mismatch: this NL → ALL other codes
            for other_id in valid_ids:
                if other_id == op_id:
                    continue
                d_mismatch.append(float(cosine(nl_vec, code_embeddings[other_id])))

        if not d_match:
            results[lang] = {"skip": True}
            continue

        d_match_arr = np.array(d_match)
        d_mismatch_arr = np.array(d_mismatch)
        observed_R = float(np.mean(d_mismatch_arr) / np.mean(d_match_arr))

        # Permutation test: shuffle which code each NL is "matched" to.
        # V5 (review-2026-05-21): substitute NaN (not 1.0) when a permutation
        # produces an empty d_match_perm, then drop NaNs before computing
        # the p-value so the null distribution is not biased toward 1.0.
        perm_Rs = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = rng.permutation(valid_ids)
            d_match_perm = []
            for j, op_id in enumerate(valid_ids):
                nl_key = f"{op_id}_{lang}"
                if nl_key in nl_embeddings and shuffled[j] in code_embeddings:
                    d_match_perm.append(float(cosine(
                        nl_embeddings[nl_key], code_embeddings[shuffled[j]]
                    )))
            if d_match_perm:
                perm_Rs[i] = np.mean(d_mismatch_arr) / np.mean(d_match_perm)
            else:
                perm_Rs[i] = np.nan
        valid_perm = perm_Rs[~np.isnan(perm_Rs)]
        n_extreme = int(np.sum(valid_perm >= observed_R))
        # V6 (review-2026-05-21): use the (k+1)/(n+1) convention so the
        # reported p_value is bounded below by 1/(n_valid+1) and is never
        # literal 0.0 — that lower bound is what reviewers expect from a
        # permutation test with n_perm=10,000.
        n_valid = int(len(valid_perm))
        p_value = float((n_extreme + 1) / (n_valid + 1)) if n_valid > 0 else float("nan")

        # Bootstrap CI for R_code.
        # V5: NaN fallback for degenerate mean_m so the bootstrap CI is not
        # silently pulled toward 1.0.
        boot_Rs = np.empty(n_boot)
        for i in range(n_boot):
            idx_m = rng.integers(0, len(d_match_arr), size=len(d_match_arr))
            idx_mm = rng.integers(0, len(d_mismatch_arr), size=len(d_mismatch_arr))
            mean_m = np.mean(d_match_arr[idx_m])
            boot_Rs[i] = np.mean(d_mismatch_arr[idx_mm]) / mean_m if mean_m > 1e-10 else np.nan
        valid_boot = boot_Rs[~np.isnan(boot_Rs)]
        if len(valid_boot) > 0:
            ci_lo = float(np.percentile(valid_boot, 2.5))
            ci_hi = float(np.percentile(valid_boot, 97.5))
        else:
            ci_lo = float("nan")
            ci_hi = float("nan")

        # Cohen's d
        s_pooled = np.sqrt(
            ((len(d_match_arr) - 1) * np.var(d_match_arr, ddof=1) +
             (len(d_mismatch_arr) - 1) * np.var(d_mismatch_arr, ddof=1)) /
            (len(d_match_arr) + len(d_mismatch_arr) - 2)
        )
        cohens_d = float((np.mean(d_mismatch_arr) - np.mean(d_match_arr)) / s_pooled) if s_pooled > 1e-10 else 0.0

        results[lang] = {
            "skip": False,
            "R_code": observed_R,
            "p_value": p_value,
            "ci_95": (ci_lo, ci_hi),
            "cohens_d": cohens_d,
            "d_match_mean": float(np.mean(d_match_arr)),
            "d_mismatch_mean": float(np.mean(d_mismatch_arr)),
            "n_match": len(d_match_arr),
            "n_mismatch": len(d_mismatch_arr),
            # C2 (review-2026-05-21): random-matching baseline, sourced from the
            # permutation null distribution. Expected ≈ 1.0 if shuffled NL→code
            # pairings produce the same mean(d_mismatch)/mean(d_match) ratio as
            # matched pairings. Used in paper §5.5 to anchor R_code = 1 as the
            # null line rather than as an asserted-but-unmeasured baseline.
            # V5: NaN-safe aggregation across the valid permutations.
            "random_baseline_R_mean": float(np.nanmean(perm_Rs)) if n_valid > 0 else float("nan"),
            "random_baseline_R_std": float(np.nanstd(perm_Rs)) if n_valid > 0 else float("nan"),
            "random_baseline_R_p95": float(np.nanpercentile(perm_Rs, 95)) if n_valid > 0 else float("nan"),
            "n_perm_valid": n_valid,
            "n_boot_valid": int(len(valid_boot)),
        }

    # Aggregate (all languages pooled)
    all_d_match, all_d_mismatch = [], []
    for lang in languages:
        for op_id in valid_ids:
            nl_key = f"{op_id}_{lang}"
            if nl_key not in nl_embeddings:
                continue
            all_d_match.append(float(cosine(nl_embeddings[nl_key], code_embeddings[op_id])))
            for other_id in valid_ids:
                if other_id != op_id:
                    all_d_mismatch.append(float(cosine(nl_embeddings[nl_key], code_embeddings[other_id])))

    if all_d_match:
        agg_R = float(np.mean(all_d_mismatch) / np.mean(all_d_match))
        results["aggregate"] = {"R_code": agg_R, "n_match": len(all_d_match)}
    else:
        results["aggregate"] = {"R_code": 0.0, "n_match": 0}

    return results
