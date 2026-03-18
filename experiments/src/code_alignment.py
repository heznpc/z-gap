"""NL-Code cross-modal alignment experiment.

Tests whether NL descriptions and their code equivalents converge in Z,
directly testing PRH for code as a modality.
"""

import numpy as np
from itertools import combinations
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
