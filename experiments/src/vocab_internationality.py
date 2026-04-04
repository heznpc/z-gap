"""Strategy 1 experiment: Vocabulary internationality explains P2 failure.

Hypothesis: Judgment ops use abstract, cross-linguistically shared vocabulary
("evaluate", "prioritize") that gets high cross-lingual similarity because
these Latinate/international terms are similar across languages --- especially
in retrieval-tuned models that use English as a pivot (CLSD, arXiv:2502.08638).
Computational ops use domain-specific terms ("transpose", "palindrome") whose
translations vary more across languages, yielding higher d_intra (lower R_C).

If per-operation vocabulary internationality score correlates strongly with
per-operation d_intra, then R_C < R_J is a measurement artifact of NL
description vocabulary, not a property of Z_sem.

Three sub-metrics:
  (1) Token overlap ratio       --- character/subword overlap across languages
  (2) Romanization similarity   --- how similar non-Latin descriptions look
                                    when romanized (captures cognates/loans)
  (3) English-pivot cosine      --- per-token similarity to English equivalents
                                    in the embedding model (CLSD connection)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine


# ── (1) Token Overlap Ratio ──────────────────────────────────────────────

def _extract_tokens(text: str) -> set[str]:
    """Extract lowercased word tokens, stripping punctuation."""
    return set(re.findall(r"[\w]+", text.lower()))


def _normalize_to_ascii(text: str) -> str:
    """Strip accents: 'Evalúa' → 'Evalua', 'número' → 'numero'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _to_latin_tokens(text: str) -> set[str]:
    """Extract Latin-script tokens after stripping accents.

    'Evalúa la calidad' → {'evalua', 'la', 'calidad'}
    This captures cognates across en/es that differ only by accents.
    """
    ascii_text = _normalize_to_ascii(text.lower())
    return set(re.findall(r"[a-z]{2,}", ascii_text))


def _stem_match(tokens_a: set[str], tokens_b: set[str], min_prefix: int = 4) -> int:
    """Count stem-level matches: tokens sharing a prefix of min_prefix chars.

    "evaluate"/"evalua" share "eval" (4-char prefix) → match.
    "ascending"/"ascendente" share "asce" (4-char prefix) → match.
    """
    matches = 0
    for a in tokens_a:
        for b in tokens_b:
            prefix_len = min(len(a), len(b), min_prefix)
            if prefix_len >= min_prefix and a[:prefix_len] == b[:prefix_len]:
                matches += 1
                break  # each token in a matches at most once
    return matches


def token_overlap_ratio(descriptions: dict[str, str]) -> float:
    """Cross-lingual vocabulary overlap via Latin-script token matching.

    Two levels of matching:
    - Exact Jaccard on Latin tokens (after accent stripping)
    - Stem-level match (4-char prefix) to catch cognates like
      "evaluate"/"evalua", "ascending"/"ascendente"

    Non-Latin script pairs (ko-zh, ko-ar, zh-ar) use character 2-gram
    overlap as a fallback.

    Returns: mean pairwise overlap score across all language pairs.
    """
    latin_tokens = {}
    for lang, desc in descriptions.items():
        latin_tokens[lang] = _to_latin_tokens(desc)

    jaccard_scores = []
    for (l1, t1), (l2, t2) in combinations(latin_tokens.items(), 2):
        union = t1 | t2
        if len(union) == 0:
            # Both non-Latin (e.g., ko-zh): use character n-gram overlap
            jaccard_scores.append(_char_ngram_overlap(
                descriptions[l1], descriptions[l2], n=2
            ))
        else:
            # Exact Jaccard
            exact = len(t1 & t2) / len(union)
            # Stem-level (4-char prefix) overlap ratio
            stem_matches = _stem_match(t1, t2)
            stem_ratio = stem_matches / max(len(t1), len(t2)) if max(len(t1), len(t2)) > 0 else 0
            # Take the max of exact and stem (stem captures cognate pairs)
            jaccard_scores.append(max(exact, stem_ratio))

    return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0


def _char_ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """Character n-gram Jaccard for non-Latin script pairs."""
    def ngrams(text):
        text = re.sub(r"\s+", "", text)
        return Counter(text[i:i+n] for i in range(len(text) - n + 1))

    na, nb = ngrams(text_a), ngrams(text_b)
    intersection = sum((na & nb).values())
    union = sum((na | nb).values())
    return intersection / union if union > 0 else 0.0


# ── (2) Romanization Similarity ──────────────────────────────────────────

def _naive_romanize(text: str) -> str:
    """Rough romanization via Unicode decomposition + transliteration.

    Not linguistically accurate, but sufficient to detect when
    loanwords/cognates produce similar Latin forms. For CJK and Arabic
    we strip to detect shared Latin loanwords embedded in text.
    """
    # Normalize to NFD, strip combining marks, keep base letters
    nfkd = unicodedata.normalize("NFKD", text)
    # Keep only ASCII letters and digits
    result = []
    for ch in nfkd:
        if ch.isascii() and ch.isalpha():
            result.append(ch.lower())
        elif ch == " ":
            result.append(" ")
    return "".join(result).strip()


def romanization_similarity(descriptions: dict[str, str]) -> float:
    """Mean pairwise character-level similarity of romanized descriptions.

    High score = descriptions contain similar Latin-origin terms even across
    scripts. Judgment words like "evaluate" often have recognizable Latin
    cognates in es ("evalua") and sometimes Arabic loanwords; computational
    terms like "palindrome" may also have cognates but domain-specific ones.

    Uses character 3-gram Jaccard on the romanized forms.
    """
    romanized = {lang: _naive_romanize(desc) for lang, desc in descriptions.items()}

    scores = []
    for (l1, r1), (l2, r2) in combinations(romanized.items(), 2):
        if len(r1) < 2 or len(r2) < 2:
            scores.append(0.0)
            continue
        scores.append(_char_ngram_overlap(r1, r2, n=3))

    return float(np.mean(scores)) if scores else 0.0


# ── (3) English-Pivot Cosine (CLSD connection) ──────────────────────────

def english_pivot_score(
    embeddings: dict[str, np.ndarray],
    op_id: str,
    languages: list[str],
) -> float:
    """Mean cosine similarity of each non-English description to the English one.

    High score = all translations land near the English embedding, consistent
    with the CLSD finding that retrieval-tuned models route cross-lingual
    matching through an English pivot.

    If the pivot effect is strong AND judgment ops cluster tighter around the
    English pivot, that explains R_J > R_C without invoking Z_sem properties.
    """
    en_key = f"{op_id}_en"
    if en_key not in embeddings:
        return 0.0

    en_vec = embeddings[en_key]
    sims = []
    for lang in languages:
        if lang == "en":
            continue
        key = f"{op_id}_{lang}"
        if key in embeddings:
            sims.append(1.0 - cosine(en_vec, embeddings[key]))

    return float(np.mean(sims)) if sims else 0.0


# ── Composite Internationality Score ────────────────────────────────────

def compute_internationality_scores(
    operations: list,  # list[Operation]
    embeddings: dict[str, np.ndarray],
    languages: list[str],
) -> list[dict]:
    """Compute all three sub-metrics for each operation.

    Returns list of dicts:
        {op_id, category, token_overlap, roman_sim, en_pivot,
         internationality (composite), d_intra}
    """
    from .metrics import compute_d_intra

    op_ids = [op.id for op in operations]
    d_intra_map = compute_d_intra(embeddings, op_ids, languages)

    results = []
    for op in operations:
        tok_ov = token_overlap_ratio(op.descriptions)
        rom_sim = romanization_similarity(op.descriptions)
        en_piv = english_pivot_score(embeddings, op.id, languages)

        # Composite: simple average of z-scored sub-metrics (z-score later)
        # For now store raw values; z-scoring happens in analyze()
        results.append({
            "op_id": op.id,
            "category": op.category,
            "token_overlap": tok_ov,
            "roman_sim": rom_sim,
            "en_pivot": en_piv,
            "d_intra": d_intra_map.get(op.id, 0.0),
        })

    # Z-score each sub-metric, then average for composite
    for metric in ["token_overlap", "roman_sim", "en_pivot"]:
        vals = np.array([r[metric] for r in results])
        mu, sigma = vals.mean(), vals.std()
        if sigma > 1e-10:
            z_vals = (vals - mu) / sigma
        else:
            z_vals = np.zeros_like(vals)
        for i, r in enumerate(results):
            r[f"{metric}_z"] = float(z_vals[i])

    for r in results:
        r["internationality"] = float(np.mean([
            r["token_overlap_z"],
            r["roman_sim_z"],
            r["en_pivot_z"],
        ]))

    return results


# ── Statistical Analysis ────────────────────────────────────────────────

def analyze_strategy1(scores: list[dict]) -> dict:
    """Full statistical analysis of Strategy 1 hypothesis.

    Tests:
      H0: Per-operation internationality score is uncorrelated with d_intra.
      H1: Higher internationality → lower d_intra (negative correlation).

    If H1 is supported AND the internationality difference between comp/judg
    categories is significant, then P2 failure is a vocabulary artifact.
    """
    internationality = np.array([s["internationality"] for s in scores])
    d_intra = np.array([s["d_intra"] for s in scores])

    # ── Test 1: Pearson & Spearman correlation (internationality vs d_intra)
    # Prediction: negative (more international vocab → lower d_intra)
    r_pearson, p_pearson = stats.pearsonr(internationality, d_intra)
    r_spearman, p_spearman = stats.spearmanr(internationality, d_intra)

    # ── Test 2: Group difference in internationality (comp vs judg)
    comp_intl = [s["internationality"] for s in scores if s["category"] == "computational"]
    judg_intl = [s["internationality"] for s in scores if s["category"] == "judgment"]
    t_stat, p_ttest = stats.ttest_ind(comp_intl, judg_intl, alternative="less")
    # alternative="less": H1 is comp_intl < judg_intl
    u_stat, p_mannwhitney = stats.mannwhitneyu(
        comp_intl, judg_intl, alternative="less"
    )

    # ── Test 3: Sub-metric group differences
    sub_metrics = {}
    for metric in ["token_overlap", "roman_sim", "en_pivot"]:
        comp_vals = [s[metric] for s in scores if s["category"] == "computational"]
        judg_vals = [s[metric] for s in scores if s["category"] == "judgment"]
        t, p = stats.ttest_ind(comp_vals, judg_vals, alternative="less")
        sub_metrics[metric] = {
            "comp_mean": float(np.mean(comp_vals)),
            "comp_std": float(np.std(comp_vals)),
            "judg_mean": float(np.mean(judg_vals)),
            "judg_std": float(np.std(judg_vals)),
            "t_stat": float(t),
            "p_value": float(p),
            "effect_size_d": float(
                (np.mean(judg_vals) - np.mean(comp_vals))
                / np.sqrt((np.var(comp_vals) + np.var(judg_vals)) / 2)
            ),
        }

    # ── Test 4: Partial correlation — does category still predict d_intra
    # after controlling for internationality?
    # If partial r(category, d_intra | internationality) ≈ 0, the vocabulary
    # confound fully explains the P2 result.
    category_binary = np.array([
        0 if s["category"] == "computational" else 1 for s in scores
    ])
    partial_r = _partial_correlation(category_binary, d_intra, internationality)

    # ── Test 5: English-pivot asymmetry (CLSD specific)
    # For each op, compute the std of {cos(en, lang) for lang in non-en}.
    # If judgment ops have lower std, they cluster tighter around English pivot.
    comp_en_pivot = [s["en_pivot"] for s in scores if s["category"] == "computational"]
    judg_en_pivot = [s["en_pivot"] for s in scores if s["category"] == "judgment"]

    # ── Test 6: Mediation analysis (Baron & Kenny, 1986)
    # Path a: category → internationality
    # Path b: internationality → d_intra (controlling for category)
    # Path c: category → d_intra (total effect)
    # Path c': category → d_intra (controlling for internationality)
    # Mediation = c - c'
    mediation = _mediation_analysis(category_binary, internationality, d_intra)

    # ── Interpret
    vocab_explains_p2 = (
        r_spearman < 0  # internationality negatively correlates with d_intra
        and p_spearman < 0.05  # significantly
        and p_mannwhitney < 0.05  # comp vs judg internationality differs
        and abs(partial_r["r"]) < abs(r_spearman) * 0.5  # partial r drops
    )

    return {
        "hypothesis": (
            "P2 failure (R_C < R_J) is explained by vocabulary internationality: "
            "judgment ops use more cross-linguistically shared vocabulary, "
            "yielding lower d_intra independent of Z_sem properties."
        ),
        "correlation": {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "interpretation": (
                "Negative r = higher internationality → lower d_intra (supports H1)"
            ),
        },
        "group_difference": {
            "comp_internationality_mean": float(np.mean(comp_intl)),
            "judg_internationality_mean": float(np.mean(judg_intl)),
            "t_stat": float(t_stat),
            "p_ttest": float(p_ttest),
            "u_stat": float(u_stat),
            "p_mannwhitney": float(p_mannwhitney),
            "effect_size_d": float(
                (np.mean(judg_intl) - np.mean(comp_intl))
                / np.sqrt((np.var(comp_intl) + np.var(judg_intl)) / 2)
            ),
        },
        "sub_metrics": sub_metrics,
        "partial_correlation": partial_r,
        "en_pivot_asymmetry": {
            "comp_en_pivot_mean": float(np.mean(comp_en_pivot)),
            "judg_en_pivot_mean": float(np.mean(judg_en_pivot)),
            "comp_en_pivot_std": float(np.std(comp_en_pivot)),
            "judg_en_pivot_std": float(np.std(judg_en_pivot)),
        },
        "mediation": mediation,
        "conclusion": {
            "vocab_explains_p2": vocab_explains_p2,
            "significance_threshold": 0.05,
            "interpretation": (
                "SUPPORTS paper argument: P2 failure is a measurement artifact"
                if vocab_explains_p2
                else "UNDERMINES paper argument: P2 failure is NOT fully explained by vocabulary"
            ),
        },
    }


def _partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> dict:
    """Partial correlation r(x, y | z) via residual method.

    Regress x on z → residual_x; regress y on z → residual_y.
    Correlate residuals.
    """
    # Residuals of x ~ z
    z_design = np.column_stack([z, np.ones(len(z))])
    beta_x = np.linalg.lstsq(z_design, x, rcond=None)[0]
    resid_x = x - z_design @ beta_x

    # Residuals of y ~ z
    beta_y = np.linalg.lstsq(z_design, y, rcond=None)[0]
    resid_y = y - z_design @ beta_y

    r, p = stats.pearsonr(resid_x, resid_y)
    return {
        "r": float(r),
        "p": float(p),
        "interpretation": (
            "If partial r ≈ 0, category has no effect on d_intra "
            "beyond what internationality already explains."
        ),
    }


def _mediation_analysis(
    x: np.ndarray,  # category (0=comp, 1=judg)
    m: np.ndarray,  # mediator (internationality)
    y: np.ndarray,  # outcome (d_intra)
) -> dict:
    """Baron & Kenny (1986) mediation: does internationality mediate
    the category → d_intra relationship?

    Path a: x → m
    Path b: m → y (controlling for x)
    Path c: x → y (total)
    Path c': x → y (controlling for m)
    Indirect effect = a * b (or c - c')
    Sobel test for significance of indirect effect.
    """
    n = len(x)

    # Path c: total effect (category → d_intra)
    X_c = np.column_stack([x, np.ones(n)])
    beta_c = np.linalg.lstsq(X_c, y, rcond=None)[0]
    c = beta_c[0]

    # Path a: category → internationality
    beta_a = np.linalg.lstsq(X_c, m, rcond=None)[0]
    a = beta_a[0]
    resid_a = m - X_c @ beta_a
    se_a = np.sqrt(np.sum(resid_a**2) / (n - 2) / np.sum((x - x.mean())**2))

    # Path b and c': category + internationality → d_intra
    X_bc = np.column_stack([x, m, np.ones(n)])
    beta_bc = np.linalg.lstsq(X_bc, y, rcond=None)[0]
    c_prime = beta_bc[0]  # direct effect
    b = beta_bc[1]        # internationality → d_intra controlling for category
    resid_bc = y - X_bc @ beta_bc
    mse = np.sum(resid_bc**2) / (n - 3)
    XtX_inv = np.linalg.inv(X_bc.T @ X_bc)
    se_b = np.sqrt(mse * XtX_inv[1, 1])

    # Indirect effect
    indirect = a * b

    # Sobel test
    se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    z_sobel = indirect / se_indirect if se_indirect > 1e-10 else 0.0
    p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))

    # Proportion mediated
    prop_mediated = indirect / c if abs(c) > 1e-10 else 0.0

    return {
        "path_a": float(a),
        "path_b": float(b),
        "path_c_total": float(c),
        "path_c_prime_direct": float(c_prime),
        "indirect_effect": float(indirect),
        "proportion_mediated": float(prop_mediated),
        "sobel_z": float(z_sobel),
        "sobel_p": float(p_sobel),
        "interpretation": (
            f"Internationality mediates {abs(prop_mediated)*100:.1f}% of the "
            f"category → d_intra effect. "
            f"Sobel test: z={z_sobel:.2f}, p={p_sobel:.4f}."
        ),
    }
