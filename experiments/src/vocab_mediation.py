"""Strategy A: Vocabulary mediation analysis for P2.

Tests whether text-level vocabulary features of NL descriptions predict
per-operation d_intra. This is NOT circular because the features are
computed from raw text (character counts, word counts, technical term
density), NOT from the same embeddings used to compute d_intra.

Hypothesis: Computational operations require domain-specific vocabulary
(e.g., "transpose", "palindrome", "cumulative sum") that translates
differently across languages, producing higher d_intra. This vocabulary
divergence IS the communicability gap manifesting at the description
level --- not an artifact, but the mechanism by which Z_sem convergence
fails to guarantee communicability.

Design avoids all flaws from the previous Strategy 1:
  (F1) No cross-script token overlap between ko/zh/ar
  (F2) No embedding-derived features (eliminates circularity)
  (F3) No Baron & Kenny mediation (inappropriate for n=100)
  (F4) Bonferroni correction for all correlations tested
  (F5) Framed as "vocabulary mediates the gap" not "artifact"
"""

from __future__ import annotations

import re
import unicodedata
from itertools import combinations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Technical vocabulary list (English)
# These are terms whose translations tend to be domain-specific and divergent
# across languages. Compiled from the 50 computational operation descriptions.
# ---------------------------------------------------------------------------
TECHNICAL_TERMS_EN = frozenset([
    # Data structure terms
    "list", "dictionary", "set", "matrix", "tree", "array", "string",
    "nested", "key", "keys", "element", "elements", "index", "value",
    # Operation terms
    "sort", "filter", "reverse", "count", "sum", "concatenate",
    "uppercase", "split", "replace", "merge", "transpose", "flatten",
    "zip", "slice", "deduplicate", "double", "multiply",
    # Math terms
    "absolute", "power", "remainder", "square", "root", "divisor",
    "median", "average", "cumulative", "binary", "prime", "modulo",
    "palindrome", "gcd",
    # Quantitative modifiers
    "ascending", "descending", "positive", "negative", "even",
    "largest", "smallest", "maximum", "minimum",
])


# ---------------------------------------------------------------------------
# Feature 1: Description length
# ---------------------------------------------------------------------------

def char_count(text: str) -> int:
    """Character count (including spaces). Language-agnostic."""
    return len(text)


def word_count_for_lang(text: str, lang: str) -> int:
    """Approximate word count, language-aware.

    For space-delimited languages (en, es, ko, ar): split on whitespace.
    For zh: count characters (each character ~ one morpheme).
    This is a rough proxy, not a tokenizer --- sufficient for correlation.
    """
    if lang == "zh":
        # Chinese: count non-whitespace characters as word-units
        return len(re.sub(r"\s+", "", text))
    else:
        # Space-delimited: split on whitespace
        return len(text.split())


# ---------------------------------------------------------------------------
# Feature 2: Technical term density (English descriptions only)
# ---------------------------------------------------------------------------

def technical_term_count(en_description: str) -> int:
    """Count how many tokens in the English description match TECHNICAL_TERMS_EN.

    Uses only the English description to avoid cross-script issues (F1).
    """
    tokens = set(re.findall(r"[a-zA-Z]+", en_description.lower()))
    return len(tokens & TECHNICAL_TERMS_EN)


def technical_term_ratio(en_description: str) -> float:
    """Fraction of English tokens that are technical terms."""
    tokens = re.findall(r"[a-zA-Z]+", en_description.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in TECHNICAL_TERMS_EN)
    return hits / len(tokens)


# ---------------------------------------------------------------------------
# Feature 3: En-Es cognate ratio (the ONE valid cross-lingual text feature)
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Remove diacritics: 'Evalua' -> 'Evalua', 'numero' -> 'numero'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _latin_tokens(text: str) -> list[str]:
    """Extract lowercased Latin-script tokens after accent stripping."""
    return re.findall(r"[a-z]{2,}", _strip_accents(text.lower()))


def en_es_cognate_ratio(en_desc: str, es_desc: str, min_prefix: int = 4) -> float:
    """Cognate ratio between English and Spanish descriptions.

    Only computed for en-es (both Latin-script). Avoids F1 entirely.
    A cognate pair is defined as two tokens sharing a prefix >= min_prefix
    characters after accent stripping (e.g., "calculate"/"calcula" share
    "calc", "ascending"/"ascendente" share "asce").

    Returns: fraction of English tokens that have a cognate in Spanish.
    """
    en_tokens = _latin_tokens(en_desc)
    es_tokens = _latin_tokens(es_desc)

    if not en_tokens:
        return 0.0

    cognate_hits = 0
    for en_tok in en_tokens:
        for es_tok in es_tokens:
            prefix_len = min(len(en_tok), len(es_tok), min_prefix)
            if prefix_len >= min_prefix and en_tok[:prefix_len] == es_tok[:prefix_len]:
                cognate_hits += 1
                break  # each en token matches at most once

    return cognate_hits / len(en_tokens)


# ---------------------------------------------------------------------------
# Feature 4: Cross-lingual length variance
# ---------------------------------------------------------------------------

def cross_lingual_length_cv(descriptions: dict[str, str]) -> float:
    """Coefficient of variation of character counts across languages.

    High CV = descriptions vary a lot in length across languages,
    suggesting the concept requires very different amounts of text
    to express in different languages.
    """
    lengths = [len(desc) for desc in descriptions.values()]
    if not lengths:
        return 0.0
    mu = np.mean(lengths)
    if mu < 1e-10:
        return 0.0
    return float(np.std(lengths) / mu)


# ---------------------------------------------------------------------------
# Assemble per-operation feature vectors
# ---------------------------------------------------------------------------

def compute_text_features(operations: list) -> list[dict]:
    """Compute text-level features for each operation.

    All features derived from raw text. NO embeddings used.

    Returns list of dicts with:
        op_id, category,
        en_char_count, en_word_count,
        mean_char_count, mean_word_count,
        technical_count, technical_ratio,
        en_es_cognate, length_cv
    """
    results = []
    for op in operations:
        descs = op.descriptions  # {lang: str}

        # Per-language lengths
        char_counts = {lang: char_count(descs[lang]) for lang in descs}
        word_counts = {lang: word_count_for_lang(descs[lang], lang) for lang in descs}

        en_desc = descs.get("en", "")
        es_desc = descs.get("es", "")

        results.append({
            "op_id": op.id,
            "category": op.category,
            # Feature 1: Description length
            "en_char_count": char_counts.get("en", 0),
            "en_word_count": word_counts.get("en", 0),
            "mean_char_count": float(np.mean(list(char_counts.values()))),
            "mean_word_count": float(np.mean(list(word_counts.values()))),
            # Feature 2: Technical term density (English only --- avoids F1)
            "technical_count": technical_term_count(en_desc),
            "technical_ratio": technical_term_ratio(en_desc),
            # Feature 3: Cognate ratio (en-es only --- avoids F1)
            "en_es_cognate": en_es_cognate_ratio(en_desc, es_desc),
            # Feature 4: Cross-lingual length variance
            "length_cv": cross_lingual_length_cv(descs),
        })

    return results


# ---------------------------------------------------------------------------
# Attach d_intra from embedding results (read-only, pre-computed)
# ---------------------------------------------------------------------------

def attach_d_intra(
    text_features: list[dict],
    d_intra_map: dict[str, float],
) -> list[dict]:
    """Attach per-operation d_intra values to the text feature records.

    d_intra_map: {op_id: d_intra} from compute_d_intra() or equivalent.
    This is the ONLY point where embedding-derived data enters, and it
    enters as the DEPENDENT variable, not as a feature.
    """
    for record in text_features:
        record["d_intra"] = d_intra_map.get(record["op_id"], float("nan"))
    return text_features


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

# The feature names we will test. This determines the Bonferroni denominator.
FEATURE_NAMES = [
    "en_char_count",
    "en_word_count",
    "mean_char_count",
    "mean_word_count",
    "technical_count",
    "technical_ratio",
    "en_es_cognate",
    "length_cv",
]
NUM_FEATURES = len(FEATURE_NAMES)  # = 8
# Bonferroni correction: alpha / NUM_FEATURES
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / NUM_FEATURES


def _spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42):
    """Spearman rho with bootstrap 95% CI for the correlation coefficient."""
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = []
    import warnings
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            r_b, _ = stats.spearmanr(x[idx], y[idx])
        if np.isnan(r_b):
            r_b = 0.0  # constant resample -> zero correlation
        boot_rhos.append(r_b)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    return float(rho), float(p), (float(ci_lo), float(ci_hi))


def analyze_vocabulary_mediation(records: list[dict]) -> dict:
    """Run the full Strategy A analysis.

    Primary analysis:
      Spearman correlation of each text feature with per-operation d_intra,
      Bonferroni-corrected across NUM_FEATURES tests.

    Secondary analysis:
      Within-category correlations (computational only, judgment only).
      If a feature predicts d_intra WITHIN categories, the effect is
      genuine and not just a proxy for the category label.

    Power analysis:
      Report the minimum detectable rho at alpha=BONFERRONI_ALPHA for
      n=50 (per category) and n=100 (pooled) at 80% power.

    Returns a structured dict suitable for JSON serialization.
    """
    d_intra = np.array([r["d_intra"] for r in records])
    categories = np.array([r["category"] for r in records])

    comp_mask = categories == "computational"
    judg_mask = categories == "judgment"
    n_comp = int(comp_mask.sum())
    n_judg = int(judg_mask.sum())

    # ── Primary: Pooled correlations (n=100) ──
    pooled_results = {}
    for feat in FEATURE_NAMES:
        x = np.array([r[feat] for r in records])
        rho, p_raw, ci = _spearman_with_ci(x, d_intra)
        pooled_results[feat] = {
            "rho": rho,
            "p_raw": p_raw,
            "p_bonferroni": min(p_raw * NUM_FEATURES, 1.0),
            "significant_bonferroni": (p_raw * NUM_FEATURES) < ALPHA,
            "ci_95": list(ci),
            "n": len(x),
        }

    # ── Secondary: Within-category correlations ──
    within_comp = {}
    within_judg = {}
    for feat in FEATURE_NAMES:
        x_all = np.array([r[feat] for r in records])

        # Computational only
        x_c = x_all[comp_mask]
        d_c = d_intra[comp_mask]
        if len(x_c) > 3 and np.std(x_c) > 1e-10:
            rho_c, p_c, ci_c = _spearman_with_ci(x_c, d_c)
        else:
            rho_c, p_c, ci_c = 0.0, 1.0, (0.0, 0.0)
        within_comp[feat] = {
            "rho": rho_c, "p_raw": p_c, "ci_95": list(ci_c), "n": n_comp,
        }

        # Judgment only
        x_j = x_all[judg_mask]
        d_j = d_intra[judg_mask]
        if len(x_j) > 3 and np.std(x_j) > 1e-10:
            rho_j, p_j, ci_j = _spearman_with_ci(x_j, d_j)
        else:
            rho_j, p_j, ci_j = 0.0, 1.0, (0.0, 0.0)
        within_judg[feat] = {
            "rho": rho_j, "p_raw": p_j, "ci_95": list(ci_j), "n": n_judg,
        }

    # ── Descriptive: feature means by category ──
    descriptive = {}
    for feat in FEATURE_NAMES:
        vals = np.array([r[feat] for r in records])
        c_vals = vals[comp_mask]
        j_vals = vals[judg_mask]
        # Mann-Whitney for group difference (non-parametric)
        if len(c_vals) > 0 and len(j_vals) > 0:
            u, p_mw = stats.mannwhitneyu(c_vals, j_vals, alternative="two-sided")
        else:
            u, p_mw = 0.0, 1.0
        descriptive[feat] = {
            "comp_mean": float(np.mean(c_vals)),
            "comp_std": float(np.std(c_vals)),
            "judg_mean": float(np.mean(j_vals)),
            "judg_std": float(np.std(j_vals)),
            "mannwhitney_U": float(u),
            "mannwhitney_p": float(p_mw),
        }

    # ── Power analysis ──
    # For Spearman at alpha_bonf, approximate via normal z-transform.
    # Minimum detectable rho at 80% power:
    #   rho_min ~ (z_alpha + z_beta) / sqrt(n - 3)
    # where z_alpha = z(alpha_bonf/2) for two-sided, z_beta = z(0.80) = 0.842
    z_alpha = stats.norm.ppf(1 - BONFERRONI_ALPHA / 2)
    z_beta = 0.842  # 80% power

    rho_min_100 = np.tanh((z_alpha + z_beta) / np.sqrt(100 - 3))
    rho_min_50 = np.tanh((z_alpha + z_beta) / np.sqrt(50 - 3))

    # ── Interpretation logic ──
    # A feature is a genuine vocabulary mediator if:
    #   (a) significant pooled correlation with d_intra (after Bonferroni), AND
    #   (b) nonzero within-category correlation (at least one category rho > 0.15)
    # If (a) but not (b): the feature is just a proxy for the category label.

    genuine_mediators = []
    category_proxies = []
    for feat in FEATURE_NAMES:
        pooled_sig = pooled_results[feat]["significant_bonferroni"]
        rho_within_c = abs(within_comp[feat]["rho"])
        rho_within_j = abs(within_judg[feat]["rho"])
        has_within = (rho_within_c > 0.15) or (rho_within_j > 0.15)

        if pooled_sig and has_within:
            genuine_mediators.append(feat)
        elif pooled_sig and not has_within:
            category_proxies.append(feat)

    return {
        "design": {
            "hypothesis": (
                "Domain-specific vocabulary in computational operation descriptions "
                "translates differently across languages, producing higher d_intra. "
                "Vocabulary divergence IS the communicability gap at the description "
                "level, not an artifact."
            ),
            "features_tested": FEATURE_NAMES,
            "num_features": NUM_FEATURES,
            "alpha": ALPHA,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "circularity_check": (
                "All features computed from raw text (character counts, word counts, "
                "keyword matching, cognate overlap). No embeddings used for features. "
                "d_intra enters only as the dependent variable."
            ),
            "cross_script_check": (
                "No cross-script comparisons between ko/zh/ar. Technical terms "
                "counted from English only. Cognate ratio computed for en-es only "
                "(both Latin script). Length features are script-agnostic."
            ),
        },
        "pooled_correlations": pooled_results,
        "within_category": {
            "computational": within_comp,
            "judgment": within_judg,
        },
        "descriptive_by_category": descriptive,
        "power_analysis": {
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "min_detectable_rho_n100": float(rho_min_100),
            "min_detectable_rho_n50": float(rho_min_50),
            "note": (
                f"At alpha={BONFERRONI_ALPHA:.4f} (Bonferroni-corrected) and 80% "
                f"power, we can detect |rho| >= {rho_min_100:.3f} with n=100 "
                f"(pooled) or |rho| >= {rho_min_50:.3f} with n=50 (per category)."
            ),
        },
        "interpretation": {
            "genuine_mediators": genuine_mediators,
            "category_proxies": category_proxies,
            "summary": _build_summary(genuine_mediators, category_proxies),
        },
    }


def _build_summary(genuine: list[str], proxies: list[str]) -> str:
    """Build a human-readable interpretation string."""
    parts = []

    if genuine:
        parts.append(
            f"Genuine vocabulary mediators (predict d_intra WITHIN categories): "
            f"{', '.join(genuine)}. These features reflect how domain-specific "
            f"vocabulary drives cross-lingual description divergence."
        )
    if proxies:
        parts.append(
            f"Category-confounded features (predict d_intra only BETWEEN "
            f"categories): {', '.join(proxies)}. These merely recapitulate "
            f"the comp/judg split and do not explain within-category variance."
        )
    if not genuine and not proxies:
        parts.append(
            "No text-level vocabulary features significantly predict d_intra "
            "after Bonferroni correction. The d_intra difference between "
            "computational and judgment operations is not driven by surface "
            "vocabulary features measurable from the descriptions."
        )

    if genuine:
        parts.append(
            "Framing: vocabulary divergence is the MECHANISM of the communicability "
            "gap, not an artifact. Computational intent requires specialized terms "
            "that translate differently, producing higher d_intra. This supports "
            "Theorem 1: convergence at Z_sem does not guarantee communicability."
        )

    return " ".join(parts)
