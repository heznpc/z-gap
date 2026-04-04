#!/usr/bin/env python3
"""Token overlap control for R_code: is NL-code alignment a lexical artifact?

If R_code > 1 is driven by shared tokens between NL descriptions and code
(e.g., "Sort the list" shares "sort"/"list" with `sorted(lst)`), then:
  1. token_overlap should negatively correlate with d_match
  2. R_code should drop for operations with zero token overlap
  3. Obfuscated code (variable names replaced) should show lower R_code

This script tests all three.
"""

import json
import re
import sys
import gc
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stimuli import get_all_operations, LANGUAGES
from src.embeddings import SentenceTransformerEmbedder, EmbeddingCache
from src.code_alignment import CODE_EQUIVALENTS

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "embeddings"

MODELS = [
    ("microsoft/unixcoder-base", "UniXcoder (code)"),
    ("paraphrase-multilingual-MiniLM-L12-v2", "MiniLM-L12 (NL)"),
    ("intfloat/multilingual-e5-large", "E5-large (NL)"),
]


# ── Token overlap computation ────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Extract lowercased alpha tokens (length >= 2), with stemming."""
    tokens = set(re.findall(r"[a-zA-Z]{2,}", text.lower()))
    # Simple stemming: remove common suffixes
    stemmed = set()
    for t in tokens:
        for suffix in ["ed", "ing", "tion", "ment", "ness", "ize", "ise", "ical"]:
            if t.endswith(suffix) and len(t) > len(suffix) + 2:
                t = t[:-len(suffix)]
                break
        stemmed.add(t)
    return stemmed


def token_overlap(nl_text: str, code_text: str) -> dict:
    """Compute token overlap between NL description and code snippet."""
    nl_tokens = _tokenize(nl_text)
    code_tokens = _tokenize(code_text)
    shared = nl_tokens & code_tokens
    union = nl_tokens | code_tokens
    jaccard = len(shared) / len(union) if union else 0.0
    return {
        "nl_tokens": sorted(nl_tokens),
        "code_tokens": sorted(code_tokens),
        "shared": sorted(shared),
        "jaccard": jaccard,
        "n_shared": len(shared),
    }


# ── Obfuscated code variants ────────────────────────────────────────

OBFUSCATED_CODE = {}
for op_id, code in CODE_EQUIVALENTS.items():
    # Replace common variable names with opaque single letters
    obf = code
    obf = re.sub(r'\blst\b', 'v0', obf)
    obf = re.sub(r'\blst1\b', 'v1', obf)
    obf = re.sub(r'\blst2\b', 'v2', obf)
    obf = re.sub(r'\bmatrix\b', 'v0', obf)
    obf = re.sub(r'\bs\b', 'v0', obf)
    obf = re.sub(r'\bs1\b', 'v1', obf)
    obf = re.sub(r'\bs2\b', 'v2', obf)
    obf = re.sub(r'\bn\b', 'v0', obf)
    obf = re.sub(r'\ba\b', 'v0', obf)
    obf = re.sub(r'\bb\b', 'v1', obf)
    obf = re.sub(r'\bd\b', 'v0', obf)
    obf = re.sub(r'\bd1\b', 'v1', obf)
    obf = re.sub(r'\bd2\b', 'v2', obf)
    obf = re.sub(r'\bt\b', 'v0', obf)
    obf = re.sub(r'\btarget\b', 'v1', obf)
    obf = re.sub(r'\brow\b', 'r', obf)
    obf = re.sub(r'\bcol\b', 'c', obf)
    obf = re.sub(r'\bsub\b', 'e', obf)
    obf = re.sub(r'\bA\b', 'M1', obf)
    obf = re.sub(r'\bB\b', 'M2', obf)
    OBFUSCATED_CODE[op_id] = obf


def run_model(model_name: str, label: str) -> dict:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    ops = get_all_operations()
    comp_ops = [op for op in ops if op.category == "computational"]
    comp_ids = [op.id for op in comp_ops]
    cache = EmbeddingCache(CACHE_DIR)
    model = SentenceTransformerEmbedder(model_name)

    # ── Embed NL descriptions ──
    nl_texts, nl_keys = [], []
    for op in comp_ops:
        for lang in LANGUAGES:
            desc = op.descriptions.get(lang)
            if desc:
                nl_texts.append(desc)
                nl_keys.append(f"{op.id}_{lang}")
    nl_array = cache.get_or_compute(model, nl_texts)
    nl_emb = {k: nl_array[i] for i, k in enumerate(nl_keys)}

    # ── Embed original code ──
    code_texts = [CODE_EQUIVALENTS[oid] for oid in comp_ids if oid in CODE_EQUIVALENTS]
    code_keys = [oid for oid in comp_ids if oid in CODE_EQUIVALENTS]
    code_array = cache.get_or_compute(model, code_texts)
    code_emb = {k: code_array[i] for i, k in enumerate(code_keys)}

    # ── Embed obfuscated code ──
    obf_texts = [OBFUSCATED_CODE[oid] for oid in code_keys]
    obf_array = cache.get_or_compute(model, obf_texts)
    obf_emb = {k: obf_array[i] for i, k in enumerate(code_keys)}

    valid_ids = [oid for oid in comp_ids if oid in code_emb]

    # ── TEST 1: Token overlap vs d_match correlation ──
    print("\n  [1/3] Token overlap vs d_match correlation")

    per_op_data = []
    for op_id in valid_ids:
        en_desc = next(op.descriptions["en"] for op in comp_ops if op.id == op_id)
        code = CODE_EQUIVALENTS[op_id]
        overlap = token_overlap(en_desc, code)

        # d_match for English
        en_key = f"{op_id}_en"
        if en_key in nl_emb:
            d_match = float(cosine(nl_emb[en_key], code_emb[op_id]))
        else:
            d_match = None

        per_op_data.append({
            "op_id": op_id,
            "overlap": overlap,
            "d_match_en": d_match,
        })

    # Correlation
    overlaps = np.array([d["overlap"]["jaccard"] for d in per_op_data if d["d_match_en"] is not None])
    d_matches = np.array([d["d_match_en"] for d in per_op_data if d["d_match_en"] is not None])
    rho, p = stats.spearmanr(overlaps, d_matches)
    print(f"    Spearman rho(token_overlap, d_match) = {rho:+.4f}  (p={p:.4f})")
    print(f"    Direction: {'MORE overlap → CLOSER (lexical confound!)' if rho < 0 else 'No lexical confound'}")

    # Show per-operation detail
    n_shared_counts = [d["overlap"]["n_shared"] for d in per_op_data]
    print(f"    Token overlap distribution: min={min(n_shared_counts)}, "
          f"max={max(n_shared_counts)}, mean={np.mean(n_shared_counts):.1f}")
    zero_overlap = [d for d in per_op_data if d["overlap"]["n_shared"] == 0]
    print(f"    Operations with ZERO token overlap: {len(zero_overlap)}/{len(per_op_data)}")

    # ── TEST 2: R_code for zero-overlap vs nonzero-overlap ops ──
    print("\n  [2/3] R_code by overlap group")

    zero_ids = [d["op_id"] for d in per_op_data if d["overlap"]["n_shared"] == 0]
    nonzero_ids = [d["op_id"] for d in per_op_data if d["overlap"]["n_shared"] > 0]

    def compute_R_for_subset(subset_ids, all_valid_ids):
        """R_code for a subset, using all valid_ids for mismatch pool."""
        d_match, d_mismatch = [], []
        for op_id in subset_ids:
            for lang in LANGUAGES:
                nl_key = f"{op_id}_{lang}"
                if nl_key not in nl_emb or op_id not in code_emb:
                    continue
                d_match.append(float(cosine(nl_emb[nl_key], code_emb[op_id])))
                for other_id in all_valid_ids:
                    if other_id != op_id:
                        d_mismatch.append(float(cosine(nl_emb[nl_key], code_emb[other_id])))
        if not d_match:
            return None, 0, 0
        R = np.mean(d_mismatch) / np.mean(d_match)
        return float(R), len(d_match), len(d_mismatch)

    R_zero, n_z, _ = compute_R_for_subset(zero_ids, valid_ids)
    R_nonzero, n_nz, _ = compute_R_for_subset(nonzero_ids, valid_ids)
    R_all, n_all, _ = compute_R_for_subset(valid_ids, valid_ids)

    print(f"    Zero overlap ({len(zero_ids)} ops):    R_code = {R_zero:.4f}" if R_zero else "    Zero overlap: N/A")
    print(f"    Nonzero overlap ({len(nonzero_ids)} ops): R_code = {R_nonzero:.4f}")
    print(f"    All ops ({len(valid_ids)} ops):       R_code = {R_all:.4f}")

    if R_zero is not None and R_zero > 1.0:
        print(f"    ** CRITICAL: R_code > 1 even with ZERO token overlap → NOT a lexical artifact **")
    elif R_zero is not None:
        print(f"    ** WARNING: R_code <= 1 for zero-overlap ops → lexical confound possible **")

    # ── TEST 3: Obfuscated code R_code ──
    print("\n  [3/3] Obfuscated code R_code")

    d_match_obf, d_mismatch_obf = [], []
    for op_id in valid_ids:
        for lang in LANGUAGES:
            nl_key = f"{op_id}_{lang}"
            if nl_key not in nl_emb or op_id not in obf_emb:
                continue
            d_match_obf.append(float(cosine(nl_emb[nl_key], obf_emb[op_id])))
            for other_id in valid_ids:
                if other_id != op_id:
                    d_mismatch_obf.append(float(cosine(nl_emb[nl_key], obf_emb[other_id])))

    R_obf = np.mean(d_mismatch_obf) / np.mean(d_match_obf) if d_match_obf else 0
    R_orig = R_all

    print(f"    Original code:   R_code = {R_orig:.4f}  d_match = {np.mean([d['d_match_en'] for d in per_op_data if d['d_match_en']]):.4f}")
    print(f"    Obfuscated code: R_code = {R_obf:.4f}  d_match = {np.mean(d_match_obf):.4f}")
    delta = R_orig - R_obf
    pct = delta / R_orig * 100 if R_orig > 0 else 0
    print(f"    Drop: {delta:+.4f} ({pct:+.1f}%)")

    if R_obf > 1.0:
        print(f"    ** R_code > 1 survives obfuscation → semantic alignment, not lexical **")
    else:
        print(f"    ** R_code <= 1 after obfuscation → lexical artifact confirmed **")

    # Show examples of obfuscation
    print(f"\n    Obfuscation examples:")
    for oid in valid_ids[:5]:
        print(f"      {oid}: {CODE_EQUIVALENTS[oid]!r:30s} → {OBFUSCATED_CODE[oid]!r}")

    del model, nl_array, code_array, obf_array
    gc.collect()

    return {
        "model": model_name, "label": label,
        "test1_correlation": {
            "spearman_rho": float(rho), "p": float(p),
            "n_zero_overlap": len(zero_ids),
        },
        "test2_by_group": {
            "R_zero_overlap": R_zero,
            "R_nonzero_overlap": R_nonzero,
            "R_all": R_all,
            "n_zero": len(zero_ids),
            "n_nonzero": len(nonzero_ids),
        },
        "test3_obfuscation": {
            "R_original": R_orig,
            "R_obfuscated": R_obf,
            "drop_absolute": delta,
            "drop_pct": pct,
            "survives": R_obf > 1.0,
        },
    }


def main():
    print("=" * 65)
    print("R_code Token Overlap Control")
    print("Is NL-code alignment a lexical artifact?")
    print("=" * 65)

    all_results = []
    for model_name, label in MODELS:
        result = run_model(model_name, label)
        all_results.append(result)

    # ── Cross-model summary ──
    print(f"\n{'='*65}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*65}")

    print(f"\n{'Model':<25s}  {'rho(overlap,d)':>14s}  {'R_zero':>7s}  {'R_orig':>7s}  {'R_obf':>7s}  {'Survives':>8s}")
    print(f"{'─'*75}")
    for r in all_results:
        t1 = r["test1_correlation"]
        t2 = r["test2_by_group"]
        t3 = r["test3_obfuscation"]
        rho_str = f"{t1['spearman_rho']:+.3f} (p={t1['p']:.3f})"
        R_z = f"{t2['R_zero_overlap']:.3f}" if t2['R_zero_overlap'] else "N/A"
        print(f"{r['label']:<25s}  {rho_str:>14s}  {R_z:>7s}  "
              f"{t3['R_original']:.3f}  {t3['R_obfuscated']:.3f}  "
              f"{'YES' if t3['survives'] else 'NO':>8s}")

    # Verdict
    all_survive = all(r["test3_obfuscation"]["survives"] for r in all_results)
    if all_survive:
        print(f"\n  VERDICT: R_code > 1 survives obfuscation in ALL models.")
        print(f"  NL-code alignment is NOT purely a lexical artifact.")
    else:
        failed = [r["label"] for r in all_results if not r["test3_obfuscation"]["survives"]]
        print(f"\n  VERDICT: R_code drops to <= 1 after obfuscation for: {', '.join(failed)}")
        print(f"  Lexical confound is present in these models.")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rcode_token_control.json"
    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return obj
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    main()
