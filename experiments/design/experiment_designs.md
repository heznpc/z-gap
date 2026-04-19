# Experiment Designs: Strategy D (Enhanced NL-Code Alignment) and Strategy 6-R (Minimal Dialect Evidence)

> 2026-04-04. Post-audit redesign. Two experiments with complete pseudocode, feasibility, and falsification conditions.

---

## Strategy D: Enhanced NL-Code Alignment (Per-Language x Per-Model R_code Matrix)

### Motivation

The paper's strongest empirical finding is NL-code alignment: R_code > 1 for both UniXcoder (1.04) and MiniLM (1.18), with permutation p < 0.0001. This directly demonstrates "convergence != communicability" when juxtaposed with P2 failure (R_C < R_J). The current evidence uses only two models, one of which (UniXcoder) is from 2022. Adding modern code embedding models and decomposing R_code by language creates a 5x4+ matrix that either strengthens or falsifies the central claim.

### Current State of the Codebase

- `experiments/src/code_alignment.py`: `compute_nl_code_alignment()` returns aggregate R_code + `per_lang_d_match` (match distances only, not per-language R_code).
- `experiments/scripts/run_code_alignment.py`: Runs UniXcoder + MiniLM. Hardcodes two models.
- `experiments/scripts/run_code_alignment_significance.py`: Permutation test + bootstrap CI + Cohen's d. Only for aggregate R_code, not per-language.
- `experiments/src/embeddings.py`: Has `SentenceTransformerEmbedder`, `OpenAIEmbedder`, `MistralEmbedder`. No Nomic or CodeSage class.
- `experiments/results/code_alignment_significance.json`: UniXcoder R_code=1.04, CI=[1.02, 1.07]; MiniLM R_code=1.18, CI=[1.15, 1.22].
- jina-embeddings-v3 and codestral-embed are in `run_all.py`'s model list but were only run for P2/P7, NOT for code alignment.

### Design

#### 1. Models (4 total: 2 existing + 2 new)

| Model | Params | Dim | Source | License | Why |
|-------|--------|-----|--------|---------|-----|
| UniXcoder-base | 125M | 768 | Microsoft, 2022 | MIT | Existing baseline, code-trained |
| MiniLM-L12 | 118M | 384 | sentence-transformers | Apache-2.0 | Existing baseline, NL-only |
| Nomic Embed Code | 7B | 768 | Nomic AI, ICLR 2025 | Apache-2.0 | SOTA code embedding, CoRNStack-trained |
| CodeSage-Large-v2 | 1.3B | 1024 | Microsoft, ICML 2024 | MIT | Code-specific, bimodal NL+code |

**Why NOT jina-embeddings-v3 or codestral-embed for code alignment:**
- jina-embeddings-v3: General multilingual, not code-specialized. Would replicate MiniLM's role.
- codestral-embed: API-based (Mistral), cost per embedding. Can add as optional 5th model.

**Nomic Embed Code access:** Available via HuggingFace (nomic-ai/nomic-embed-code), loads through sentence-transformers with `trust_remote_code=True`. Requires ~14GB VRAM for full 7B; can use GGUF quantized version or the smaller nomic-embed-text-v1.5 (137M, 768d) as fallback.

**CodeSage-Large-v2 access:** HuggingFace (microsoft/codesage-large-v2). Loads via transformers AutoModel. Requires custom encode function (not sentence-transformers compatible out of the box).

#### 2. Per-Language R_code Computation

The key change: compute R_code separately for each (language, model) pair, producing a 5x4 matrix.

```
PSEUDOCODE: compute_per_language_R_code(nl_embeddings, code_embeddings, comp_ids, languages)

    for each lang in languages:
        d_match_lang = []
        d_mismatch_lang = []

        for each op_id in comp_ids:
            if op_id not in code_embeddings: continue
            code_vec = code_embeddings[op_id]
            nl_key = f"{op_id}_{lang}"
            if nl_key not in nl_embeddings: continue

            # d_match: THIS language's description -> THIS operation's code
            d_match_lang.append(cosine(nl_embeddings[nl_key], code_vec))

            # d_mismatch: THIS language's description -> OTHER operations' code
            for other_id in comp_ids:
                if other_id == op_id: continue
                if other_id not in code_embeddings: continue
                d_mismatch_lang.append(cosine(nl_embeddings[nl_key], code_embeddings[other_id]))

        R_code_lang = mean(d_mismatch_lang) / mean(d_match_lang)
        # Note: use ALL mismatches, not sampled 10. Per-language has only 50 match pairs.

    return {lang: R_code_lang for lang in languages}
```

**Critical difference from current code:** The existing `compute_nl_code_alignment()` pools all languages for d_match and samples only 10 mismatched operations x 2 languages for d_mismatch. The per-language version uses ALL 49 mismatched operations for each language, giving 50 match pairs and 50*49=2450 mismatch pairs per language. This is computationally trivial (50 ops x 5 langs x 50 ops = 12,500 cosine distances per model).

#### 3. Statistical Tests (Per Cell)

For each (language, model) cell in the 5x4 matrix:

```
PSEUDOCODE: per_cell_statistics(d_match_lang, d_mismatch_lang, n_perm=10000, n_boot=10000)

    observed_R = mean(d_mismatch_lang) / mean(d_match_lang)

    # 1. Permutation test: H0 is that NL-code pairing is arbitrary
    #    Shuffle which code snippet each NL description is "matched" to
    perm_R_values = []
    for i in 1..n_perm:
        shuffled_code_ids = random_permutation(comp_ids)
        pairing = {comp_ids[j]: shuffled_code_ids[j] for j in range(len(comp_ids))}
        # Recompute d_match under shuffled pairing
        d_match_perm = [cosine(nl_embeddings[f"{op}_{lang}"],
                               code_embeddings[pairing[op]]) for op in comp_ids]
        # d_mismatch unchanged (it's already "other ops")
        perm_R = mean(d_mismatch_lang) / mean(d_match_perm)
        perm_R_values.append(perm_R)
    p_value = fraction(perm_R_values >= observed_R)

    # 2. Bootstrap CI for R_code
    boot_R = []
    for i in 1..n_boot:
        idx_m = resample_with_replacement(len(d_match_lang))
        idx_mm = resample_with_replacement(len(d_mismatch_lang))
        boot_R.append(mean(d_mismatch_lang[idx_mm]) / mean(d_match_lang[idx_m]))
    ci_lo, ci_hi = percentile(boot_R, [2.5, 97.5])

    # 3. Cohen's d
    s_pooled = sqrt(((n_m-1)*std(d_match)^2 + (n_mm-1)*std(d_mismatch)^2) / (n_m + n_mm - 2))
    d = (mean(d_mismatch) - mean(d_match)) / s_pooled

    return observed_R, p_value, ci_lo, ci_hi, d
```

**Multiple comparison correction:** 20 cells (5 languages x 4 models). Apply Holm-Bonferroni to p-values before reporting.

#### 4. Complete Pipeline Pseudocode

```
PSEUDOCODE: run_enhanced_code_alignment()

    ops = get_all_operations()
    comp_ids = [op.id for op in ops if op.category == "computational"]
    languages = ["en", "ko", "zh", "ar", "es"]

    models = [
        SentenceTransformerEmbedder("microsoft/unixcoder-base"),
        SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2"),
        NomicCodeEmbedder("nomic-ai/nomic-embed-code"),     # new class needed
        CodeSageEmbedder("microsoft/codesage-large-v2"),     # new class needed
    ]

    cache = EmbeddingCache(CACHE_DIR)
    results_matrix = {}  # (model_name, lang) -> {R_code, p, ci, d}

    for model in models:
        # Embed NL descriptions (all languages)
        nl_texts, nl_keys = [], []
        for op in ops:
            if op.category != "computational": continue
            for lang in languages:
                desc = op.descriptions.get(lang)
                if desc:
                    nl_texts.append(desc)
                    nl_keys.append(f"{op.id}_{lang}")
        nl_array = cache.get_or_compute(model, nl_texts)
        nl_embeddings = {k: nl_array[i] for i, k in enumerate(nl_keys)}

        # Embed code snippets
        code_texts = [CODE_EQUIVALENTS[op_id] for op_id in comp_ids
                      if op_id in CODE_EQUIVALENTS]
        code_keys = [op_id for op_id in comp_ids if op_id in CODE_EQUIVALENTS]
        code_array = cache.get_or_compute(model, code_texts)
        code_embeddings = {k: code_array[i] for i, k in enumerate(code_keys)}

        # Per-language R_code
        for lang in languages:
            d_match, d_mismatch = collect_per_lang_distances(
                nl_embeddings, code_embeddings, comp_ids, lang)
            R, p, ci_lo, ci_hi, cohens_d = per_cell_statistics(
                d_match, d_mismatch)
            results_matrix[(model.name, lang)] = {
                "R_code": R, "p_value": p,
                "ci_95": (ci_lo, ci_hi), "cohens_d": cohens_d,
                "d_match_mean": mean(d_match), "d_mismatch_mean": mean(d_mismatch),
                "n_match": len(d_match), "n_mismatch": len(d_mismatch),
            }

        # Aggregate R_code (for backward compatibility)
        all_d_match = concatenate all per-lang d_match
        all_d_mismatch = concatenate all per-lang d_mismatch
        results_matrix[(model.name, "aggregate")] = per_cell_statistics(
            all_d_match, all_d_mismatch)

    # Holm-Bonferroni correction on all 20 per-language p-values
    all_p_values = [results_matrix[(m, l)]["p_value"]
                    for m in model_names for l in languages]
    corrected_p = holm_bonferroni(all_p_values)
    # Assign corrected p-values back

    # Output: 5x4 matrix + aggregate row
    save_json(results_matrix)
    plot_heatmap(results_matrix)  # R_code values, color-coded
    plot_per_lang_bar(results_matrix)  # grouped bar: language x model
```

#### 5. New Embedding Classes Needed

```python
# NomicCodeEmbedder
class NomicCodeEmbedder(EmbeddingModel):
    """Nomic Embed Code (ICLR 2025, CoRNStack-trained)."""
    def __init__(self, model_name="nomic-ai/nomic-embed-code"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._model_name = model_name
    def encode(self, texts):
        return self._model.encode(texts, normalize_embeddings=True)
    # ... name, dimension properties

# CodeSageEmbedder
class CodeSageEmbedder(EmbeddingModel):
    """CodeSage-Large-v2 (ICML 2024, bimodal NL+code)."""
    def __init__(self, model_name="microsoft/codesage-large-v2"):
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model_name = model_name
    def encode(self, texts):
        # Mean pooling over last hidden state
        # Batch processing with padding
        ...
    # ... name, dimension properties
```

### Predictions

| Language | Predicted R_code rank | Rationale |
|----------|----------------------|-----------|
| en | Highest | English dominates code training data (GitHub ~60% English comments) |
| es | Second | Romance language, heavy in HuggingFace/GitHub |
| zh | Middle | Large code corpus but script distance from code tokens |
| ko | Low | Small code corpus, agglutinative morphology |
| ar | Lowest | Smallest code corpus, RTL, few Arabic-commented repos |

**D_train effect prediction:** The ranking should track the language's representation in code training corpora. If it instead tracks typological distance from English, that points to a different mechanism (structural transfer rather than data frequency).

**Code-trained vs NL-only models:** Code-trained models (UniXcoder, Nomic, CodeSage) should show higher R_code than NL-only (MiniLM) because their training explicitly aligns NL and code modalities. If MiniLM beats code-trained models, it suggests R_code is driven by surface lexical overlap rather than semantic alignment.

### Feasibility

| Item | Estimate |
|------|----------|
| Compute: Nomic 7B embeddings | ~2h on single A100 (250 NL + 50 code = 300 texts) |
| Compute: CodeSage-Large | ~30min on single GPU |
| Compute: Per-cell permutation (20 cells x 10k perms) | ~4h total (parallelizable) |
| API costs | $0 (all local models) |
| Data requirements | Already have all 50 comp ops x 5 languages + 50 code snippets |
| Human labor | 0 (no new stimuli needed) |
| **Total calendar time** | **1-2 days** |

**Fallback if Nomic 7B is too large:** Use nomic-embed-text-v1.5 (137M, 768d) instead. Still recent (2024), still from the Nomic family, and runs on consumer GPU.

### What the Result Means for the Paper

**Best case (expected):** R_code > 1 for all 20 cells. English has highest R_code, Korean/Arabic lowest. Code-trained models show higher R_code than MiniLM. This creates a rich 5x4 table for the paper that simultaneously demonstrates:
1. NL-code convergence is real (all R_code > 1)
2. It is modulated by D_train (language ranking)
3. It is modulated by model architecture (code-trained > NL-only)
4. It coexists with P2 failure (convergence != communicability)

**Paper narrative:** "Table X shows R_code > 1 for every language-model pair (20/20 after Holm-Bonferroni correction), confirming NL-code alignment. However, R_code varies from [lowest] (Korean, UniXcoder) to [highest] (English, Nomic), revealing D_train dependence. Combined with P2 failure (R_C < R_J), this directly demonstrates our thesis: the computational semantics of code converge in Z, but the NL descriptions used to communicate that semantics diverge across languages."

### Falsification Conditions

1. **R_code <= 1 for any language on a code-trained model** after Bonferroni correction. This would mean that even with explicit NL-code training, some languages' NL descriptions are no closer to matching code than to mismatching code. Severity: moderate. Could be a D_train effect for extremely under-represented languages.

2. **R_code <= 1 for English on any model.** This would undermine the entire NL-code alignment finding, since English should be the easiest case. Severity: fatal for this experiment.

3. **NL-only model (MiniLM) consistently beats code-trained models.** This would suggest R_code measures surface lexical overlap (code keywords appearing in English NL descriptions) rather than semantic alignment. Severity: reinterpretation needed. R_code becomes a lexical artifact, not evidence for Z convergence.

4. **No language ranking pattern (R_code ~uniform across languages).** This would mean D_train does not modulate code alignment. Severity: weak. Removes one claim but does not invalidate the main finding.

5. **Per-language R_code CI includes 1.0 for more than half the cells.** This would mean the aggregate R_code > 1 is driven by a few strong language-model pairs, not a universal phenomenon. Severity: moderate. Weakens generalizability claim.

---

## Strategy 6-R: Minimal Dialect Evidence (Paraphrase Baseline + Two Dialect Pairs)

### Motivation

The original Strategy 6 was designed to show a continuum: d_paraphrase < d_dialect < d_cross_lingual. The audit found five fatal flaws: (a) no dialect data exists in the system, (b) the Operation dataclass constructor silently misassigns positional args, (c) the `dialectal_continuum` metric computes d_inter not d_intra across dialects, (d) p_value=0.0 is hardcoded, and (e) GPT-4o cannot reliably generate rare dialects. This redesign addresses all five while remaining achievable.

### Current Bugs to Fix

#### Bug 1: Operation Dataclass Field Ordering

The `Operation` dataclass is:
```python
@dataclass
class Operation:
    id: str
    category: str
    descriptions: dict[str, str] = field(default_factory=dict)
    dialect_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    has_implicit_criteria: bool = False
```

All 100 operations are constructed as:
```python
Operation("comp_01_sort_asc", "computational",
          {"en": "Sort...", "ko": "...", ...},  # descriptions (positional arg 3)
          ["list", "sort"],                      # <-- THIS IS POSITION 4 = dialect_descriptions!
          False)                                 #     THIS IS POSITION 5 = tags!
```

The 4th positional argument `["list", "sort"]` is assigned to `dialect_descriptions` (type `dict[str, dict[str, str]]`), and `False` is assigned to `tags` (type `list[str]`). Python dataclasses accept any value regardless of annotated type at runtime, so this silently succeeds. The `tags` field receives `False` (a bool), and `dialect_descriptions` receives `["list", "sort"]` (a list). Neither field is ever iterated in the current codebase, so no runtime error occurs.

**Fix required:** Either switch to keyword arguments, reorder the dataclass fields, or add `__post_init__` type checking.

#### Bug 2: `dialectal_continuum` Metric Definition

The function computes:
- `d_cross_lingual`: same op, different languages -- this is correct
- `d_cross_dialect`: same op, same language, different dialects -- this is correct
- `d_within_dialect`: different ops, same language+dialect -- this is d_inter, NOT "within-dialect invariance"

Then it computes `R_cross_lingual = d_within_dialect / d_cross_lingual`. This ratio tests whether d_inter > d_intra at the cross-lingual level, NOT whether the continuum holds.

**What we actually need:** Three distance levels:
- d_paraphrase: same op, same language, different phrasings (same variety)
- d_dialect: same op, same language, different dialect varieties
- d_cross_lingual: same op, different languages

The continuum prediction is: d_paraphrase < d_dialect < d_cross_lingual.

#### Bug 3: `p_value=0.0` Hardcoded

In `predictions.py`, `test_p2_dialectal` returns `p_value=0.0` with a TODO comment. No actual statistical test is performed.

### Design

#### Phase 1: Paraphrase Distance Baseline (No Dialects Needed)

This phase requires zero dialect data. It establishes the FLOOR of linguistic variation.

**Stimuli:** For each of the 30 selected operations (15 computational + 15 judgment) x 5 languages, generate 3 standard-language paraphrases of the existing description.

Example for comp_01_sort_asc, English:
- Original: "Sort the list in ascending order"
- Paraphrase 1: "Arrange the elements from smallest to largest"
- Paraphrase 2: "Put the list items in increasing order"
- Paraphrase 3: "Order the entries of the list from low to high"

**Generation method:** GPT-4o is reliable for standard-language paraphrasing (unlike rare dialects). Prompt:

```
You are generating paraphrases for a computational linguistics experiment.
Given this {language} instruction: "{description}"
Generate 3 paraphrases that:
- Mean exactly the same thing
- Use different vocabulary and sentence structure
- Stay in standard {language} (no dialects, no slang)
- Are roughly the same length as the original
Return as a JSON list of 3 strings.
```

Total: 30 ops x 5 langs x 3 paraphrases = 450 paraphrase descriptions. GPT-4o cost: ~$0.50.

**Human validation:** The Korean and Arabic paraphrases should be spot-checked by a native speaker (the author for Korean; a colleague for Arabic). English, Chinese, Spanish paraphrases can be validated by the author.

```
PSEUDOCODE: compute_paraphrase_baseline(embeddings, paraphrase_embeddings,
                                         operation_ids, languages)

    d_paraphrase_list = []  # same op, same lang, different phrasing
    d_intra_list = []       # same op, different lang (existing d_intra)

    for op_id in operation_ids:
        for lang in languages:
            # d_paraphrase: original vs each paraphrase
            orig_key = f"{op_id}_{lang}"
            if orig_key not in embeddings: continue
            for p_idx in [1, 2, 3]:
                para_key = f"{op_id}_{lang}_para{p_idx}"
                if para_key not in paraphrase_embeddings: continue
                d_paraphrase_list.append(
                    cosine(embeddings[orig_key], paraphrase_embeddings[para_key]))

            # also: paraphrase-paraphrase pairs
            para_vecs = [paraphrase_embeddings[f"{op_id}_{lang}_para{i}"]
                         for i in [1,2,3]
                         if f"{op_id}_{lang}_para{i}" in paraphrase_embeddings]
            for a, b in combinations(para_vecs, 2):
                d_paraphrase_list.append(cosine(a, b))

        # d_intra: same op, different languages (reuse existing metric)
        vecs = [embeddings[f"{op_id}_{lang}"]
                for lang in languages
                if f"{op_id}_{lang}" in embeddings]
        for a, b in combinations(vecs, 2):
            d_intra_list.append(cosine(a, b))

    mean_d_para = mean(d_paraphrase_list)
    mean_d_intra = mean(d_intra_list)

    # Bootstrap CI for the difference
    n_boot = 10000
    boot_diffs = []
    for i in 1..n_boot:
        bp = resample(d_paraphrase_list)
        bi = resample(d_intra_list)
        boot_diffs.append(mean(bi) - mean(bp))
    ci_lo, ci_hi = percentile(boot_diffs, [2.5, 97.5])
    p_value = fraction(boot_diffs <= 0)  # H0: d_intra <= d_paraphrase

    return {
        "d_paraphrase": mean_d_para,
        "d_intra": mean_d_intra,
        "ratio": mean_d_intra / mean_d_para,  # should be > 1
        "difference": mean_d_intra - mean_d_para,
        "ci_95": (ci_lo, ci_hi),
        "p_value": p_value,
        "n_paraphrase_pairs": len(d_paraphrase_list),
        "n_intra_pairs": len(d_intra_list),
    }
```

**Expected result:** d_paraphrase << d_intra. If not, cross-lingual distance is no larger than within-language rephrasing distance, and the entire Z-gap framework has no gap to explain.

#### Phase 2: Two Dialect Pairs

**Korean: Standard Seoul vs Gyeongsang (경상)**
- Gyeongsang dialect has ~10 million speakers, extensive written documentation, and distinctive pitch accent + morphological differences.
- Example: "목록을 오름차순으로 정렬하라" (Standard) vs "목록을 오름차순으로 정렬해라" / "목록 오름차순으로 정렬하이소" (Gyeongsang)
- Source: Author (native Korean speaker) generates descriptions. NOT GPT-4o.
- The author should produce all 30 Gyeongsang descriptions personally, using characteristic features: sentence-final suffix changes (-하이소, -하이가, -데이, etc.), pitch accent markers not available in text, but lexical and morphological differences ARE present.
- Validation: Compare against Gyeongsang dialect corpora (e.g., National Institute of Korean Language regional speech corpus).

**Arabic: MSA vs Egyptian Arabic**
- Egyptian Arabic has ~100 million speakers, extensive written presence (social media, film, literature).
- Example: "رتب القائمة بترتيب تصاعدي" (MSA) vs "رتب الليستة من الصغير للكبير" (Egyptian)
- Source: Native Egyptian Arabic speaker produces 30 descriptions. NOT GPT-4o.
- Validation: Researcher familiar with MSA/Egyptian distinction reviews.

**Total new stimuli:** 30 ops x 2 dialect varieties = 60 descriptions, manually produced.

```
PSEUDOCODE: compute_dialect_distances(embeddings, dialect_embeddings,
                                       paraphrase_embeddings,
                                       operation_ids, languages_with_dialects)

    # Three distance levels
    d_paraphrase_list = []  # from Phase 1
    d_dialect_list = []     # same op, same language, standard vs dialect
    d_cross_lingual_list = []  # same op, different languages

    for op_id in operation_ids:
        # d_dialect: standard vs dialect variety
        for lang, dialect_name in [("ko", "gyeongsang"), ("ar", "egyptian")]:
            standard_key = f"{op_id}_{lang}"
            dialect_key = f"{op_id}_{lang}_{dialect_name}"
            if standard_key in embeddings and dialect_key in dialect_embeddings:
                d_dialect_list.append(
                    cosine(embeddings[standard_key], dialect_embeddings[dialect_key]))

        # d_cross_lingual: reuse from Phase 1
        vecs = [embeddings[f"{op_id}_{lang}"]
                for lang in ["en", "ko", "zh", "ar", "es"]
                if f"{op_id}_{lang}" in embeddings]
        for a, b in combinations(vecs, 2):
            d_cross_lingual_list.append(cosine(a, b))

    # d_paraphrase: from Phase 1

    # Three-level comparison
    mean_d_para = mean(d_paraphrase_list)
    mean_d_dialect = mean(d_dialect_list)
    mean_d_cross = mean(d_cross_lingual_list)

    # Bootstrap CI for each pairwise difference
    # Test 1: d_dialect > d_paraphrase
    boot_diff_1 = bootstrap(d_dialect_list, d_paraphrase_list, n=10000)
    p1 = fraction(boot_diff_1 <= 0)

    # Test 2: d_cross_lingual > d_dialect
    boot_diff_2 = bootstrap(d_cross_lingual_list, d_dialect_list, n=10000)
    p2 = fraction(boot_diff_2 <= 0)

    # Holm-Bonferroni on [p1, p2]

    continuum_holds = mean_d_para < mean_d_dialect < mean_d_cross
    both_significant = corrected_p1 < 0.05 and corrected_p2 < 0.05

    return {
        "d_paraphrase": mean_d_para,
        "d_dialect": mean_d_dialect,
        "d_cross_lingual": mean_d_cross,
        "continuum_holds": continuum_holds,
        "both_significant": both_significant,
        "test_para_vs_dialect": {"diff": mean_d_dialect - mean_d_para, "p": p1},
        "test_dialect_vs_cross": {"diff": mean_d_cross - mean_d_dialect, "p": p2},
    }
```

#### Phase 3: Per-Dialect-Pair Decomposition

Report Korean and Arabic dialect results separately, since they may behave differently:

```
PSEUDOCODE: per_dialect_pair_analysis()

    for lang, dialect in [("ko", "gyeongsang"), ("ar", "egyptian")]:
        # Distances for this pair only
        d_para_this_lang = [paraphrase distances for this language only]
        d_dialect_this_pair = [standard-dialect distances for this pair]
        d_cross_this_lang = [this language vs all others]

        # Check ordering
        ordering_holds = mean(d_para) < mean(d_dialect) < mean(d_cross)

        # Effect size: Cohen's d between dialect and paraphrase distributions
        d_effect = cohens_d(d_dialect_this_pair, d_para_this_lang)

        # Visualization: three overlapping histograms per dialect pair
        plot_three_distributions(d_para_this_lang, d_dialect_this_pair,
                                d_cross_this_lang, title=f"{lang}-{dialect}")
```

### Infrastructure Changes Required

1. **Fix Operation dataclass:** Switch all 100 constructors to keyword arguments, OR reorder fields so `tags` comes before `dialect_descriptions`. The cleanest fix is keyword arguments in every constructor call.

2. **New metric function:** `compute_three_level_distances()` replacing `dialectal_continuum()`. The old function should be deprecated, not deleted (to avoid breaking existing test code).

3. **New stimuli storage:** Add paraphrase descriptions and dialect descriptions to the Operation objects or to a separate data file. Recommend separate JSON files:
   - `data/paraphrases.json`: `{op_id: {lang: [para1, para2, para3]}}`
   - `data/dialects.json`: `{op_id: {lang_dialect: description}}`

4. **Remove p_value=0.0 hardcoding** in `test_p2_dialectal()`.

### Predictions

| Comparison | Predicted ordering | Predicted ratio |
|------------|-------------------|-----------------|
| d_paraphrase vs d_intra | d_para << d_intra | 2-5x (paraphrases are much closer than cross-lingual) |
| d_para vs d_dialect | d_para < d_dialect | 1.2-2x (dialect adds systematic distance) |
| d_dialect vs d_cross_lingual | d_dialect < d_cross | 1.5-3x (cross-lingual > cross-dialectal) |
| Korean gap (std-Gyeongsang) | Moderate | Gyeongsang is mutually intelligible, moderate morphological difference |
| Arabic gap (MSA-Egyptian) | Larger than Korean | MSA-Egyptian has lexical, phonological, and syntactic divergence approaching separate-language level |

**If Arabic MSA-Egyptian distance approaches cross-lingual distances:** This is actually expected and interesting. Egyptian Arabic is arguably a separate language from MSA. If d(MSA, Egyptian) ~ d(Arabic, Spanish), it supports the paper's argument that the dialect-language boundary is continuous, not categorical.

### Feasibility

| Item | Estimate |
|------|----------|
| GPT-4o paraphrase generation | 450 paraphrases, ~$0.50, ~30 min |
| Korean paraphrase validation | 1h (author) |
| Arabic paraphrase validation | 1h (colleague) |
| Gyeongsang dialect descriptions | 2-3h (author, 30 descriptions) |
| Egyptian Arabic descriptions | 2-3h (native speaker, 30 descriptions) |
| Egyptian Arabic speaker recruitment | 1-3 days (university colleague or online) |
| Embedding computation | ~1h per model (trivial) |
| Statistical analysis | ~1h (script execution) |
| **Total calendar time** | **3-5 days** (bottleneck: Arabic speaker) |
| **Total cost** | **< $5** |

### What the Result Means for the Paper

**Best case:** d_paraphrase < d_dialect < d_cross_lingual, with both transitions statistically significant. This provides "illustrative evidence" (explicitly framed as such, not proof) that:
1. Cross-lingual distance is not just rephrasing distance (paraphrase baseline rules this out).
2. Linguistic distance is graded, not binary.
3. The Z-gap is continuous: even within a language, dialectal variation creates measurable distance in embedding space.

**Paper framing:** "We present preliminary evidence from two well-documented dialect pairs (Korean standard/Gyeongsang, Arabic MSA/Egyptian) and a within-language paraphrase baseline. While this does not constitute a comprehensive dialect survey, it establishes the gradient nature of the communicability gap: cross-lingual distance consistently exceeds dialectal distance, which in turn exceeds paraphrase distance (Table X, all p < 0.05 after Holm-Bonferroni correction)."

### Falsification Conditions

1. **d_paraphrase >= d_intra.** Cross-lingual distance is no larger than rephrasing distance. This means the entire Z-gap is an artifact of description variation, not language difference. Severity: FATAL for the paper's framework. If rephrasing an English instruction changes its embedding as much as translating it to Korean, then d_intra measures nothing about cross-lingual convergence.

2. **d_dialect > d_cross_lingual for either pair.** A single dialect pair is more distant than languages. This contradicts the continuum prediction. Severity: moderate for Korean (unexpected), expected for Arabic (MSA-Egyptian may be separate languages). If it happens for Korean standard-Gyeongsang, the embedding model may not handle Korean dialects, or the dialect descriptions may contain errors.

3. **d_dialect ~ d_paraphrase (no significant difference).** Dialects are no more distant than paraphrases. This means either: (a) embedding models collapse dialect differences (OOV tokens mapped to standard), or (b) the dialects chosen are too similar. Severity: moderate. Interpretation: embedding models trained on standard-language data may be insensitive to dialectal variation. This is itself an interesting finding about D_train effects, but it means we cannot use this experiment to demonstrate the continuum.

4. **Ordering holds for one dialect pair but not the other.** Partial support. If Korean shows the continuum but Arabic does not (or vice versa), we report both results honestly and note that the dialect-language boundary varies by typological context. Not fatal, but weakens the generalizability claim.

---

## Implementation Priority

| Priority | Task | Experiment | Effort |
|----------|------|------------|--------|
| 1 | Fix Operation dataclass bug | Both | 1h |
| 2 | Add NomicCodeEmbedder + CodeSageEmbedder classes | Strategy D | 2h |
| 3 | Implement per-language R_code computation | Strategy D | 2h |
| 4 | Run Strategy D (embed + compute + significance) | Strategy D | 1 day |
| 5 | Generate paraphrases (GPT-4o) | Strategy 6-R | 1h |
| 6 | Write Gyeongsang descriptions (author) | Strategy 6-R | 3h |
| 7 | Source Egyptian Arabic descriptions | Strategy 6-R | 1-3 days |
| 8 | Implement three-level metric | Strategy 6-R | 2h |
| 9 | Run Strategy 6-R (embed + compute + significance) | Strategy 6-R | 1h |
| 10 | Write paper sections | Both | 1 day |

**Total: ~5-7 days, with Strategy D completable in 2 days independently of Strategy 6-R.**

---

## Relationship Between the Two Experiments

These experiments serve complementary roles:

- **Strategy D** (NL-code alignment) is the paper's STRONGEST evidence and should be prioritized. It demonstrates convergence at the execution level across languages and models.

- **Strategy 6-R** (dialect evidence) is ILLUSTRATIVE, providing gradient evidence for the communicability gap. It is not essential for the paper's core argument but enriches the discussion.

If time is limited, Strategy D alone is sufficient for the paper. Strategy 6-R adds depth but can be deferred to the camera-ready version or a follow-up.

The two experiments connect through the paper's central equation: **convergence (measured by R_code across languages) != communicability (measured by d_intra across languages, modulated by dialect and paraphrase distances)**. Strategy D shows the left side; Strategy 6-R shows the right side is graded, not binary.
