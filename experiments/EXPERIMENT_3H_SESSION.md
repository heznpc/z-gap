# 3-Hour Session Plan: Strategy D Completion + Strategy 6-R Redo

> 2026-04-11. Drafted for a fresh Claude Opus 4.6 session.
> References: `EXPERIMENT_DESIGNS.md` (design spec), `AUDIT_P2_STRATEGIES.md` (audit of original strategies), `TODO.md` (ARR 5/25 deadline).

## Assumption

Claude Opus 4.6 is available without practical rate/cost limits for this session. This unblocks the stimulus-generation bottlenecks that forced the original EXPERIMENT_DESIGNS.md to estimate 3-5 days (native speaker recruitment). It does **not** replace native-speaker validation for camera-ready, but the LLM-generated stimuli are acceptable for the ARR submission if clearly framed as "LLM-generated probes, pending native-speaker audit."

## Goal

Fill two concrete gaps in the existing experiments so that the EMNLP submission's two strongest empirical sections (NL-code alignment + dialect continuum) are complete enough to write up by 5/25.

---

## Current State (verified 2026-04-11)

### Strategy D — EXECUTED, one model missing

- **File:** `results/strategy_d_code_alignment.json`
- **Models run:** UniXcoder, MiniLM-L12, nomic-embed-text-v1.5, E5-large (4 × 5 langs = 20 cells)
- **Result:** all 20 cells R_code > 1, 19/20 significant after Holm-Bonferroni (ar/ko cells for UniXcoder are marginal)
- **Gap:** `CodeSage-Large-v2` from the original design (`EXPERIMENT_DESIGNS.md §Strategy D / Models`) was never added. It is the only modern **code-trained** model in the planned set; without it the paper cannot make the "code-trained > NL-only" claim. Current data actually shows MiniLM (NL-only) often beats UniXcoder (code-trained), which is the falsification-condition #3 from the design. CodeSage is the deciding model.

### Strategy 6-R — EXECUTED, data is degenerate

- **File:** `results/strategy_6r_dialect_results.json`
- **Reported ordering:** `d_dial < d_para < d_cross` (continuum "fails")
- **Root cause (not a bug):** `data/dialect_stimuli.json` stores English `british`/`indian` variants instead of the Korean Gyeongsang + Egyptian Arabic dialects specified in `EXPERIMENT_DESIGNS.md §Strategy 6-R / Phase 2`. The "british" entries for `comp_01_sort_asc` through `comp_15_length` are **identical** to the original English text. Empirically `d_dial ≈ 0` is therefore correct for the data — but the data is not testing dialects.
- **Fix:** regenerate `dialect_stimuli.json` with real Korean Gyeongsang + Egyptian Arabic dialect variants (via Opus 4.6 as dialect oracle), and add cross-lingual paraphrases so the baseline is computed per-language, not English-only.

---

## Task Budget

| # | Task | Est. | Hard dependency |
|---|------|------|-----------------|
| 1 | Regenerate `dialect_stimuli_v2.json` via Opus 4.6 | 40 min | none |
| 2 | Rewrite `run_strategy_6r_dialect.py` for v2 schema + rerun | 30 min | 1 |
| 3 | Implement `CodeSageEmbedder` class | 30 min | none |
| 4 | Add CodeSage to `run_strategy_d_code_alignment.py` + rerun | 30 min | 3 |
| 5 | Synthesis: cross-model comparison figure + results digest | 20 min | 2, 4 |
| 6 | Commit + sanity check | 10 min | 5 |
| **Total** | | **~160 min** | |

Tasks 1+3 are independent — run them in parallel (background Bash for Opus generation while you implement CodeSage).

---

## Task 1 — Regenerate `dialect_stimuli_v2.json`

### Target schema

```json
{
  "comp_01_sort_asc": {
    "original_en": "Sort the list in ascending order",
    "original_ko": "목록을 오름차순으로 정렬하라",
    "original_zh": "按升序对列表进行排序",
    "original_ar": "رتب القائمة بترتيب تصاعدي",
    "original_es": "Ordena la lista en orden ascendente",

    "paraphrases": {
      "en": ["...", "...", "..."],
      "ko": ["...", "...", "..."],
      "zh": ["...", "...", "..."],
      "ar": ["...", "...", "..."],
      "es": ["...", "...", "..."]
    },

    "dialects": {
      "ko_gyeongsang": "목록 오름차순으로 정렬하이소",
      "ar_egyptian":   "رتب الليستة من الصغير للكبير"
    }
  },
  ...
}
```

30 operations total (15 comp + 15 judg — same IDs as current file). The `original_*` fields should be **copied** from `src/stimuli.py` operations, not regenerated, so they stay aligned with the cached embeddings.

### Generation commands

Use three Opus 4.6 calls, one for each generation category. Each call returns strict JSON.

**(a) Cross-lingual paraphrases** — for the 4 non-English languages. English paraphrases already exist in `dialect_stimuli.json` — copy those.

Prompt shell (run once per language ∈ {ko, zh, ar, es}):

```
You are generating paraphrases for a computational linguistics experiment on 
embedding-space distances. I will give you 30 operation descriptions in {LANG}. 
For each, produce 3 paraphrases that:
- Preserve exact meaning
- Use different vocabulary and sentence structure from the original and from 
  each other
- Stay in STANDARD {LANG} (no dialects, no slang)
- Are roughly the same length as the original
- Do NOT include the operation ID or any English

Return a single JSON object keyed by operation ID:
{"comp_01_sort_asc": ["para1", "para2", "para3"], ...}

Operations:
<paste 30 operations with IDs and {LANG} descriptions from src/stimuli.py>
```

**(b) Korean Gyeongsang dialect** — single Opus call:

```
You are a native speaker of Gyeongsang Korean (경상 방언, Busan/Daegu area). 
I have 30 standard Korean operation descriptions. Produce a Gyeongsang version 
for each that:

- Preserves exact meaning
- Uses characteristic Gyeongsang features:
  * sentence-final suffixes: -하이소, -하이가, -이데이, -제, -능교, -나
  * lexical substitutions where natural (e.g., 어떡해 → 우야노, 그래 → 기래)
  * copula contractions (-이다 → -이라, -이야)
  * declarative intonation markers where orthographically possible
- Does NOT use pitch accent notation (phonological only)
- Preserves technical terms (정렬, 리스트, 문자열 etc.)
- Is roughly the same length as the original

Return strict JSON: {"comp_01_sort_asc": "gyeongsang text", ...}

Operations (standard Korean):
<paste 30 operations with IDs and ko descriptions>
```

**(c) Egyptian Arabic dialect** — single Opus call:

```
You are a native speaker of Cairene Egyptian Arabic (اللهجة المصرية، القاهرية). 
I have 30 Modern Standard Arabic operation descriptions. Produce an Egyptian 
version for each that:

- Preserves exact meaning
- Uses characteristic Egyptian features:
  * ج → g (pronounced 'gīm', but keep orthography ج)
  * present tense prefix بـ (e.g., بيرتّب instead of يرتّب)
  * demonstratives: ده، دي، دول
  * particles: بتاع، كده، أوي
  * Egyptian lexicon where it differs from MSA (e.g., قائمة → ليستة optional)
- Preserves technical terms that are loanwords in Egyptian usage
- Is roughly the same length as the original

Return strict JSON: {"comp_01_sort_asc": "egyptian text", ...}

Operations (MSA):
<paste 30 operations with IDs and ar descriptions>
```

### Validation (5 min, author)

- Spot-check 5 Korean Gyeongsang entries: are the sentence-final suffixes grammatical? Do the lexical substitutions make sense?
- Spot-check 5 Korean paraphrases: are they actually different from the originals?
- Egyptian Arabic: no native-speaker validation available in this session. Document as limitation.

### Output

- `data/dialect_stimuli_v2.json` (new file; keep the old `dialect_stimuli.json` untouched for backwards compat)
- `data/dialect_stimuli_v2_prompts.md` — copy of the three prompts actually used, for reproducibility

---

## Task 2 — Rerun Strategy 6-R on v2 data

### File changes

1. `run_strategy_6r_dialect.py`:
   - Load `dialect_stimuli_v2.json` instead of `dialect_stimuli.json`
   - Replace the `DIALECT_PAIRS` logic with per-language dialect distances:
     - `d_dialect_ko` = cosine(`ko` embedding, `ko_gyeongsang` embedding), one per op
     - `d_dialect_ar` = cosine(`ar` embedding, `ar_egyptian` embedding), one per op
     - `d_dialect_combined` = concat of the two (n=60)
   - Compute `d_paraphrase` **per language** from the new v2 paraphrases (5 langs × 30 ops × 3 paraphrases → up to 450 pairs; also include paraphrase-paraphrase pairs within language)
   - Keep `d_cross_lingual` computation as is (uses main stimuli, not dialect file)

2. Add `BAAI/bge-m3` and `intfloat/multilingual-e5-large` stay; add `Qwen/Qwen3-Embedding-8B` only if GPU memory permits (skip otherwise — 8B FP16 needs ~16GB VRAM).

3. Bootstrap tests unchanged (10k resamples, same `bootstrap_diff` function). Run Holm-Bonferroni on the two p-values per model.

### New metric to add

Per-language continuum check — does the ordering hold within each language?

```
for lang in [en, ko, zh, ar, es]:
    d_para_lang = [paraphrase distances for this language only]
    d_dial_lang = [dialect distance for this language — only ko/ar have it]
    d_cross_lang = [cross-lingual distances from this language to others]

    ordering_holds[lang] = mean(d_para_lang) < mean(d_dial_lang or d_cross_lang) 
                                              < mean(d_cross_lang)
```

### Acceptance criteria

- Continuum holds (`d_para < d_dial < d_cross`) on **at least 2 of 3** embedding models for the **combined** (ko+ar) dialect metric, after Holm-Bonferroni correction.
- If the ordering inverts: do NOT silently report success. Inspect the raw distances and escalate in the results digest (Task 5).

### Output

- `results/strategy_6r_dialect_results.json` (overwritten; keep the old file as `strategy_6r_dialect_results_v1_english.json` for the appendix)
- `results/figures/strategy_6r_dialect_continuum.png` (updated — 3 subplots, one per model)
- New: `results/figures/strategy_6r_per_language.png` — per-language bar chart showing paraphrase vs dialect (where applicable) vs cross-lingual

---

## Task 3 — `CodeSageEmbedder` class

### Why not sentence-transformers

CodeSage-Large-v2 (microsoft/codesage-large-v2, 1.3B, 1024d) is not wrapped for sentence-transformers out of the box. It requires `transformers.AutoModel` + mean pooling + L2 normalization.

### New class — add to `src/embeddings.py`

```python
class CodeSageEmbedder(EmbeddingModel):
    """CodeSage-Large-v2 (Microsoft ICML 2024, bimodal NL+code)."""

    def __init__(self, model_name: str = "codesage/codesage-large-v2", 
                 device: str | None = None, batch_size: int = 8):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True).eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(device)
        self._batch_size = batch_size
        # Probe dimension with a dummy forward pass
        with torch.no_grad():
            probe = self._tokenizer("x", return_tensors="pt").to(device)
            out = self._model(**probe)
            self._dim = out.last_hidden_state.shape[-1]

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        from tqdm import tqdm
        vecs = []
        for i in tqdm(range(0, len(texts), self._batch_size), 
                      desc=f"CodeSage {len(texts)}"):
            batch = texts[i:i + self._batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True, 
                max_length=512, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                out = self._model(**enc)
            # Mean pool over tokens, masking padding
            hidden = out.last_hidden_state  # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            pooled = summed / counts
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().numpy())
        return np.concatenate(vecs, axis=0).astype(np.float32)

    @property
    def name(self) -> str:
        return f"hf_{self._model_name.split('/')[-1]}"

    @property
    def dimension(self) -> int:
        return self._dim
```

### Cache key compatibility

`EmbeddingCache` uses the embedder's `name` property. `hf_codesage-large-v2` will have its own cache bucket and won't collide with sentence-transformers models.

### Fallback if loading fails

CodeSage-Large is 1.3B params — on CPU embedding 300 texts (250 NL × 5 langs + 50 code) takes roughly 5-10 min; on a consumer GPU (8GB) it fits in FP16 and takes ~1 min. If OOM or loading fails:

1. Fall back to **`Salesforce/codesage-base`** (356M, same architecture, same encode pattern, smaller footprint).
2. If even that fails, fall back to **`jinaai/jina-embeddings-v2-base-code`** (161M, code-specialized, sentence-transformers compatible — can reuse `SentenceTransformerEmbedder`).

Record which model actually ran in the results JSON so the paper table shows the truth.

---

## Task 4 — Add CodeSage to Strategy D

### File changes

`run_strategy_d_code_alignment.py`:

```python
from src.embeddings import (SentenceTransformerEmbedder, CodeSageEmbedder, 
                            EmbeddingCache)

MODELS = [
    ("microsoft/unixcoder-base", "UniXcoder (code)", "st", {}),
    ("paraphrase-multilingual-MiniLM-L12-v2", "MiniLM-L12 (NL)", "st", {}),
    ("nomic-ai/nomic-embed-text-v1.5", "Nomic v1.5 (NL+code)", "st", 
     {"trust_remote_code": True}),
    ("intfloat/multilingual-e5-large", "E5-large (NL)", "st", {}),
    ("codesage/codesage-large-v2", "CodeSage-Large (code)", "codesage", {}),
]

def _build_embedder(kind: str, model_name: str, kwargs: dict):
    if kind == "st":
        return SentenceTransformerEmbedder(model_name, **kwargs)
    elif kind == "codesage":
        return CodeSageEmbedder(model_name, **kwargs)
    raise ValueError(kind)
```

Replace the `SentenceTransformerEmbedder(model_name, **kwargs)` line in `run_model()` with `_build_embedder(kind, model_name, kwargs)`. Nothing else changes — `compute_per_language_R_code`, Holm-Bonferroni, figure code all stay.

### Only the new model needs fresh embeddings

UniXcoder, MiniLM, Nomic v1.5, E5-large are all cached in `results/embeddings/`. `EmbeddingCache.get_or_compute` will skip them. CodeSage will be the only model actually embedding 300 texts.

### Update Holm-Bonferroni

After adding CodeSage, there are 25 per-language p-values (5 models × 5 langs). Holm-Bonferroni correction scales automatically in `holm_bonferroni()`. Double-check the printed summary includes the 5th row.

### Acceptance criteria

- CodeSage produces R_code > 1 for all 5 languages after Holm-Bonferroni correction.
- If CodeSage R_code is **lower** than MiniLM's across most languages: report honestly and update the paper's narrative. This would be evidence against "code-trained > NL-only" and would strengthen the paper's "convergence ≠ communicability" framing (R_code is a general property, not a code-training artifact).
- If CodeSage fails to load: fall back per Task 3, note it in the results JSON.

### Output

- `results/strategy_d_code_alignment.json` (overwritten)
- `results/strategy_d_code_alignment_v1_4models.json` (copy of current file)
- `results/figures/strategy_d_rcode_heatmap.png` (updated to 5×5)
- New: `results/figures/strategy_d_codetrained_vs_nlonly.png` — grouped bar chart: x=language, 3 bars = [code-trained mean (UniXcoder+CodeSage), NL+code mean (Nomic), NL-only mean (MiniLM+E5-large)]. This directly visualizes the claim.

---

## Task 5 — Synthesis digest

Create `results/session_2026-04-11_digest.md` with:

1. **Strategy D update (5 models)** — 5×5 R_code table, Holm-Bonferroni p-values, cohen's d.
2. **Code-trained vs NL-only finding** — one sentence stating which group won and by how much.
3. **Strategy 6-R v2 results** — 3 models × continuum test, per-language breakdown, which models support the ordering.
4. **Limitations** — LLM-generated Gyeongsang/Egyptian Arabic, no native-speaker validation, paraphrases were also LLM-generated.
5. **Implications for paper** — which sentences in `paper/main_emnlp.tex` need editing (cite line numbers).

Keep under 500 words. This is for the author's own tracking, not for the paper.

---

## Task 6 — Commit + sanity check

Per user's memory: no `Co-Authored-By: Claude` in research repo commits.

Two commits (separate so they can be reverted independently):

1. `add Strategy D CodeSage-Large-v2 model` — new embedder + script update + new results + new figure
2. `rerun Strategy 6-R with Korean Gyeongsang and Egyptian Arabic dialects` — new stimuli file + script rewrite + new results + new figure

Do **not** delete the v1 files — keep them as appendix reference.

Sanity check before committing:
- `results/strategy_d_code_alignment.json` has 5 entries, each with 5 per-language cells + aggregate
- `results/strategy_6r_dialect_results.json` has the new schema (ko_gyeongsang, ar_egyptian keys)
- All referenced figures exist in `results/figures/`
- `pytest experiments/` (if tests exist) passes

---

## Risks and known issues

| Risk | Mitigation |
|------|------------|
| Opus 4.6-generated Gyeongsang contains errors | Author spot-checks 5 entries; document as limitation; mark v1 (English) as control in appendix |
| Opus 4.6-generated Egyptian Arabic cannot be author-validated | Flag as "pending native-speaker audit"; keep v1 control as backup |
| CodeSage OOM or trust_remote_code fails | Fallback chain per Task 3 |
| 5-model Holm-Bonferroni makes borderline cells non-significant | Acceptable — report uncorrected and corrected p-values both |
| `d_dial > d_cross_lingual` for Arabic (Egyptian may be functionally separate language) | Expected; mention that dialect-language boundary is continuous, which **supports** the paper's broader claim |

## Out of scope for this session

- Adding Nomic Embed Code 7B (requires 14GB VRAM + separate embedder class) → future session
- OmniSONAR evaluation (from TODO.md) → future session
- Generating Jeju / Min-nan / Gulf Arabic dialects — these remain LLM-untrustworthy per AUDIT_P2_STRATEGIES.md
- Paper text edits — do those in a follow-up session after reviewing the digest

---

## Quickstart for the new session

```bash
cd /Users/ren/IdeaProjects/Paper/z-gap
source experiments/.venv/bin/activate  # or create if missing: python -m venv experiments/.venv
pip install -r experiments/requirements.txt

# Read these three files first:
# - experiments/EXPERIMENT_3H_SESSION.md  (this file)
# - experiments/EXPERIMENT_DESIGNS.md     (full design with falsification conditions)
# - experiments/AUDIT_P2_STRATEGIES.md    (what failed in the original 6 strategies)

# Then start Task 1 (dialect generation) and Task 3 (CodeSage class) in parallel.
```
