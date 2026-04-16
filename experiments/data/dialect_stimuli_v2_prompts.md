# dialect_stimuli_v2.json — Generation Metadata

**Date:** 2026-04-11
**Model:** Claude Opus 4.6 (in-session, not via external API)
**Generator script:** `experiments/scripts/gen_dialect_stimuli_v2.py`
**Source design doc:** `experiments/EXPERIMENT_3H_SESSION.md` §Task 1
**Stop-rule + OOV audit:** `experiments/EXPERIMENT_3H_SESSION.md` §Task 2 +
  reviewer session refinements applied prior to generation.

---

## Coverage

| Category              | Count | Source                                              |
|-----------------------|------:|-----------------------------------------------------|
| `original.*` (5 lang) |   150 | **Copied** from `src/stimuli.py` (no regeneration)  |
| `paraphrases.en`      |    90 | **Copied** from `data/dialect_stimuli.json` (v1)    |
| `paraphrases.ko`      |    90 | Opus 4.6 in-session                                 |
| `paraphrases.zh`      |    90 | Opus 4.6 in-session                                 |
| `paraphrases.ar`      |    90 | Opus 4.6 in-session                                 |
| `paraphrases.es`      |    90 | Opus 4.6 in-session                                 |
| `dialects.ko_gyeongsang` | 30 | Opus 4.6 in-session                                 |
| `dialects.ar_egyptian`   | 30 | Opus 4.6 in-session                                 |
| **Total new items**   |   420 |                                                     |

Originals copied to preserve embedding cache coherence (cache keys hash the
exact text).

---

## Prompt templates (applied in-session, not via API)

### (a) Cross-lingual paraphrases — per language ∈ {ko, zh, ar, es}

```
For each of these 30 operation descriptions in {LANG}, produce 3 paraphrases
that:
- Preserve exact meaning
- Use different vocabulary and sentence structure from the original and
  from each other
- Stay in STANDARD {LANG} (no dialects, no slang)
- Are roughly the same length as the original
- Preserve technical terms (list, sort, string, number, etc. — in the
  language's standard technical vocabulary)

Return strict JSON: {"op_id": ["para1", "para2", "para3"], ...}
```

### (b) Korean Gyeongsang dialect

```
Generate Gyeongsang Korean (경상 방언, Busan/Daegu register) versions of the
30 standard Korean operation descriptions. For each:

- Preserve exact meaning
- Use written-form Gyeongsang markers:
  * sentence-final: -하이소 (polite imperative), -뿌라 (perfective imperative),
    -봐라 (suggestive imperative), -이데이 (informative declarative)
  * lexical substitutions where natural: 디비다 (뒤집다), 치아다 (치우다),
    바까다 (바꾸다), 맨들다 (만들다), 짤라다 (자르다), 시 (세)
  * informal address particles: 한 놈 한 놈, 좀
- Do NOT use pitch accent notation (phonological only)
- Preserve technical terms (정렬, 리스트, 목록, 문자열, 평균, 합, 길이)
- Length similar to original

Return strict JSON: {"op_id": "gyeongsang text", ...}
```

### (c) Egyptian (Cairene) Arabic dialect

```
Generate Egyptian Arabic (اللهجة المصرية، القاهرية) versions of the 30 MSA
operation descriptions. For each:

- Preserve exact meaning
- Use Egyptian features:
  * lexical substitution: قائمة → ليستة, رقم → رقم (kept)
  * verbs: هات (bring/get), طلّع (take out), شيل (remove), الزق (stick),
    خلّي (make/let), حطّ (put), شوف (see/look), اختار (choose)
  * demonstratives: ده، دي، دول
  * particles: بتاع (of/belonging), علشان (so that), كده, ولا لأ
  * conjunction: اللي (who/that)
- Orthography: ج kept as-is (spoken as /g/ but written ج)
- Preserve technical terms that are Egyptian loanwords (ليستة for list)
- Length similar to original

Return strict JSON: {"op_id": "egyptian text", ...}
```

---

## Known limitations (must be reflected in paper § Limitations)

1. **In-session LLM generation, not native-speaker authored.** All paraphrases
   (ko/zh/ar/es) and both dialects were produced by Opus 4.6. No back-translation
   round-trip, no professional translator review.

2. **Korean Gyeongsang validation: PENDING user spot-check.**
   Author (Korean-speaking researcher) has spot-check ability for 경상 방언
   but full 30-item audit not yet performed at time of generation. Session
   design (`EXPERIMENT_3H_SESSION.md` §Task 1 Validation) calls for 5-item
   spot-check before Task 2 execution.

3. **Egyptian Arabic validation: NOT POSSIBLE in this session.**
   No native Arabic speaker available. Marked as **camera-ready blocker** —
   must be audited by a native Cairene speaker before publication. This is
   an acknowledged risk in AUDIT_P2_STRATEGIES.md §Strategy 6 ("방언/언어
   분류 논쟁: 광동어-보통화, 이집트 아랍어-MSA는 언어학적으로 별개 언어
   수준"). The paper §Limitations must state this explicitly.

4. **Gyeongsang in written form is attenuated.** 경상 방언 characteristic
   features are primarily phonological/intonational; written-form differences
   from standard Korean are relatively small. The generated cells use:
   (a) sentence-ending replacements (-하이소/-뿌라/-봐라), (b) selected lexical
   substitutions (디비다, 치아다, 바까다, etc.), (c) informal particles (한 놈,
   젤). This is the best-available approximation for embedding-space analysis,
   not a linguistic transcription standard.

5. **Egyptian lexicon choices may be over-selected.** Where both Egyptian and
   MSA are used interchangeably by Cairene speakers, generator preferred the
   more distinctly Egyptian form (ليستة over قائمة, etc.) to maximize
   orthographic signal. Native speakers may find some substitutions unnatural
   in a technical instruction context.

6. **Out-of-scope dialects excluded per AUDIT.** Jeju, Min-nan, Gulf Arabic
   were explicitly excluded because AUDIT_P2_STRATEGIES.md §Strategy 6
   flagged them as LLM-untrustworthy.

---

## Pre-existing audits / warnings applied

- **AUDIT §Strategy 6** ("임베딩 모델 방언 미학습 → OOV 토큰 → 노이즈"):
  Task 2 must run a **tokenizer OOV sanity check** (`tokens/char` ratio vs
  standard ko/ar baseline, threshold 1.5×) before computing dialect distances.
  Cells that fail the OOV check are dropped from the main analysis and
  reported in appendix.

- **Stop rule for Strategy 6-R** (reviewer session agreement, 2026-04-11):
  After bootstrap, if 95% CI lower bound of `(d_cross - d_dial)` ≤ 0, OR
  equivalently `p(d_cross ≤ d_dial) > 0.05`, then Strategy 6-R is **dropped**
  from the paper entirely. No reframing as "continuous boundary" is allowed.

---

## Provenance integrity

- Originals and English paraphrases are byte-identical copies of
  `src/stimuli.py` (accessed via `get_all_operations()`) and v1
  `data/dialect_stimuli.json` respectively. Verified by assertion in
  `gen_dialect_stimuli_v2.py` (no editing, no reformatting).

- New entries (ko/zh/ar/es paraphrases + both dialects) are literal dict
  definitions in the generator script, committed to the repo for full
  regeneration reproducibility.

- v1 file `data/dialect_stimuli.json` remains untouched. Backup of the
  (summary-format) v1 results saved as
  `results/strategy_6r_dialect_results_v1_english.json` during Task 0
  preflight.
