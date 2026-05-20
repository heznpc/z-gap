# Research Decisions Log

Records non-obvious choices with rationale. Append-only; don't rewrite history.

Format: `## YYYY-MM-DD -- <short title>` with **Context**, **Decision**, **Why**.

---

## 2026-04-19 -- Repository restructure to DDD-style layout + venue separation

**Context**: paper/ contained `main.tex`, `main_colm.tex`, `main_emnlp.tex` (three parallel versions of the same paper with content drift), plus venue-specific style files (acl.sty, colm2026_conference.sty, etc.) mixed with general style (natbib.sty, fancyhdr.sty). Root had AUDIT_P2_STRATEGIES.md, TODO.md, review.md.

**Decision**:
  - paper/ now holds only the canonical manuscript + its immediate build dependencies (acl.sty, natbib.sty, fancyhdr.sty stay at paper/ root because LaTeX needs them next to main.tex).
  - submissions/colm-2026/ and submissions/emnlp-2026/ are venue snapshots: each has its own main.tex (frozen as of this commit), venue-specific style files, and a NOTES.md.
  - All venue submission tex files now reference `../../paper/references.bib` for bibliography (single source of truth for citations).
  - AUDIT_P2_STRATEGIES.md -> planning/drafts/audit_p2_strategies.md (lowercase + underscores).
  - experiments/EXPERIMENT_*.md -> experiments/design/ (three design docs grouped).

**Why**: The three-tex problem was the canonical example of venue-optimization bleeding into content. Separating submissions/<venue>/main.tex as frozen venue snapshots makes the drift explicit and recoverable. Venue-specific style files live with their submissions; paper/ stays compilable standalone.

---

## 2026-04-19 -- experiments/scripts vs experiments/src kept separate

**Context**: The portfolio template puts runnable code in experiments/src/. z-gap already had `scripts/` (runnable entry points) and `src/` (library modules) as a Python research project convention.

**Decision**: Preserve the distinction. `src/` holds importable library (analysis.py, embeddings.py, metrics.py, ...); `scripts/` holds runnable entry points (run_*.py) that import from src/.

**Why**: This is a mature Python project structure. Collapsing scripts/ into src/ would hide a real architectural boundary (library vs runnable). The portfolio template accepts this as a valid expansion.

---

## 2026-04-19 -- Content drift between paper/main.tex and submissions/*/main.tex

**Context**: paper/main.tex, submissions/colm-2026/main.tex, and submissions/emnlp-2026/main.tex have meaningful content drift (different abstracts, different framing, different "contributions" lists).

**Decision**: Freeze the submission copies as venue-specific snapshots. paper/main.tex is the preferred version going forward. Do not auto-sync.

**Why**: The divergence reflects real editorial work (COLM-specific framing, EMNLP review responses). Collapsing to one auto-synced source would erase those decisions; keeping them frozen lets the user reconcile manually.

---

## 2026-05-21 -- Pre-experiment research review for Strategy D cross-model extension

**Context**: Before extending `run_strategy_d_code_alignment.py` from 4 to 7 embedding models, a 9-dimension review surfaced three Critical issues and three Major issues that needed to be reflected in `paper/main.tex` (limitations + main text) before re-running the experiment, so the experiment is not invalidated by a known reviewer-side weakness discovered after the fact.

**Decisions**:

  - **C1 (Pretraining contamination)**: The 50 computational stimuli are all Python stdlib idioms (`sorted`, `max`, `len`, ...). Embedding models almost certainly saw these exact NL↔code pairings during pretraining. Added a `Pretraining contamination caveat` paragraph to `paper/main.tex` §5.5 and a corresponding bullet to Limitations. `R_code > 1` is now interpreted as "at least as strong as pretraining co-occurrence would predict," not as independent evidence for Z_sem convergence. Decisive separation deferred to tier-2/tier-3 OOD stimuli (already in `experiments/data/stimuli/` but unanalyzed).

  - **C2 (Random matching baseline)**: The permutation test (n=10,000) is now explicitly framed in the main text as the random-matching baseline with null R ≈ 1. The shuffled-pairing R distribution mean will be exported to `results/strategy_d_code_alignment.json` per language for transparency.

  - **C3 (HuggingFace model revision pin)**: `sentence-transformers` pulls the model card's `main` branch at load time. For this pilot we accept the floating-`main` risk and rely on the existing `EmbeddingCache` (`.npz` keyed by `(model_name, text_hash)`) to freeze the actual computed embeddings. Pinning via `revision=` is left as a Minor TODO once the cross-model matrix lands.

  - **M1 (Trivial stimuli)**: Added `Stimulus complexity` paragraph to Limitations stating that conclusions apply to stdlib-idiom-level operations only; tier-2/tier-3 stimuli exist but are not yet analyzed.

  - **M2 (Translation provenance)**: Added `Translation provenance` paragraph stating that translations were produced by the first author with LLM-assisted draft and bilingual review; no formal IAA. This is acknowledged as a recognized weakness for cross-lingual claims.

  - **M3 (Model robustness wrap)**: `run_strategy_d_code_alignment.py` model loop wrapped in `try/except` per model so a single OOM/network/trust-remote-code failure does not abort the full 7-model sweep.

  - **M4 (Prior art)**: Web search confirmed no per-language × per-model NL-code alignment matrix exists in the cross-lingual representation literature as of 2026-05. OmniSONAR (Meta, 2026.03; arXiv:2603.16606) is the closest concurrent work but operates at the model-level multi-modal embedding axis, not the cross-lingual gradient within a fixed code-stimulus set. A "to our knowledge, first" qualifier was added to §5.5.

  - **M5 (P3 multi-model probing scope)**: Deferred. This session's experiment is NL-code alignment only (Strategy D scope). P3 cross-lingual probing on the 7-model set is a separate follow-up PR.

  - **M6 (Codestral Embed)**: Excluded for this session. `.env` has no `MISTRAL_API_KEY`, and the user constrained the session to Claude Code-accessible models. Sentence-transformers / open-source HF only.

**Why**: The pre-experiment review caught contamination and baseline-framing issues that, if discovered after results were reported, would have required a paper revision plus a fresh experiment. Catching them before the cross-model extension lets a single PR carry the corrected framing and the new evidence simultaneously.

---

## 2026-05-21 -- Strategy D 7-model results + einops dependency fix

**Context**: After the pre-experiment review PR (#3) merged, the Strategy D extension ran on 7 models. First run: 6/7 succeeded; Nomic v1.5 failed with `ImportError: einops` because its `trust_remote_code` module imports `einops` lazily and the package was not in `requirements.txt`. The M3 try/except wrap correctly isolated the failure so the other 6 models completed cleanly.

**Decisions**:

  - Added `einops>=0.7` to `experiments/requirements.txt` and `pyproject.toml` dependencies. The package is needed only by Nomic's remote-code path; pinning loosely (`>=0.7`) is sufficient because the API has been stable since 0.6.
  - Re-ran Strategy D with einops installed. All 7 models succeeded. Final matrix: **35/35 cells with $R_{\text{code}} > 1$ and $p < 0.05$ after Holm-Bonferroni**. Permutation-null mean ∈ [1.000, 1.005] across all cells (C2 baseline framing empirically confirmed).
  - Paper §5.5 Table updated 4-row → 7-row. Body text revised from "20/20 cells" to "35/35 cells", added "Third pattern" paragraph on the E5 family's partial scale-convergence ($1.13 \to 1.14 \to 1.20$ at $384/768/1024$d).
  - The pretraining contamination caveat (C1) added in PR #3 stays unchanged — adding more models does not address contamination, only cross-model robustness.

**Why**: The 7-model extension was the empirical contribution this session aimed to land. Catching einops as a soft-dep blocker (rather than as a paper-level claim error) preserved the cross-model robustness claim. The E5-family scale-convergence finding is a side effect of the extension that strengthens P1 in a way the previous mixed-family P1 test (MiniLM/mpnet/E5-large) could not.

---

## 2026-05-21 -- Strategy E: multi-model P3 cross-lingual probing (closes M5)

**Context**: The 2026-05-21 pre-experiment review classified M5 (P3 multi-model probing) as a deferred Major: the paper's original P3 claim (90% category, 86% operation transfer) was supported by linear probes trained on MiniLM-L12 embeddings only. This left the "Z_sem stratifies and is cross-lingually accessible" claim resting on a single model.

**Decisions**:

  - Added `experiments/scripts/run_strategy_e_multimodel_probing.py` running P3 (category 2-way + operation 100-way LogisticRegression probes) on the same 7-model set as Strategy D. Per-cell statistics: accuracy + one-sided binomial test against chance. Includes run_meta block, per-model try/except, and heatmap figure outputs in the Strategy D pattern.

  - Result (paper §5.5 P3 Results table now 7 rows): P3 is **supported in multilingual NL models but is model-class dependent**. Multilingual NL family (MiniLM, E5 small/base/large, BGE-M3): category transfer 0.86–0.99, operation transfer 0.86–0.98. Code-trained (UniXcoder: 0.67 / 0.18) and mixed NL+code (Nomic: 0.62 / 0.23) show near-perfect English training but collapse on cross-lingual transfer.

  - Side finding (P1 echo within P3): the E5 family alone shows clean scale-convergence in operation transfer: 0.89 (384d) → 0.96 (768d) → 0.98 (1024d), under fixed architecture and training recipe. This mirrors the NL-code alignment scale-convergence reported in the Strategy D table.

  - Paper interpretation refined: cross-lingual Z_sem separability is a property of the multilingual NL training distribution, not an intrinsic property of every embedding space with $R_{\text{code}} > 1$. The original P3 claim is preserved for multilingual NL but no longer generalized across model classes.

  - Limitations bullet on "Z stratification" updated: "not validated across model families" replaced with "supported on 7 models with model-class dependence"; remaining work narrowed to decoder-only LLM hidden states + tier2/tier3 OOD stimuli.

**Why**: M5 was the highest-leverage of the deferred items because the single-model P3 weakness was a reviewer attack surface and the cache of NL embeddings from Strategy D made the 7-model probing run almost free (~3 min). Discovering the model-class dependence (Nomic / UniXcoder collapse) is a genuine new finding that the original MiniLM-only P3 could not have produced.

---

## 2026-05-21 -- Strategy F: OOD NL-code alignment (closes C1 deferred portion)

**Context**: The C1 contamination caveat added in PR #3 explicitly pointed at tier2/tier3 stimuli as the deferred test: "Decisive separation requires either out-of-distribution operations (novel composite stimuli ... released in the experiment repository for this purpose) or matched-perplexity controls." The OOD stimuli existed in `experiments/data/stimuli/tier2_multistep.json` (30 multi-step) and `tier3_compositional.json` (20 compositional) but had never been embedded or analyzed.

**Decisions**:

  - Added `experiments/scripts/run_strategy_f_ood_alignment.py` running the same 7-model R_code matrix as Strategy D, but on the 50 OOD operations (binary_search, merge_sort, BFS, DFS, Bellman-Ford, topological_sort, A*, dynamic programming, ...) with multi-line function bodies (mean NL length 180 chars vs. 55 for tier 1; multi-line code vs. 1-liners).
  - Pre-registered hypothesis structure in the runner docstring before running: H_strong (R_code holds), H_weak (R_code drops to ~1, confirming caveat), H_partial (model-specific). This is recorded in the source file, not added post-hoc.
  - **Result**: 35/35 OOD cells significant. Aggregate R_code is HIGHER than tier-1 for every model (UniXcoder 1.07→1.15; MiniLM 1.16→1.31; Nomic 1.07→1.16; E5-small 1.13→1.28; E5-base 1.14→1.31; E5-large 1.20→1.33; BGE-M3 1.16→1.36). Cohen's d up to 4.12 (en, E5-large). The memorization hypothesis predicted a drop; observed direction is the opposite.

  - **Interpretation**: longer and more distinctive multi-step NL + multi-line function bodies provide more discriminating signal; the embedding alignment exploits this richer surface form rather than being damaged by reduced co-occurrence frequency. C1 deferred portion via OOD stimuli is **closed in favor of stronger PRH-for-code support**, not in favor of confirming the contamination concern.

  - Paper updated:
    - §5.5 contamination caveat paragraph: removed "left to future work" framing; pointer to OOD experiment below.
    - §5.5 new "Out-of-distribution NL-code alignment" paragraph + 7×5 OOD table + tier1↔OOD aggregate comparison.
    - Limitations "Pretraining contamination of NL-code stimuli" bullet renamed to "(partially addressed)" with summary of OOD result; residual matched-perplexity work remains future.

**Why**: This was the single most important deferred item because the contamination caveat (added in PR #3 for paper integrity) explicitly predicted a directional outcome. Running the test and reporting the result---in either direction---is what distinguishes the caveat from rhetorical hedging. The observed direction (OOD effect stronger than tier-1) is the strongest empirical anchor for the paper's PRH-for-code claim that the embedding-only paradigm can produce.
