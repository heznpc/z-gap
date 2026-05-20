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
