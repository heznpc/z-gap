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
