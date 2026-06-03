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

---

## 2026-05-21 -- Model registry with frozen HuggingFace SHAs (closes C3)

**Context**: The C3 fix in PR #3 accepted floating-`main` risk for the pilot and relied on the existing `EmbeddingCache` for embedding-level reproducibility. After Strategy D / E / F all landed using the same 7-model set, the cost of pinning revision SHAs became trivial (one fetch via `huggingface_hub.HfApi`) and the benefit grew (any reviewer re-running the pipeline 6 months from now would otherwise pull a moved `main`).

**Decisions**:

  - Added `experiments/src/model_registry.py` with `MODELS_7_FROZEN` — the 7 (model_name, label, kwargs) tuples used by Strategy D / E / F, each pinned to its `main` commit SHA observed on 2026-05-21 via `HfApi().model_info(repo).sha`. `registry_sha_summary()` returns a JSON-serializable mapping for `run_meta` blocks.
  - sentence-transformers `>=5.5` accepts `revision=` in `SentenceTransformer.__init__`; confirmed via `inspect.signature`.
  - Refactored Strategy D / E / F runners to `from src.model_registry import MODELS_7_FROZEN, registry_sha_summary` and replaced their inline MODELS lists. Each runner's `run_meta` now includes `model_revisions` so the SHAs are recorded in every results JSON for forensic reproducibility.
  - `experiments/README.md` Reproducibility envelope bullet added: model-weight pinning policy + pointer to the registry's refresh snippet.

**Why**: C3 was originally classified as a Minor TODO because the embedding cache covered the practical reproducibility need. Centralizing the registry now (rather than after another experiment lands) prevents future SHA drift between runners and gives reviewers a single auditable location for "which exact weights did this paper use?"

---

## 2026-05-21 -- Extra-high recall code review (15 findings, all fixed)

**Context**: After PRs #1-#7 landed, `/code-review` ran at xhigh-effort recall mode (5 angles × ≤8 candidates → 1-vote verify → sweep, capped at 15 findings). Output was 15 confirmed/plausible defects spanning paper text drift, statistical method gaps, cache-poisoning vectors, and a partial-success silent-corruption hole in the FWE pipeline.

**Decisions (all 15 fixed in a single review-closure PR)**:

  - **V1 / V14 cache key drops revision + basename collision across orgs**: `SentenceTransformerEmbedder.name` now uses `f"st_{repo.replace('/', '__')}@{rev[:8]}"`. EmbeddingCache._key collisions across `intfloat/x` vs `sentence-transformers/x` and across SHA bumps are now distinct. C3 closure actually holds end-to-end.

  - **V13 cache key delimiter collision**: `EmbeddingCache._key` switched from `f"{m}:{'|'.join(texts)}"` to a JSON-encoded payload hash so texts containing `|` (e.g. `s1 | s2` in the union stimulus) cannot collide with split variants. Verified inline: `['a|b','c']` and `['a','b|c']` now hash to distinct keys.

  - **V18 dimension None for trust_remote_code modules**: `SentenceTransformerEmbedder.dimension` falls back to a one-text encode probe when the deprecated `get_sentence_embedding_dimension()` returns None. Nomic v1.5 no longer risks a silent skip from `int(None)`.

  - **V9 Mistral Retry-After hang**: `respect_retry_after_header=False` on the urllib3 Retry; backoff_factor=1 bounds total wait to ~31s instead of up to 5× server-sent `Retry-After`. Eliminates the multi-hour silent stall mode.

  - **V10 OpenAI timeout regression**: 60s → 300s. Legacy batch callers that need >60s server-side processing no longer hit a spurious timeout under the new client.

  - **V5 perm/bootstrap fallback to 1.0**: substituted NaN instead. `random_baseline_R_mean` now uses `np.nanmean`. New result range [1.0001, 1.0046] (tier1) / [1.0005, 1.0086] (OOD), still ≈1 as the paper claims, but no longer biased by silent 1.0 imputations.

  - **V6 p_value floor**: `(n_extreme + 1) / (n_valid + 1)` convention adopted. Reported p-values are now bounded below by `1/(n_perm+1) ≈ 1e-4`; no cell reports literal `0.0` (verified post-rerun: min nonzero p = 0.0001 across all 70 D+F cells). Reviewer push-back surface closed.

  - **V8 partial-success FWE silent invalidation**: Strategy D/E/F main() now `sys.exit(2)` on any failed model unless `Z_GAP_ALLOW_PARTIAL_RESULTS=1` is set. The "across 35 cells" claim in the paper can no longer be silently invalidated by a single OOM / trust_remote_code drift. The Nomic einops episode from PR #4 is the exact failure mode this guards against; the previous lenient behavior would have let it slip if `failed_models` had been ignored.

  - **V7 figures-before-save**: Strategy D/E/F save JSON BEFORE generating figures, with figures in a try/except. Multi-hour compute is no longer lost to a matplotlib font-cache failure.

  - **V11 Strategy E `categories[op_id]` KeyError**: replaced with `categories.get()` + explicit `_label` helper that returns None for unknown categories. Empty per-language test sets also produce `{skip: true}` cells with NaN accuracy instead of crashing on `clf.predict(np.array([]))`.

  - **V12 tier2/tier3 op_id uniqueness**: `load_ood_stimuli()` now asserts uniqueness with the duplicate list surfaced in the error message. Today's stimuli pass (verified inline: 50/50 unique), but a future id collision will fail loudly.

  - **V2 synthesis JSON envelope shim**: `_normalize_results_envelope()` unwraps `{_meta, results}` to a plain list so `run_cross_experiment_synthesis.py` keeps working with the new D/E/F JSON shape. Also added strategy_e and strategy_f to its known-files list.

  - **V20 synthesis treats `aggregate` as a 6th language**: explicit `if lang == "aggregate": continue` in the per-language counter loop. The "n_significant / total_cells" rate is now denominated against the real 5-language × 7-model = 35 grid, not 42.

  - **V4 datetime.utcnow deprecation in Strategy D**: replaced with `datetime.now(datetime.UTC)` to match Strategy E/F and survive future Python ≥3.13 removal.

  - **V3 paper §5.5 / Limitations "20 cells / four models / 20/20" drift (3 locations)**: updated to "35 cells / seven models / 35/35 + OOD 35/35", matching the Strategy D/E/F tables already inserted in PR #4/#5/#6.

**Re-execution**: Strategies D/E/F rerun after all fixes (~5 min, 7/7 models succeeded each). Cell-level R_code values unchanged at 2-decimal precision except UniXcoder tier1 aggregate (1.0649 ≈ 1.06 vs. previously printed 1.07 — rounding). Cohen's d_max for OOD shifted from E5-large (4.12) to E5-base (4.42); paper updated. All 35/35 + 35/35 + multi-model P3 conclusions hold.

**Why**: Recall-mode review surfaces real bugs at the cost of some false positives. Of the 15 confirmed findings, V8 (silent FWE invalidation) and V1 (cache key drops revision) would have been the most damaging if discovered after EMNLP submission. Closing them all in a single review-closure PR keeps the paper-evidence chain (Strategy D 35/35 tier1, P3 7-model, Strategy F 35/35 OOD) sound under reviewer push-back.

---

## 2026-06-03 -- Venue retarget to TACL + dialect integrity (T2) + decoder-only scope

**Context**: The "EMNLP 2026 ARR May" submission framing across README / submissions NOTES / registry was a prior-Claude-session assumption, NOT a user goal (user correction 2026-06-03). The paper was never submitted anywhere. arXiv is blocked (no endorser; arXiv tightened endorsement 2026-01-21). A Zenodo DOI preprint record exists. New target chosen by the user: **TACL** (journal, OpenReview, no endorsement, revise-and-resubmit fits a position+pilot paper by a single independent author; honest negative results are an asset in journal review, a liability in fast conference review).

**Honest-status correction needed (pending)**: `README.md:2` "Status: Under review (EMNLP 2026, ARR May cycle)" is FALSE (never submitted) and must be corrected to "Reproducible artifact + Zenodo DOI; target: TACL". Tracked separately from this entry.

**Dialect integrity (P0) — research-first redesign**:
A literature scan (date-anchored 2026-06-03) reframed the fix. Findings:
  - dialect-robustness has established methodology: VALUE (Ziems et al. 2022), DialectGen (arXiv:2510.14949), PTEB (arXiv:2510.06730 paraphrase axis), MMTEB (cross-lingual axis).
  - real dialect corpora exist: **MADAR** (25 Arab-city parallel) and **NADI 2024** (Egyptian↔MSA sentence-level) for Arabic; **AI Hub Korean Dialect** corpus (Gyeongsang, Jeju) for Korean.
  - LLM-generated dialect data is acceptable for augmentation but weak for measurement claims: "a small amount of human-annotated data beats much larger synthetic" (arXiv:2506.12158).

  Audit of the current artifact confirmed the claim was unsupportable:
  - `data/dialect_stimuli.json` (v1): `british`/`indian` are byte-identical to original English → "British d≈0.001" is an artifact of identical strings, not dialect robustness.
  - `data/dialect_stimuli_v2.json` exists with real Gyeongsang/Egyptian dialects but is **LLM-generated (Opus 4.6)** and was **never run** (strategy_6r results == v1_english results).
  - The `ordering` field in `strategy_6r_dialect_results.json` is mislabeled: it reads "dial < para < cross" for all 3 models, but only E5-large actually satisfies it; MiniLM and BGE-M3 show dial < cross < para (paraphrase distance EXCEEDS cross-lingual).

  **Decision: T2 (honest scope-down), chosen over T1 (real-corpus integration) and T3 (LLM-probe appendix).**
  - T1 (MADAR/NADI/AI-Hub) is the highest-rigor option but carries weeks of data-access friction for a SECONDARY result — disproportionate; deferred as a *sourced upgrade path*, not vague future work.
  - T3 (v2 LLM-dialect in appendix) adds a new "is this Gyeongsang authentic?" attack surface + runner-rewrite work, diluting the "we don't rest on LLM data" signal — rejected.
  - T2 retracts the continuum claim, reports only what the real data supports (embedding models collapse within-English orthographic variation; paraphrase-vs-cross-lingual ordering is model-dependent), and cites VALUE/MADAR/NADI/AI-Hub as the correct methodology + immediate next step. Converts a fabrication liability into a methodological-awareness strength.

  **Applied**: rewrote `paper/main.tex` dialect paragraph (§5 pilot) to the scope-note form; added 4 bib entries (ziems2022value, bouamor2018madar, abdulmageed2024nadi, aihub2024kodialect — author lists use "and others", AI Hub year approximate, flagged for camera-ready verification). v1/v2 data + generator preserved in repo for the T1 upgrade.

**Decoder-only scope cleanup**: tightened two Limitations bullets — (a) claims explicitly scoped to encoder-style sentence embedders; (b) added the frontier-closed-model unprobeability point (GPT/Claude/Gemini expose no embedding/hidden-state API → decoder-only extension is necessarily restricted to open-weight LLMs); (c) fixed a staleness bug where OOD stimuli were listed as "future work" though Strategy F already analyzed them (35/35 OOD).

**CodeSage status**: NOT done this PR. Unlike dialect T1, CodeSage is cheap (open-weight, no access friction) and closes a real attack surface (the "NL-only > code-trained" claim rests on a single code-trained model, UniXcoder). Recommended as the next pre-TACL task (needs a `CodeSageEmbedder` AutoModel+mean-pool class + D/E/F rerun, ~1-2h). If skipped, the "NL-only > code-trained" claim must be scoped to "single code-trained model in our set".

**Why**: TACL's rigor gate punishes exactly the two soft spots this entry addresses — an unsupportable graded-continuum claim resting on degenerate/mislabeled data, and an under-delimited decoder-only scope. Fixing both before submission (rather than after a reviewer flags them) is the disciplined-restraint signal a journal editor rewards.

---

## 2026-06-03 -- CodeSage attempt: blocked by version-rot, scoped the claim (E)

**Context**: After PR #9 (dialect T2 + decoder-only), the next pre-TACL task was CodeSage-Large-v2 as a modern code-trained model to close the "single code-trained model (UniXcoder)" gap behind the "NL-only > code-trained" claim. Estimated cheap (open-weight, no access friction).

**What happened**: The estimate was wrong. `codesage/codesage-large-v2` (1.3B, 2048-dim, ungated, SHA 6e5d6dc1) downloads fine but its 2024-era `trust_remote_code` module is multi-layer incompatible with the installed (2025/2026-era) transformers:
  1. `from transformers.modeling_utils import Conv1D` — Conv1D relocated to `transformers.pytorch_utils`. Fixable with a re-export shim (tried, worked).
  2. `'CodeSageModel' object has no attribute 'all_tied_weights_keys'` — newer transformers `from_pretrained` finalization expects a property the old custom model class does not define. Not safely fixable without reverse-engineering tied-weights (risk: wrong weight load -> subtly wrong embeddings).

**Decision: Option E (scope the claim, defer the model), chosen over C (isolated legacy-transformers env, ~45min+ with its own install uncertainty + lazy-load refactor) and D (pivot to another trust_remote_code code embedder = same version-rot lottery).**

Rationale: "cheap" was the only reason CodeSage out-prioritized dialect-T1; once it turned expensive + correctness-risky, the same logic that deferred dialect-T1 (disproportionate cost for a SECONDARY result) applies. Forcing a monkeypatched load would violate the rigor the TACL effort is built on. Consistency with the dialect-T2 discipline ("if real data can't support it, scope the claim") demanded the same treatment here.

**Applied**:
  - paper §5.5: "NL-only models achieve higher R_code than code-trained models" -> scoped to a within-set observation, not a categorical claim, with the broader-code-trained-set requirement stated inline.
  - paper §5.5 P3 paragraph: added a parenthetical noting both code-oriented models are the only ones in the set.
  - Limitations: new "Code-trained model coverage" bullet naming CodeSage-Large-v2 / Qodo-Embed as the intended additions and the CodeSage v2 remote-code/transformers incompatibility as the practical deferral reason.
  - Reverted the embeddings.py Conv1D shim (dead code: insufficient on its own, and the main pipeline never loads CodeSage).

**Local cleanup**: deleted the 2.4GB CodeSage download (`~/.cache/huggingface/hub/models--codesage--codesage-large-v2` + remote-code module) at user request; zero repo dependency (no .npz was ever computed since the load failed). The 7 paper models remain cached.

**Note for a future revisit**: `jinaai/jina-embeddings-v2-base-code` (a code-specialized embedder) is already in the local HF cache and could fill the code-trained robustness slot without CodeSage's legacy-environment requirement — a cleaner Option-D path if the code-trained comparison is reopened.
