# TODO — Beyond the Chomsky Wall

## Submission Target
- **Venue**: EMNLP 2026 (ARR submission)
- **Cycle**: April 15, 2026 (first attempt) → May 25, 2026 (revision if needed)
- **Commitment deadline**: August 2, 2026

## Before April 15 (ARR April cycle)

### Pilot Experiment Expansion
- [ ] Add latest multilingual models: Llama-3.1, Qwen2.5, Gemma-2, Aya-23
- [ ] Add code-trained models: CodeLlama, DeepSeek-Coder-v2
- [ ] Scale operations from 100 → 300–500
- [ ] P1 retest: same model family at different scales (e.g., E5-small/base/large)
- [ ] P6: cross-model D_train sensitivity (GPT-4 vs HyperCLOVA X vs Qwen)

### Paper Polish
- [ ] ARR Responsible NLP Checklist (separate form, not in .tex)
- [ ] Add 1–2 more running examples (e.g., date parsing, name entity)
- [ ] Sovereign AI → computational sovereignty connection in Ethics (optional)

## After Review (May cycle revision if needed)
- [ ] Address reviewer feedback
- [ ] Strengthen weak points identified in reviews
- [ ] Resubmit to ARR May 25 cycle

## Files
- `paper/main.tex` — original v5 (custom format)
- `paper/main_emnlp.tex` — EMNLP/ACL format (submit this)
- `paper/references.bib` — BibTeX bibliography
- `paper/acl.sty`, `paper/acl_natbib.bst` — ACL style files
- `experiments/` — pilot experiment code and data
