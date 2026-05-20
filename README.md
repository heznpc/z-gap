Research Program: 3 (Representation, Language, and Cultural Cognition)
Status: Under review (EMNLP 2026, ARR May cycle) — reproducible artifact
Relationship to other work: Anchor of Program 3 (companions: macaronic, third-vertex-llm, habitus)

# Z-Gap — Beyond the Chomsky Wall

The Platonic Representation Hypothesis (PRH) claims that networks trained on different modalities converge toward a shared latent $Z$. This paper accepts PRH and asks the next question: **does convergence imply communicability?** We argue $Z$ is stratified — $Z_{\text{sem}}$ (what is computed) converges cross-culturally, while $Z_{\text{proc}}$ (derivation path) and $Z_{\text{prag}}$ (communicative frame) remain culturally mediated — so existence and communicability are distinct properties. A pilot across 5 languages × 100 operations shows P2 (cross-lingual NL-NL invariance) failing at the description level even as NL-code alignment succeeds: **convergence without communicability**.

## Currently implemented

- `paper/main.tex` — canonical manuscript (970 lines, ACL-styled)
- `paper/references.bib` — shared bibliography for all venue copies
- `submissions/emnlp-2026/main.tex` — EMNLP 2026 venue snapshot (880 lines)
- `submissions/colm-2026/main.tex` — COLM 2026 venue snapshot (847 lines)
- `experiments/` — reproducible pilot: 100 stimuli (50 computational + 50 judgment) × 5 languages × dialectal variants (~1,800 inputs), embedded through 8 models (MiniLM, E5-small/base/large, BGE-M3, Qwen3-Embedding-8B, jina-v3, Codestral Embed). Tests P1, P2, P2-dialect, P3, P7
- `planning/` — TODO, decisions log, review notes, P2-strategy audit

## Planned

- EMNLP 2026 commitment (Aug 2, 2026) after ARR May reviews; conference Oct 24–29, Hungary
- Scope expansion (more languages or operations) only if review responses require it
- Reconcile content drift between `paper/main.tex` and the two `submissions/*/main.tex` snapshots manually before each venue cycle

## Design intent

- **DDD-style layout** (`paper/` canonical, `submissions/<venue>/` frozen snapshots): forces editorial drift between venues to be explicit rather than silently mutating one shared file. Rationale in `planning/decisions.md` 2026-04-19 entry.
- **`experiments/scripts/` vs `experiments/src/`** is a library-vs-entry-point split, not a version distinction.
- **5 languages × 100 ops** is sized as a pilot, not a benchmark: enough to show the qualitative P2 break, small enough to remain reproducible end-to-end on a single machine.
- **Z stratification** is the load-bearing theoretical move — it lets PRH stay true while explaining why two systems sharing $Z$ can still fail to communicate.

## Non-goals

- Refuting PRH. The paper refines it, not against it.
- A single auto-synced manuscript across venues. Venue snapshots are intentionally frozen.
- A general theory of "communicability" for arbitrary modalities — scope is NL ↔ code.
- Claims about subjective experience or consciousness from representational similarity. The neuroscience parallel is by analogy only.

## Redacted

- (none — this repo carries no external persons, tokens, or third-party identifiers)

## Reproduce the pilot

```bash
cd experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OpenAI + Mistral keys
python scripts/run_all.py
```

See [experiments/README.md](experiments/README.md) for model list and prediction-to-script mapping.

## License

MIT
