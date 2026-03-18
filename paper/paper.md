# Beyond the Chomsky Wall: Platonic Representations as the Convergence Point of Natural Language and Code

> Target: COLM 2027 or ICML 2027 Workshop
> Format: 8 pages + references + appendix

---

## Abstract

Programming languages use context-free (Type-2) grammars because deterministic parsing requires them — a mathematical necessity, not a convention. Three waves of natural-language programming — keyword substitution, structured NL, and LLM-mediated translation — all operate at the surface-form level and inherit this constraint. We argue that the Platonic Representation Hypothesis (Huh et al., ICML 2024) reframes the problem: if NL and code converge toward a shared latent structure Z, the Chomsky wall constrains only surface-form projections, not representations. We propose Z is stratified — Z_semantic (computational result) converges cross-culturally while Z_procedural (derivation path) and Z_pragmatic (communicative frame) remain culturally mediated. We formalize disambiguation cost migration using information theory, showing that specification complexity is conserved across interfaces. Training data composition (D_train) contaminates even Z_semantic, creating ethical risks. We derive seven falsifiable predictions, connect Z_semantic to denotational semantics, and identify a verification paradox that constrains the deployment path.

---

## 1. Introduction

Every mainstream programming language — from FORTRAN (1957) to Rust (2015) — uses a context-free grammar. This is not arbitrary. Compilers require deterministic parsing: the same source must produce the same parse tree every time. Context-free grammars (Chomsky Type-2) guarantee this in O(n) to O(n³) time. Natural language does not — and cannot, because identical sentences carry different meanings depending on context. Even languages that appear "natural" — Python's `if x > 5:`, SQL's `SELECT * FROM users` — are engineered to remain within Type-2 bounds.

Attempts to make programming "natural" have come in three waves: **Wave 1** (keyword substitution, 1960s–) replaces English tokens with native-language equivalents (COBOL, Korean languages like 씨앗 and Han) — the grammar remains Type-2. **Wave 2** (structured NL, 2024–) constrains prose into parseable templates (CoRE, SudoLang, Spec-Driven Development) — new formal grammars that look like English. **Wave 3** (LLM-mediated translation, 2023–) generates code from free-form NL (Claude Code, Cursor, Lovable) — the dominant "vibe coding" paradigm, but non-deterministic: same input can produce different outputs, and its non-determinism has been shown to be fundamentally intractable (ICSE 2026).

The QWERTY keyboard was designed to prevent typewriter jams. The constraint vanished with computers — but the layout persisted. Programming syntax may be in a similar position: the original constraint (deterministic parsing) produced context-free grammars; LLMs can process context-sensitive input; but the entire toolchain is built on formal syntax.

We argue that the Platonic Representation Hypothesis offers a framework for understanding this question. Natural language and code are projections of a shared latent structure (Z); the Chomsky hierarchy describes relationships between surface forms, not representations; LLMs operate where the Type-1/Type-2 distinction is irrelevant; the productive direction is to execute at the representation level directly.

**Contributions:** (1) A formal framework connecting PRH to the code–language divide, including connection to denotational semantics and information-theoretic formalization of disambiguation conservation (§3, §4). (2) Stratification of Z with differential convergence, D_train dependence analysis, and ethical implications (§3.4–3.5, Appendix B, E). (3) Seven falsifiable predictions, evaluation criteria, a pilot experiment design, and a verification paradox (§3.7, §5, §6.5).

**Running example.** Throughout this paper: **"주어진 목록에서 가장 큰 값 세 개를 찾아라"** ("Find the three largest values in the given list"). This sentence is computationally precise yet linguistically ambiguous — "가장 큰" (largest) lacks type information, comparison criterion, boundary handling, and output ordering. In Type-2 programming, the programmer resolves all ambiguities by writing `sorted(lst, reverse=True)[:3]`. We track how each wave, each Z layer, and the verification paradox handle these ambiguities.

---

## 2. Current Landscape: Surface-Form Approaches

All three waves operate at the surface-form level. Wave 1 changes terminal symbols but preserves the parse tree. Wave 2 imposes structure on NL, creating new Type-2 grammars — the paradox is that making NL precise enough to execute requires removing the properties that make it natural. Wave 3 works for prototyping but is non-deterministic: our running example produces `sorted(lst, reverse=True)[:3]`, `heapq.nlargest(3, lst)`, and a buggy list comprehension across three runs. See Appendix A for code examples.

| Dimension | Wave 1: Keyword | Wave 2: Structured NL | Wave 3: LLM Translation | Proposed: Z-Level |
|-----------|:-:|:-:|:-:|:-:|
| **Chomsky level** | Type-2 (unchanged) | Type-2 (re-imposed) | Type-1 → Type-2 | Type-1 → Z → Type-2 |
| **Determinism** | Full | Full | Non-deterministic | Deterministic (target) |
| **Expressiveness** | = base language | < NL | ≈ NL | = NL |
| **Verification** | Standard | Standard | None (stochastic) | At Z (open problem) |
| **Disambiguation** | By programmer | By programmer | By LLM (non-det.) | By system (at Z) |

Both RAG and NL→Code map NL to structured output via learned representations, but code requires exact semantics where RAG tolerates approximation — cosine similarity 0.95 is fine for document retrieval but a bug for program execution.

---

## 3. The Platonic Representation Hypothesis

### 3.1 Overview and Formal Framework

Huh et al. (ICML 2024) present evidence that neural networks trained on different data, objectives, and modalities converge toward a shared statistical model of reality in their representation spaces. The hypothesis posits a "platonic" ideal representation — a shared latent structure of which images, text, and other modalities are projections. Ziyin et al. (2025) prove that embedded deep linear networks trained with SGD become "Perfectly Platonic" — every pair of layers learns the same representation up to rotation. Crucially, they identify six conditions that break convergence (weight decay, label transformation, saddle convergence, input heterogeneity, gradient flow vs. SGD, finite-step edge-of-stability), suggesting PRH convergence is the *default* for well-behaved training but can fail under specific conditions. We introduce semi-formal definitions to ground the analysis:

**Definition 1 (Modality Encoder).** An encoder E_i : M_i → ℝ^d maps inputs from modality M_i to d-dimensional representations; a decoder D_i : ℝ^d → M_i maps back.

**Definition 2 (Representational Convergence).** For semantically equivalent inputs x ∈ M_i and y ∈ M_j: lim_{scale→∞} sim(E_i(x), E_j(y)) → 1.

**Definition 3 (Shared Latent Structure).** Z ≈ lim_{scale→∞} Im(E_i) for all modalities i. Critically, Z = Z(scale, D_train) — two models at the same scale trained on different corpora may converge to different Z (§3.5).

**Definition 4 (Z Stratification).** Z decomposes into: **Z_sem** (what is computed, cross-culturally convergent), **Z_proc** (how it is derived, culturally mediated), **Z_prag** (for whom it is framed, culturally specific).

**Definition 5 (The Chomsky Wall).** The computational boundary between Type-2 and Type-1. *Surface-level* if it constrains only parsing, not representations.

**Central Claim.** If PRH holds for code as a modality, the Chomsky wall constrains only the projection D_i : Z → M_i, not Z itself. Operations at Z are not subject to Chomsky hierarchy constraints because Z is continuous — it has no grammar to classify.

### 3.2 Connection to Denotational Semantics

Denotational semantics (Scott & Strachey, 1971) maps programs to mathematical objects in a semantic domain **D**, independent of syntax. We observe a structural parallel: Z_semantic is a neural approximation of **D**.

**Conjecture.** lim_{scale→∞} Z_sem(E(d)) ≈ ⟦P⟧ (up to continuous bijection), where d is a NL description and ⟦P⟧ is the denotation of program P. This explains why Z_semantic converges cross-culturally (denotational semantics is culture-invariant), why convergence is stronger for computational domains (well-defined denotations), and why compositionality is the hard problem (denotational compositionality requires homomorphic structure NL does not guarantee). For our running example: the denotation of `sorted(lst, reverse=True)[:3]` is a function List[Comparable] → List[Comparable]; PRH claims "가장 큰 값 세 개를 찾아라" converges to the same point in Z_semantic.

### 3.3 Evidence for Code as a Modality

- **Aligned embeddings.** UniXcoder achieves 77.1% MRR on NL-code search across six languages.
- **Cross-model transferability.** Steering vectors from 7B models control 70B model behavior (ACL 2025) — Z structure is recoverable across architectures.
- **Language-independent code semantics.** LLMs develop language-independent semantic representations of code, with syntax-specific components in early layers feeding shared semantic components in middle layers (ICSE 2025 NIER).
- **The Naturalness Hypothesis.** Software is statistically natural — code cross-entropy approaches that of English text (Hindle et al., 2012), a precondition for PRH convergence.
- **Translation emergence.** LLMs trained primarily on NL generate code without code-specific training; cross-language synergies follow scaling laws (Cassano et al., 2025).
- **Bidirectional transfer.** Training on NL documentation improves code completion, and vice versa.

### 3.4 The Stratification of Z

Different mathematical traditions derive the same theorem via different paths (Bourbaki: axiomatic; Soviet: constructive; Indian: intuitive-inductive). The theorem is identical; the derivation differs. Similarly, `sort(array)` produces the same output regardless of functional or imperative derivation.

| Layer | Content | Convergence | Evidence |
|-------|---------|-------------|----------|
| Z_sem | What is computed | Cross-culturally convergent | Semantic Hub Hypothesis (ICLR 2025); shared grammatical features (NAACL 2025) |
| Z_proc | How it is derived | Culturally mediated | Math tradition differences; reasoning-language effects (Li et al., 2026) |
| Z_prag | For whom it is framed | Culturally specific | Pragmatic inference failures (2024); Cultural Frame Switching (COLING 2025) |

**Running example.** Z_sem: select top-3 by magnitude (culture-invariant). Z_proc: Korean developer may filter ("큰 값을 제외하지 않는다"), Haskell developer may write `take 3 . sortBy (flip compare)`. Z_prag: "~아라" is a Korean imperative frame.

### 3.5 Multilingual Convergence and D_train Dependence

PRH implies a single convergence point. Multilingual LLMs complicate this. Two models compete:

**Model A: The multilingual human.** The LLM code-switches between culturally-mediated representations — Korean input activates a Korean-inflected Z; English input activates an Anglo-American one.

**Model B: The Platonic oracle.** The LLM contains a culture-invariant Z; cultural variations are decoding artifacts.

Evidence increasingly favors **Model A for judgment, Model B for structure**: 9–56pp moral judgment gaps across languages including preference reversals (Vida et al., 2024; Agarwal, 2024), GPT-4o displays Cultural Frame Switching with distinct "personalities" per language (COLING 2025), and reasoning-language effects contribute twice the variance of input-language effects (Li et al., 2026). But shared grammatical features persist across typologically diverse languages (Brinkmann et al., NAACL 2025), a semantic hub appears in middle layers (Wu et al., ICLR 2025), and language identity is encoded via sparse dimensions separate from semantic content (Zhong et al., 2025).

**Resolution:** Z_semantic converges (Model B); Z_procedural and Z_pragmatic remain culturally distributed (Model A). The Aristotelian critique (Gröger et al., 2026) refines further: what converges is local neighborhood *topology* (preserved neighborhoods), not global *geometry* (preserved distances). Z may be a topological rather than geometric structure.

**D_train dependence.** Z is not determined by scale alone — it depends on training data cultural composition: Z = Z(scale, D_train). Even "purely computational" domains carry implicit cultural defaults: sort order conventions, null handling semantics, error semantics (Python exceptions vs. Go explicit errors vs. C undefined behavior), and number formatting. The clean Z_semantic/Z_procedural separation is an idealization: Z_semantic(observed) = Z_semantic(ideal) + bias(D_train). The sovereign AI movement (Korea's five-consortium initiative, Japan's ABCI 3.0, SEA-LION) implicitly acknowledges this — if Z were truly Platonic, culturally-native models would be unnecessary. The market is voting for Model A. See Appendix B for detailed analysis.

### 3.6 The Convergence-Determinism Gap

PRH gives sim → 1 - δ, but execution requires δ = 0. Three resolutions: (1) **Discretization at Z** — equivalence classes, formally equivalent to constructing a new grammar over Z. (2) **Verification after projection** — accept δ > 0 at Z, demand δ = 0 at output. (3) **Probabilistic execution** — statistical guarantees (probability 1 - 10^{-9}). This gap is the central technical obstacle.

### 3.7 Falsifiable Predictions

**P1 (Scale-Convergence).** NL-code cosine distance decreases monotonically with model scale.
**P2 (Cross-Lingual Semantic Invariance).** "정렬하라" (Korean: sort) and "sort this" (English) are closer in Z than "sort this" and "reverse this" in English. This should *not* hold for judgment-involving operations.
**P3 (Stratification Separability).** Probing classifiers for *what is computed* generalize cross-lingually; probes for *how* show language-specific patterns.
**P4 (Domain-Dependent Determinism).** Constrained decoding achieves higher consistency for computational than judgment tasks.
**P5 (Disambiguation Migration).** Total disambiguation effort is conserved — syntax errors migrate to semantic clarification queries.
**P6 (D_train Sensitivity).** Z_semantic divergence across models correlates with D_train cultural divergence, even for "purely computational" operations with implicit criteria.
**P7 (Spacing Robustness).** Z distance between spacing variants of the same sentence < Z distance between semantically different sentences with identical spacing (see Appendix C).

---

## 4. Reinterpreting the Chomsky Wall

### 4.1 The Wall Is Real — at the Surface

| Type | Grammar | Parsing Complexity | Examples |
|------|---------|-------------------|----------|
| 3 | Regular | O(n) | Regex |
| 2 | Context-free | O(n³), deterministic | Python, C, Java |
| 1 | Context-sensitive | PSPACE-complete | Natural language |
| 0 | Unrestricted | Undecidable | Turing machines |

No deterministic algorithm resolves Type-1 ambiguity in polynomial time. Type-2 constraints also function as a *disambiguation mechanism*: every valid expression has exactly one parse.

### 4.2 The Wall Dissolves — at the Representation Level

LLMs map input sequences to high-dimensional representations via attention over full context. "x가 5보다 크면" (Korean NL), `if x > 5:` (Python), `(> x 5)` (Lisp) all map to nearby points (Semantic Hub Hypothesis). Language identity is encoded via sparse dimensions orthogonal to semantic content (Zhong et al., 2025). The Chomsky hierarchy describes parsing complexity of surface forms — it says nothing about representing meaning.

**Surface-form boundaries.** Between NL and Z exists a spectrum of representational directness: NL (linguistic, culturally mediated) → emoji/pictographs (iconic, semi-universal) → mathematical symbols (formal, universal) → code operators → Z (continuous). Emoji are natural experiments in Z_semantic — near-universal yet exhibiting cultural variation that mirrors Z stratification (🙏 = prayer/gratitude/high-five). See Appendix D. Tokenization introduces a bottleneck: spacing conventions (Korean 띄어쓰기, Chinese segmentation) leak through tokenizers into Z, and a "token tax" (Petrov et al., 2024) means non-Latin languages require 2–15× more tokens for the same content. See Appendix C.

### 4.3 Disambiguation Migration

Removing the Chomsky wall does not remove the need for disambiguation — it **migrates** the cost. In Type-2 programming, the programmer disambiguates at write time via grammar constraints; in representation-level execution, the system disambiguates at Z.

We conjecture a *disambiguation conservation principle*: any system accepting ambiguous input must resolve ambiguity somewhere, and the total cost is approximately conserved across architectures. This is a design heuristic, not a formal theorem — but it suggests that representation-level systems should not promise to eliminate programming difficulty, only to *relocate* it from syntactic precision to semantic verification.

The practical question is whether the relocation is net beneficial: is it easier for a human to learn `if x > 5:` or to verify that the system correctly interpreted "큰 값을 제외하라" ("exclude large values")? For professional developers, the current paradigm may be more efficient. For the billions who can express computational intent in natural language but cannot program, the tradeoff favors representation-level systems even if disambiguation cost is conserved.

**Running example.** "가장 큰 값 세 개를 찾아라" contains at least four ambiguities: (1) type — what kind of values? (2) comparison criterion — "가장 큰" by what measure? (3) boundary — fewer than three elements? (4) output ordering — sorted result? In Type-2, `sorted(lst, reverse=True)[:3]` resolves all four silently — the type system handles (1), comparison operator handles (2), slicing handles (3), and sort determines (4). At Z, each must be resolved by inference, query, or default.

**Information-theoretic formalization.** For task T with unique behavior B(T), the specification complexity K(T) is invariant across interfaces:

    K(T) ≤ H(S) + H(T|S) + O(1)

where H(S) is surface-form entropy and H(T|S) is residual ambiguity. Type-2 interfaces maximize H(S) (syntactic precision) and minimize H(T|S). NL interfaces minimize H(S) but face H(T|S) > 0 — remaining bits must be provided through disambiguation. This connects to Piantadosi et al. (2012), who argue ambiguity is a feature of efficient NL communication — but resolution cost cannot be avoided when exact computation is required. Via Rate-Distortion Theory: there is a minimum specification rate below which semantic error exceeds acceptable thresholds. Type-2 operates at zero distortion with high rate; NL operates at lower rate with nonzero distortion corrected downstream.

**Implication.** Representation-level systems do not reduce total specification information — they redistribute it: less from the user upfront (lower syntactic cognitive load), more from the system at Z (higher inference/verification cost).

### 4.4 Why Translation Fails but Representation Might Work

Wave 3 accepts NL (Type-1), translates to code (Type-2), executes. The translation step is non-deterministic and does not preserve semantics (Cheung, 2025). A representation-level approach instead: (1) accept NL, (2) map to Z, (3) execute from Z directly or project to code deterministically via constrained decoding.

---

## 5. Toward Representation-Level Execution

### 5.1 Architecture

A representation-level system would: (1) accept any surface form, (2) normalize surface form (spacing recovery, OCR correction), (3) encode to Z, (4) verify at Z and disambiguate (cultural context + completeness check), (5) execute at Z or project to Type-2.

```
User input (any surface form, any language, any modality)
    → Surface normalization (spacing, OCR, ASR)
    → Encoder to Z (SONAR-like, 200 languages)
    → Cultural context + Verification + Disambiguation at Z
    → Direct execute at Z (LCM) | Project to code (constrained decoding)
```

SONAR supports speech as input — the spacing problem disappears when the input modality has no spacing to begin with.

### 5.2 Existing Pieces

- **Constrained decoding** (Outlines, LMQL): Z → Type-2, made deterministic. Requires specifying target grammar.
- **Large Concept Models** (Meta): Operates on SONAR sentence embeddings — architecturally closest to the proposed pipeline, but performs generation, not execution.
- **Latent execution systems**: LaSynth (Meta, NeurIPS 2021) learns latent representations to approximate execution of partially generated programs. Latent Program Network (LPN, NeurIPS 2025 spotlight) learns a latent space of implicit programs and executes via neural decoder — the closest existing system to Wave 4, achieving 3rd place at ARC Prize 2024. COCONUT (Meta, 2024) bypasses language tokens entirely, reasoning in continuous latent space. Code World Models (CWM, Meta FAIR, 2025) models program execution trajectories, tracking variable states — but at the token level, not in SONAR space. These demonstrate that execution at Z is *feasible* in limited domains; the gap is unifying them with PRH-predicted convergence.
- **Neuro-symbolic systems** (Scallop, Lobster/ASPLOS 2026): Combine neural perception with symbolic reasoning, bridging continuous representations and discrete logic. Directly relevant to the Z → verification pipeline. Differentiable relaxations of discrete constraints may enable partial verification without full Type-2 projection.

### 5.3 Open Problems

1. **Determinism at Z** — PRH convergence ≠ identity. Temperature-zero decoding helps but doesn't eliminate stochasticity.
2. **Verification at Z** — Can we verify program properties in continuous embedding space? Connects to neural theorem proving.
3. **The grounding problem** — `x > 5` must mean *exactly* "strictly greater than 5." What representational precision suffices?
4. **Compositionality** — Code composes via strict nesting; NL composes loosely. Categorically: the question is whether a functor F: NL_comp → Prog preserves compositional structure. Most feasible for *compositionally transparent* NL ("first sort, then take top three"); hardest for *compositionally opaque* NL ("make it like the old version but faster").
5. **The specification problem** — If the user writes NL → system maps to Z, the user cannot inspect Z. This is a new form of opacity: not "I can't read the code" but "I can't read the intent representation." Unlike code opacity, which can be resolved by reading source, Z-level opacity has no corresponding inspection mechanism.
6. **Cultural invariance** — If Z_semantic carries cultural inflection for judgment-involving domains (§3.5), the "Platonic ideal" is a family Z(c), not a single Z. Execution must choose which Z layer to operate on — and the choice itself may be culturally mediated.
7. **Surface-form invariance** — The encoder must map all spacing/OCR/typographic variants of the same meaning to the same Z. Current tokenizers fail this for many languages — Korean 띄어쓰기 variants, Chinese segmentation alternatives, and English OCR artifacts all produce different token sequences for identical content. Byte-level models (ByT5, CANINE) bypass tokenization but at higher compute cost.

### 5.4 Evaluation Criteria

| Criterion | Metric | Baseline | Target |
|-----------|--------|:--------:|:------:|
| Convergence | NL-code cosine similarity | 0.77 MRR | > 0.95 |
| Determinism | Same NL → same output rate | ~0.6 | > 0.99 |
| Cross-lingual invariance | σ of sim. across NL languages | N/A | σ < 0.05 |
| Verification at Z | % properties verifiable without projection | 0% | > 50% |
| Disambiguation efficiency | Clarification queries per task | N/A | ≤ syntax errors/task |
| Spacing robustness | spacing variation / semantic variation | N/A | > 10× |
| D_train invariance | Cross-model Z_sem agreement | N/A | > 0.9 cosine |

### 5.5 Pilot Experiment

We propose a concrete experiment testing P1, P2, P6, and P7: 50 computational + 50 judgment operations described in five languages (English, Korean, Chinese, Arabic, Spanish), embedded through SONAR, sentence-transformers, and frontier models. The discriminability ratio R = d_inter/d_intra tests whether cross-lingual same-operation similarity exceeds within-language different-operation similarity. Extensions test spacing robustness (Korean 4 variants) and D_train sensitivity (GPT-4 vs. HyperCLOVA X vs. Qwen). Estimated cost: <$1,500; timeline: 3–4 weeks. See Appendix F for full protocol.

---

## 6. Discussion

### 6.1 Implications

For compilers, the wall is absolute. For LLM-based systems, it is irrelevant in practice. For hybrid systems, it becomes a design choice. Python occupies the closest point to NL within the Type-2 boundary — the productive question is not "push further within Type-2?" but "accept NL and project to Type-2 via Z?"

If representation-level execution becomes practical, programming languages shift from human-writable specifications to machine-generated projections from Z — languages become output formats. Debugging shifts from code to Z, requiring new visualization tools and provenance links.

### 6.2 Cultural Boundary and Safety

Z_semantic is culture-invariant for purely computational domains (sorting, arithmetic); Z carries cultural inflection for judgment-involving domains (risk assessment, prioritization) — requiring Z(c) rather than Z. D_train dependence creates ethical risks: computational colonialism (English-derived norms as "neutral" Z), consent opacity (users cannot see which cultural assumptions shaped interpretation), accountability gaps, and cultural feedback loops (Z-projected outputs re-enter D_train). Representation-level systems must make cultural parameterization explicit. See Appendix E for full analysis and mitigation paths.

### 6.3 Non-English Programming

Korean programmers face a double translation: thought (Korean) → code (English syntax) → execution. Representation-level execution would eliminate both: Korean intent → Z → execution. Research on reasoning-language effects (Li et al., 2026) suggests the "thinking in English" overhead is real — the global talent pool could expand by an order of magnitude.

### 6.4 The Verification Paradox

A deeper tension emerges: if verification requires formal methods, and formal methods require Type-2 input, the system must project from Z to Type-2 for verification — reintroducing the Chomsky wall at the verification stage. The wall is dissolved at input but reassembled at verification.

```
NL input → Z → [verify at Z? → no current tools]
                      ↓
                 project to Type-2 → verify → execute
                                     ↑
                                     wall reappears
```

Near-term systems will use a **hybrid architecture**: Z provides expressiveness; Type-2 projection provides verifiability. Full representation-level execution — input, verification, and execution all at Z — requires advances in neural verification. The neuro-symbolic literature (Scallop, Lobster) offers partial tools: differentiable verification predicates that operate on continuous representations without full Type-2 projection. But these handle only simple properties; general program verification at Z remains open. The paradox is not fatal but constrains the deployment path: the Chomsky wall retreats inward, from the user-facing boundary to the system-internal verification layer. Progress is measured by how far inward the wall can be pushed.

### 6.5 Limitations

- PRH is a hypothesis, not a theorem; sufficiency for exact semantics is unclear.
- "Representation-level execution" is a research direction, not a working system.
- The Z stratification is a conceptual framework, not empirically validated.
- The disambiguation conservation relies on Kolmogorov complexity (uncomputable in general).
- The D_train analysis suggests Z_semantic "culture-invariance" may be an idealization.
- The ethical analysis identifies risks but proposes heuristics, not proven safeguards.

---

## 7. Related Work

**PRH and convergence.** Huh et al. (ICML 2024) established convergence; Ziyin et al. (2025) proved it for deep linear networks; Gröger et al. (2026) showed global convergence is confounded but local topology persists.

**Multilingual representations.** The Semantic Hub Hypothesis (Wu et al., ICLR 2025) demonstrates shared representations; however, multilingual ≠ multicultural (Rystrøm et al., 2025): LLMs produce different moral judgments across languages and reflect creator ideology (Buyl et al., 2025).

**NL programming and vibe coding.** Wave 3's non-determinism is fundamentally intractable (ICSE 2026). The Moltbook incident (2026) exposed 1.5M API keys through vibe-coded services.

**Latent execution.** LPN (NeurIPS 2025), COCONUT (Meta, 2024), LaSynth (NeurIPS 2021) demonstrate execution in representation space. "Beyond Syntax" (ICSE 2025) shows LLMs develop language-independent code semantics. CWM (Meta, 2025) models code execution at token level.

**Tokenization and naturalness.** Tokenizers introduce up to 15× cost unfairness across languages (Petrov et al., 2024). Software is "natural" — statistically predictable like NL (Hindle et al., 2012).

---

## 8. Conclusion

For seventy years, programming languages have been constrained to context-free grammars — not by choice but by the mathematical requirements of deterministic parsing. Three waves of natural-language programming have attempted to bridge the gap; all operate at the surface-form level.

The Platonic Representation Hypothesis suggests this wall is a property of surface forms, not representations. Neural networks trained on language and code converge toward shared representations where the Type-1/Type-2 distinction is irrelevant. But Z is not monolithic: it stratifies into Z_semantic (convergent), Z_procedural (culturally mediated), Z_pragmatic (culturally specific). Z depends on D_train, and even Z_semantic carries bias from training data cultural composition.

The disambiguation cost borne by Type-2 grammars does not vanish — it migrates to the representation level. We formalize this as an information-theoretic conservation law: K(T) ≤ H(S) + H(T|S). This creates a verification paradox: near-term systems must project back to Type-2 for verification, pushing the wall inward. Between NL and Z, a spectrum of representational directness exists — from emoji to mathematical symbols to code — and surface-form properties like spacing leak through tokenizers into Z, creating additional bottlenecks.

We connect Z_semantic to denotational semantics, derive seven falsifiable predictions, and propose a pilot experiment. The pieces exist — SONAR, LCM, LPN, constrained decoding, cultural steering vectors, neuro-symbolic verification. They have not been unified under the representation-convergence framework that PRH provides. The research agenda is to unify them — while confronting the cultural stratification that determines where representation-level execution can and cannot succeed.

The pieces exist — SONAR embeddings, Large Concept Models, Latent Program Networks, constrained decoding, cultural steering vectors, neuro-symbolic verification. They have not been unified under the representation-convergence framework that PRH provides. The research agenda is to unify them — while confronting the cultural stratification of Z and the D_train dependence that determines where representation-level execution can and cannot succeed.

Like QWERTY, formal syntax may be a vestige of a transcended constraint. Unlike QWERTY, the cost of persistence is not ergonomic inefficiency but a fundamental barrier: billions of people can express computational intent in natural language but cannot program. The wall need not fall everywhere — but where Z_semantic converges, it need not stand. And we must ensure that the Z through which convergence is measured does not silently privilege one culture's computational norms over all others.

---

## References

[1] Huh, M., Cheung, B., Wang, T., and Isola, P. "The Platonic Representation Hypothesis." ICML 2024.
[2] Chomsky, N. "Three Models for the Description of Language." IRE Trans. Information Theory, 1956.
[3] CoRE / AIOS. "Natural Language Is All a Computer Needs." COLM 2025.
[4] Elliott, E. "SudoLang: A Powerful Pseudocode Programming Language for LLMs." 2023.
[5] Fowler, M. "Spec-Driven Development: Tools." martinfowler.com, 2025.
[6] Brooker, M. "Natural Language Programming." brooker.co.za, 2025.
[7] Keles, A. "LLMs Could Be But Shouldn't Be Compilers." alperenkeles.com, 2025.
[8] Shumailov, I. et al. "AI Models Collapse When Trained on Recursively Generated Data." Nature, 2024.
[9] Shire. "AI Coding Agent Language." phodal/shire, GitHub, 2025.
[10] Xodn348. "Han: Korean Programming Language with LLVM." GitHub, 2025.
[11] Du, X. et al. "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval." EMNLP 2025.
[12] Wu, W. et al. "The Semantic Hub Hypothesis." ICLR 2025.
[13] Brinkmann, M. et al. "Shared Representations of Latent Grammatical Concepts." NAACL 2025 (Oral).
[14] Gröger, F., Wen, S., and Brbic, M. "Revisiting the Platonic Representation Hypothesis: An Aristotelian View." 2026.
[15] Vida, K., Damken, F., and Lauscher, A. "Decoding Multilingual Moral Preferences." AAAI/ACM AIES, 2024.
[16] Agarwal, U. "Ethical Reasoning and Moral Value Alignment of LLMs Depend on Prompt Language." LREC-COLING 2024.
[17] Li, N. et al. "Untangling Input Language from Reasoning Language." 2026.
[18] Veselovsky, V. et al. "Localized Cultural Knowledge is Conserved and Controllable in LLMs." 2025.
[19] Buyl, M. et al. "Large Language Models Reflect the Ideology of their Creators." npj AI, 2025.
[20] Naous, T. and Xu, W. "On The Origin of Cultural Biases in Language Models." NAACL 2025.
[21] Rystrøm, S. et al. "Multilingual != Multicultural." 2025.
[22] Ahn, J. et al. "Impact of Language Switching on Personality Traits in LLMs." COLING 2025.
[23] Ziyin, L. et al. "Proof of a Perfect Platonic Representation Hypothesis." 2025.
[24] Wendler, C. et al. "Rethinking Cross-lingual Alignment." 2025.
[25] Shen, Y. et al. "Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?" 2025.
[26] Chen, X. et al. "Cross-model Transferability on Platonic Representations of Concepts." ACL 2025.
[27] Cassano, F. et al. "Scaling Laws for Code: Every Programming Language Matters." 2025.
[28] Meta AI. "Large Concept Models." 2024.
[29] SONAR. "Sentence-Level Multimodal and Language-Agnostic Representations." Meta AI, 2024.
[30] Cheung, A. "LLM-Based Code Translation Needs Formal Compositional Reasoning." UC Berkeley, 2025.
[31] Hu, J. et al. "Manner Implicatures in Large Language Models." Scientific Reports, 2024.
[32] Mancuso, P. et al. "An Information-Geometric View of the Platonic Hypothesis." NeurIPS 2025 Workshop.
[33] Zhong, Y. et al. "Language Lives in Sparse Dimensions." 2025.
[34] IntentCoding. "Amplifying User Intent in Code Generation." 2026.
[35] CultureManager. "Mind the Gap in Cultural Alignment." 2026.
[36] Hindle, A. et al. "On the Naturalness of Software." ICSE 2012.
[37] Allamanis, M. et al. "A Survey of Machine Learning for Big Code and Naturalness." ACM Computing Surveys, 2018.
[38] Scott, D. and Strachey, C. "Toward a Mathematical Semantics for Computer Languages." 1971.
[39] Cousot, P. and Cousot, R. "Abstract Interpretation." POPL 1977.
[40] Manhaeve, R. et al. "DeepProbLog." NeurIPS 2018.
[41] Li, Z. et al. "Scallop: A Language for Neurosymbolic Programming." PLDI 2023.
[42] Shannon, C. "Coding Theorems for a Discrete Source with a Fidelity Criterion." 1959.
[43] Petrov, A. et al. "Language Model Tokenizers Introduce Unfairness Between Languages." NeurIPS 2024.
[44] Xue, L. et al. "ByT5: Towards a Token-Free Future." TACL, 2022.
[45] Clark, J. H. et al. "CANINE: Tokenization-Free Encoder." TACL, 2022.
[46] "Beyond Syntax: How Do LLMs Understand Code?" ICSE 2025 NIER.
[47] Macfarlane, J. and Bonnet, C. "Latent Program Network." NeurIPS 2025 (Spotlight).
[48] Hao, S. et al. "COCONUT: Reasoning in Continuous Latent Space." 2024.
[49] Chen, X., Song, D., and Tian, Y. "LaSynth: Latent Execution for Neural Program Synthesis." NeurIPS 2021.
[50] Meta FAIR. "Code World Models." 2025.
[51] Piantadosi, S. T., Tily, H., and Gibson, E. "The Communicative Function of Ambiguity in Language." Cognition, 2012.
[52] "Vibe Coding as a Reconfiguration of Intent Mediation." 2025.
[53] "Reflections on the Reproducibility of Commercial LLM." ICSE 2026.
[54] Huang, J. and Yang, Z. "Lobster: GPU-Accelerated Neurosymbolic Framework." ASPLOS 2026.
[55] Lu, X. et al. "Learning from the Ubiquitous Language: Emoji Usage." UbiComp 2016.
[56] Barbieri, F. et al. "How Cosmopolitan Are Emojis?" ACM Multimedia 2016.
[57] Phillipson, R. "Linguistic Imperialism." Oxford University Press, 1992.
[58] Matthias, A. "The Responsibility Gap." Ethics and Information Technology, 2004.
[59] Durmus, E. et al. "Measuring Representation of Subjective Global Opinions in LMs." NeurIPS 2023.
[60] Nguyen, T. T. H. et al. "Survey of Post-OCR Processing Approaches." ACM Computing Surveys, 2021.

---

## Appendix A: Detailed Code Examples

Our running example across the three waves:

**Wave 1** (Han — Korean Rust):
```
함수 상위셋(목록: 배열<정수>) -> 배열<정수> {
    정렬(목록, 역순=참)[..3]
}
```

**Wave 2** (CoRE-style):
```
TASK: 상위 세 값 추출
INPUT: 목록 (정수 배열)
STEPS:
  1. 목록을 내림차순 정렬
  2. 처음 세 원소 반환
OUTPUT: 정수 배열
```

**Wave 3** (LLM-mediated):
```
User: "주어진 목록에서 가장 큰 값 세 개를 찾아라"
LLM → sorted(lst, reverse=True)[:3]  # Run 1
LLM → heapq.nlargest(3, lst)         # Run 2
LLM → [x for x in lst if x >= sorted(lst)[-3]]  # Run 3 (buggy)
```

**The RAG parallel.** RAG maps NL queries to documents via vector similarity; NL→Code maps intent to execution. The critical difference: RAG tolerates cosine similarity 0.95; code requires exact semantics (0.95 is a bug).

---

## Appendix B: Training Distribution Dependence of Z

Z = Z(scale, D_train). If D_train is culturally skewed (English 40–90% of tokens), Z inherits dominant-culture priors. Even "purely computational" domains carry implicit cultural defaults:

- **Sort order**: Korean lexicographic order differs from English.
- **Null handling**: SQL NULL semantics vary across databases, reflecting design cultures.
- **Number formatting**: "큰 값" activates different thresholds by cultural context.
- **Error semantics**: Python exceptions vs. Go explicit errors vs. C undefined behavior.

Define training distribution divergence Δ_D(Z₁, Z₂) = ||Z(s, D₁) - Z(s, D₂)|| at fixed scale. PRH claims Δ_D → 0 as s → ∞, but the Aristotelian critique suggests geometric agreement may vanish while topological agreement persists.

**Running example.** Model A (80% English D_train) maps "가장 큰 값 세 개를 찾아라" to a Python-idiomatic representation (sorted + slice). Model B (50% Korean D_train) may weight "가장 큰" with additional Korean business/academic connotations ("most significant" vs. "numerically largest"). The models may agree on topology (this is a "select top-k" operation) while disagreeing on geometry (implicit comparison criterion).

---

## Appendix C: Tokenization Bottleneck and Spacing Analysis

The mapping E : surface_form → Z is mediated by tokenization. Spacing conventions leak through:

| Type | Languages | Tokenization Behavior |
|------|-----------|----------------------|
| Space-delimited, stable | English, Spanish | BPE effective |
| Complex spacing rules | Korean (띄어쓰기) | Spacing errors common |
| No word boundaries | Chinese, Japanese, Thai | Requires segmentation model |
| Agglutinative | Turkish, Finnish | Aggressive subword splitting |

**Korean spacing variants** (all express identical intent):
```
(a) "주어진 목록에서 가장 큰 값 세 개를 찾아라"  (correct)
(b) "주어진목록에서 가장큰값 세개를 찾아라"      (informal)
(c) "주어진 목록 에서 가장 큰값 세개 를 찾아라"  (errors)
(d) "주어진목록에서가장큰값세개를찾아라"          (no spacing)
```

Critically, **all four are readable by Korean speakers** — Korean's agglutinative morphology (조사, 어미) provides structural cues. English "findthethreelargestvalues" is much harder — English relies on spaces as the primary word boundary signal. Spacing removal degrades information asymmetrically: Korean loses fluency but preserves parsability; English loses both.

**English is not immune.** OCR on justified text, hyphenated line breaks ("pro-\ngramming"), multi-column layouts, and degraded scans all produce spacing-degraded English input.

**Token tax** (Petrov et al., 2024): same semantic content requires more tokens in non-Latin languages (~1.6× Korean, ~2.5× Thai vs. English). More tokens = more opportunity for representation drift.

**Prediction 7 protocol.** Take 50 Korean computational descriptions, generate 4 spacing variants each, embed through SONAR / BPE-based / byte-level models. Measure R_spacing = d_semantic / d_spacing. Byte-level models (ByT5, CANINE) should exhibit higher R_spacing than BPE-based models.

**The ideal encoder behaves like a Korean speaker** — using morphological knowledge to reconstruct boundaries from spaceless text, not depending on surface whitespace.

---

## Appendix D: Symbols, Emoji, and the Spectrum of Representational Directness

Between NL and Z exists a spectrum:

| Surface Form | Example | Cultural Dep. | Ambiguity |
|-------------|---------|:---:|:---:|
| NL sentence | "가장 큰 값 세 개를 찾아라" | High | High |
| Emoji sequence | 🔝3️⃣📊 | Medium | Medium |
| Math notation | max₃(L) | Low | Low |
| Code | `sorted(L)[-3:]` | None | None |
| Z | E("top 3") ∈ ℝ^d | None (target) | None (target) |

**Emoji as Z_semantic experiments.** Unicode emoji encode concepts without linguistic structure, bypassing the Chomsky hierarchy entirely. The Unicode Consortium's standardization is a manually curated Z_semantic → surface-form projection: 🔥 = "fire" = "불" = "fuego" = "火."

**But emoji exhibit Z stratification.** 📊 = "chart" (near-universal Z_sem). 🙏 = prayer/gratitude/high-five (culturally divergent Z_proc). 👍 = positive/offensive (culturally specific Z_prag). Cross-cultural emoji studies confirm: sentiment ratings differ by up to 3 points across cultures (Lu et al., 2016; Barbieri et al., 2016).

**Emoji composition parallels compositionality.** ZWJ sequences (👩+💻=👩‍💻) are partially systematic but break for complex compositions — mirroring the NL compositionality problem.

**Mathematical symbols achieved Z_semantic convergence before neural networks** — through millennia of standardization. `>` means "greater than" in every tradition. PRH predicts neural networks automate this convergence.

---

## Appendix E: Ethical Implications of D_train-Dependent Z

Five dimensions of ethical risk:

**1. Computational colonialism.** English-dominant D_train defines what computation means globally. Non-English computational traditions are represented as *deviations* from this baseline. The parallel to linguistic imperialism (Phillipson, 1992) is direct.

**2. The "purely computational" boundary is culturally situated.** Is privacy a computational concern or cultural one? Is fairness in sorting a computational property? Is "fail loudly" vs. "fail silently" purely computational? What counts as "purely computational" varies across traditions — the claim that Z_semantic is culture-invariant may itself be culturally situated.

**3. Consent and transparency.** Z-level cultural opacity has no inspection mechanism — assumptions are distributed across billions of parameters. Users cannot see which cultural priors were applied.

**4. Accountability gap.** Model developer, user, and system each lack responsibility for culturally misaligned interpretation — a three-way "responsibility gap" (Matthias, 2004) specific to cultural interpretation.

**5. Feedback loop risks.** Z-projected outputs re-enter D_train → cultural analogue of model collapse (Shumailov et al., 2024): computational approach diversity narrows around initial D_train cultural modes.

**Mitigation paths.** (M1) Cultural auditing of Z. (M2) Explicit Z(c) parameterization. (M3) Pluralistic D_train via sovereign AI initiatives. (M4) D_train transparency ("nutrition labels" for models). (M5) Representation-level auditing: same NL input in multiple languages → verify Z_semantic agreement.

---

## Appendix F: Pilot Experiment Design (Full Protocol)

**Stimuli.** 50 computational operations (sort, filter, mean, reverse, max, count, ...) + 50 judgment operations (prioritize, evaluate, summarize, assess risk, ...). Each in 5 languages (English, Korean, Chinese, Arabic, Spanish) by native-speaker computational linguists. Total: 500 descriptions.

**Models.** SONAR (sentence-level, 200 languages), StarCoder-2 (code-specialized), frontier model embeddings (Claude/GPT-4 via probing).

**Protocol.**
1. Embed all 500 descriptions.
2. d_intra(o_i) = mean pairwise cosine distance among 5 language embeddings of operation o_i.
3. d_inter(l_k) = mean pairwise cosine distance among 100 operation embeddings in language l_k.
4. R = d_inter / d_intra. R > 1 means cross-lingual same-operation similarity > within-language different-operation similarity.

**Predictions.** P1: R increases with model scale. P2: R_C > R_J. Stratification: d_intra ≈ 0 for computational ops, d_intra > 0 for judgment ops.

**Controls.** Scrambled descriptions (lower bound), Google Translate vs. native (translation artifacts), within-language paraphrases (upper bound).

**Extension P6 (D_train).** Compare GPT-4 (English D_train), HyperCLOVA X (Korean), Qwen (Chinese). Cross-model divergence for implicit-criteria operations should exceed divergence for explicit-criteria operations.

**Extension P7 (Spacing).** Korean: 4 spacing variants × 100 operations = 400. Chinese: 2 segmentation variants (jieba vs. pkuseg). R_spacing = d_semantic / d_spacing. Compare BPE vs. byte-level models.

**Extension (Voice).** Whisper transcriptions of spoken descriptions (no spacing, no punctuation) as extreme spacing-robustness test.

**Feasibility.** No new models needed — existing APIs only. Cost: <$1,500 with all extensions. Timeline: 3–4 weeks.
