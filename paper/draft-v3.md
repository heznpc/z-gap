# Beyond the Chomsky Wall: Platonic Representations as the Convergence Point of Natural Language and Code

> Draft v3 — 2026-03-18
> Target: COLM 2027 or ICML 2027 Workshop
> Format: 8 pages + references

> **Changes from v2:** Added formal definitions (§3.2), falsifiable predictions (§3.7), evaluation criteria (§5.5), disambiguation migration analysis (§4.3), verification paradox (§6.6), formal taxonomy table (§2.5). Tightened abstract, compressed §2 and §6, fixed incomplete references.

---

## Abstract

Programming languages use context-free (Type-2) grammars because deterministic parsing requires them — not by convention but by mathematical necessity. Natural language is context-sensitive (Type-1 or beyond) and cannot be deterministically parsed. Three waves of natural-language programming — keyword substitution, structured NL, and LLM-mediated translation — all operate at the surface-form level and inherit this constraint. We argue that the Platonic Representation Hypothesis (Huh et al., ICML 2024) reframes the problem: if NL and code converge toward a shared latent structure Z in representation space, the Chomsky wall constrains only surface-form projections, not representations themselves. We propose that Z is stratified — Z_semantic (computational result) converges cross-culturally while Z_procedural (derivation path) and Z_pragmatic (communicative frame) remain culturally mediated — and that multilingual LLMs behave as code-switching polyglots, not Platonic oracles. We formalize these claims, derive five falsifiable predictions, and propose an evaluation framework for representation-level program execution. The disambiguation cost currently borne by Type-2 grammars does not vanish — it migrates to the representation level, creating both opportunities and a verification paradox that constrains the deployment path.

---

## 1. Introduction

### 1.1 The Persistence of Formal Syntax

Every mainstream programming language — from FORTRAN (1957) to Rust (2015) — uses a context-free grammar. This is not arbitrary. Compilers require deterministic parsing: the same source must produce the same parse tree every time. Context-free grammars (Chomsky Type-2) guarantee this in O(n) to O(n³) time. Natural language does not — and cannot, because identical sentences carry different meanings depending on context.

This constraint has shaped the entire history of programming language design. Even languages that appear "natural" — Python's `if x > 5:`, SQL's `SELECT * FROM users WHERE age > 5` — are carefully engineered to remain within Type-2 bounds. They read like English, but they parse like formal languages.

### 1.2 The Three Waves of Natural Language Programming

Attempts to make programming "natural" have come in three waves, each operating at the surface-form level:

**Wave 1: Keyword substitution (1960s–present).** Replace English tokens with native-language equivalents. COBOL's `ADD X TO Y GIVING Z` (1959), Korean languages like 씨앗 (1994) and Han (2025). The grammar remains context-free; only the terminal symbols change.

**Wave 2: Structured natural language (2024–present).** Allow prose-like expressions within constrained templates. CoRE (COLM 2025), SudoLang, Shire, Spec-Driven Development. All impose structure that renders the input parseable — effectively creating new formal languages that look like English.

**Wave 3: LLM-mediated translation (2023–present).** Free-form NL → LLM → generated code → execution. Claude Code, Cursor, Lovable. The dominant paradigm as of 2026 ("vibe coding"). It works, but the LLM acts as a non-deterministic compiler: the same input can produce different outputs.

### 1.3 The QWERTY Analogy

The QWERTY keyboard was designed in the 1870s to prevent typewriter jams. The mechanical constraint vanished with electric typewriters, then computers — but the layout persisted. Programming language syntax may be in a similar position: the original constraint (deterministic parsing) produced context-free grammars; LLMs have introduced a runtime that can process context-sensitive input; but the entire toolchain is built on formal syntax.

The question is whether LLMs merely relax the constraint (like an electric typewriter still using QWERTY) or eliminate it (like voice input bypassing the keyboard).

### 1.4 This Paper

We argue that the Platonic Representation Hypothesis offers a framework for understanding this question — and why the answer may be neither. Instead:

- Natural language and code are both projections of a shared latent structure (Z)
- The Chomsky hierarchy describes relationships between surface forms, not between representations
- LLMs operate at the representation level, where the Type-1/Type-2 distinction is irrelevant
- The productive direction is not to make NL parseable (Wave 1–2) or translate it stochastically (Wave 3), but to execute at the representation level directly

**Contributions:**
1. A taxonomy of natural-language programming approaches organized by which level (surface vs. representation) they operate on, with a formal comparison (§2)
2. A semi-formal framework: definitions of Z, representational convergence, Z stratification, and the Chomsky wall as a surface-form constraint (§3.2)
3. Analysis of PRH applied to code–language divide, engaging with the formal proof (Ziyin et al., 2025) and the Aristotelian critique (Gröger et al., 2026) (§3)
4. Stratification of Z into Z_semantic, Z_procedural, Z_pragmatic, with evidence for differential convergence (§3.4–3.5)
5. Analysis of disambiguation cost migration from grammar to representation level, and the resulting verification paradox (§4.3, §6.6)
6. Five falsifiable predictions derived from the framework (§3.7)
7. Evaluation criteria and metrics for representation-level execution (§5.5)
8. A research agenda grounded in existing systems (LCM/SONAR) and six open problems (§5)

---

## 2. Current Landscape: Surface-Form Approaches

### 2.1 Keyword Substitution Languages

| Language | Year | Base Language | Mechanism |
|----------|------|--------------|-----------|
| COBOL | 1959 | — | English-like verbs (`ADD`, `MOVE`, `PERFORM`) |
| 씨앗 (Siat) | 1994 | C-like | Korean keywords, sentence-ending with `.` |
| 아희 (Aheui) | 2005 | Befunge-like | Korean jamo-based 2D esoteric language |
| Nuri | 2020 | Haskell-like | Korean functional syntax |
| Han | 2025 | Rust/LLVM | Korean keywords (`함수`, `만약`, `변수`) |

The grammar is isomorphic to an existing formal language. The parse tree does not change. These languages cannot express anything their base language cannot — they are syntactic sugar over Type-2 grammars.

### 2.2 Structured Natural Language Systems

**CoRE / AIOS (Rutgers, COLM 2025)** unifies NL, pseudocode, and flowcharts with explicit delimiters. **SudoLang** mixes NL expressions with formal constructs for LLM execution. **Shire (UnitMesh/Phodal)** interprets NL instructions within an IDE context. **Spec-Driven Development** (GitHub Spec Kit, Kiro, Tessl) makes NL specifications the source of truth.

All impose structure on NL to make it interpretable. The moment structure is imposed, a new formal grammar is born — and the Chomsky level resets to Type-2. **The paradox:** making NL precise enough to execute requires removing precisely the properties that make it natural.

### 2.3 LLM-Mediated Translation

The user writes free-form NL; the LLM generates executable code. Key systems: Claude Code, Cursor, Lovable, Replit, MetaGPT, Devin.

**What works:** For exploration, prototyping, and one-off tasks, this is remarkably effective.

**What doesn't:** (a) Non-determinism — same prompt → different code across sessions; the "stochastic prior lock-in" problem (Ploidy, 2026). (b) Verification gap — the MoltBook incident (2026) demonstrated that vibe-coded services can leak 1.5M API keys. (c) Semantic non-preservation — Cheung (UC Berkeley, 2025) formalizes that LLMs translate *syntax* but not *semantics*: translated programs may pass surface tests while violating invariants.

### 2.4 The RAG Parallel

```
RAG:          NL query → vector similarity → relevant docs → LLM synthesis
NL→Code:      NL intent → [???] → executable result
```

Both map NL to structured output via learned representations. The critical difference: RAG tolerates approximate matches (cosine similarity 0.95 is fine), but code requires exact semantics (0.95 is a bug). This is why Wave 3 works for exploration but fails for production.

### 2.5 A Formal Comparison

| Dimension | Wave 1: Keyword Sub. | Wave 2: Structured NL | Wave 3: LLM Translation | Proposed: Z-Level |
|-----------|---------------------|-----------------------|------------------------|-------------------|
| **Chomsky level** | Type-2 (unchanged) | Type-2 (re-imposed) | Type-1 input → Type-2 output | Type-1 input → Z → Type-2 output |
| **Determinism** | Full | Full | Non-deterministic | Deterministic (target) |
| **Expressiveness** | = base language | < natural language | ≈ natural language | = natural language |
| **Verification** | Standard (Type-2) | Standard (Type-2) | None (stochastic) | At Z (open problem) |
| **Cultural adaptation** | Terminal symbols only | Template structure | Implicit in LLM | Explicit via Z(c) |
| **Disambiguation** | By programmer (grammar) | By programmer (template) | By LLM (non-deterministic) | By system (at Z) |
| **What changes** | Keywords | Surface syntax | Generation process | Execution level |

---

## 3. The Platonic Representation Hypothesis

### 3.1 Overview

Huh et al. (ICML 2024) present evidence that neural networks trained on different data, objectives, and modalities converge toward a shared statistical model of reality in their representation spaces. The hypothesis posits a "platonic" ideal representation — a shared latent structure (Z) of which images, text, and other modalities are projections.

```
Vision data (X) ────┐
                     ├───→  Shared latent structure (Z)  ← "Platonic ideal"
Language data (Y) ──┘
Code data (?) ──────┘
```

Ziyin et al. (2025) prove that embedded deep linear networks trained with SGD become "Perfectly Platonic" — every pair of layers learns the same representation up to rotation. Crucially, they identify six conditions that break convergence (weight decay, label transformation, saddle convergence, input heterogeneity, gradient flow vs. SGD, finite-step edge-of-stability), suggesting PRH convergence is the *default* for well-behaved training but can fail under specific conditions.

### 3.2 Formal Framework

We introduce semi-formal definitions to ground the analysis.

**Definition 1 (Modality Encoder).** For a data modality M_i (natural language, code, images, etc.), an *encoder* E_i : M_i → ℝ^d maps inputs to a d-dimensional representation space. A *decoder* D_i : ℝ^d → M_i maps representations back to the modality.

**Definition 2 (Representational Convergence).** Encoders E_1, ..., E_k exhibit *representational convergence* if, for semantically equivalent inputs x ∈ M_i and y ∈ M_j:

    lim_{scale→∞} sim(E_i(x), E_j(y)) → 1

where sim is a normalized similarity metric and scale encompasses model parameters, data volume, and training compute. PRH claims this convergence holds across modalities for neural networks at sufficient scale.

**Definition 3 (Shared Latent Structure).** Z is the limit space toward which encoder images converge: Z ≈ lim_{scale→∞} Im(E_i) for all modalities i. In practice, Z is approximated by the shared representation space of large-scale multimodal models (e.g., SONAR's 200-language sentence embeddings).

**Definition 4 (Z Stratification).** Z decomposes into three projections with decreasing convergence strength:

- **Z_sem : Z → S** (semantic layer) — the computational result (*what*). Cross-culturally convergent: ||ΔZ_sem||_cross-cultural < ε for purely computational domains.
- **Z_proc : Z → P** (procedural layer) — the derivation path (*how*). Culturally mediated: ||ΔZ_proc||_cross-cultural ≫ ε even for equivalent computations.
- **Z_prag : Z → Q** (pragmatic layer) — the communicative frame (*for whom*). Culturally specific: ||ΔZ_prag||_cross-cultural is maximal.

**Definition 5 (The Chomsky Wall).** The *wall* is the computational boundary between Type-2 (context-free, deterministically parseable) and Type-1 (context-sensitive, PSPACE-complete). The wall is *surface-level* if it constrains only parsing of surface forms, not operations on representations.

**Central Claim.** If PRH holds for code as a modality (Def. 2), then the Chomsky wall (Def. 5) constrains only the projection D_i : Z → M_i, not the representation Z itself. Operations at Z are not subject to Chomsky hierarchy constraints because Z is continuous, not discrete — it has no grammar to classify.

### 3.3 Evidence for Code as a Modality

While Huh et al. focus on vision and language, code fits naturally into the framework:

- **Aligned embedding spaces.** Models trained jointly on NL and code (CodeBERT, UniXcoder, StarCoder) develop representations where code snippets and NL descriptions map to nearby points. UniXcoder achieves 77.1% MRR on NL-code search across six programming languages, indicating convergence across multiple target languages.
- **Cross-model transferability.** Linear transformations align latent spaces of different LLMs, and steering vectors from smaller models (7B) control behavior in larger ones (70B) (ACL 2025). This weak-to-strong transferability is a strong PRH prediction — if representations converge toward Z, the structure of Z should be recoverable across architecturally distinct models.
- **Translation emergence.** LLMs trained primarily on NL can generate code without code-specific training, suggesting code structure is partially recoverable from NL representations. Cross-language synergies in code follow predictable scaling laws (Cassano et al., 2025).
- **Bidirectional transfer.** Copilot-style models improve code completion by training on NL documentation, and improve documentation by training on code — predicted by PRH if both modalities project from the same Z.

### 3.4 The Stratification of Z

The preceding analysis assumes Z is monolithic. But consider how different mathematical traditions derive the same theorem: the Bourbaki school proceeds via axiomatic abstraction; the Soviet tradition favors constructive methods; Indian mathematical traditions historically employed intuitive-inductive reasoning. The theorem is identical — the derivation paths differ systematically. The same applies to code: `sort(array)` produces the same output regardless of whether a developer reaches for recursion (functional tradition) or iteration (imperative tradition).

This motivates the Z stratification (Definition 4):

| Layer | Content | Convergence | Evidence |
|-------|---------|-------------|----------|
| Z_sem | What is computed | Cross-culturally convergent | Semantic Hub Hypothesis (Wu et al., ICLR 2025); shared grammatical features across diverse languages (Brinkmann et al., NAACL 2025) |
| Z_proc | How it is derived | Culturally mediated | Mathematical tradition differences; programming paradigm preferences; reasoning-language effects (Li et al., 2026) |
| Z_prag | For whom it is framed | Culturally specific | LLM pragmatic inference failures (Scientific Reports, 2024); Cultural Frame Switching (COLING 2025) |

### 3.5 Multilingual Convergence or Cultural Superposition?

PRH implies a single convergence point. Multilingual LLMs complicate this. Two models:

**Model A: The multilingual human.** The LLM code-switches between culturally-mediated representations. Z is a superposition of culturally-inflected spaces — Korean input activates a Korean-inflected Z; English input activates an Anglo-American one.

**Model B: The Platonic oracle.** The LLM contains a culture-invariant Z; cultural variations are decoding artifacts.

Evidence increasingly favors **Model A for judgment, Model B for structure:**

*Favoring Model A:*
- 9–56 percentage point moral judgment gaps across prompt languages, including preference reversals (Vida et al., AIES 2024; Agarwal, LREC-COLING 2024)
- GPT-4o displays Cultural Frame Switching — distinct "personalities" in different languages (COLING 2025)
- Reasoning-language effects contribute twice the variance of input-language effects (Li et al., 2026)
- Cultural steering vectors conserved across non-English languages can selectively activate alternative world-models (Veselovsky et al., 2025)

*Favoring Model B:*
- Abstract grammatical concepts encoded in shared feature directions across typologically diverse languages (Brinkmann et al., NAACL 2025)
- Shared semantic hub in middle layers across languages and modalities (Wu et al., ICLR 2025)
- Language identity encoded via sparse dimensions separate from semantic content (Zhong et al., 2025)

**Resolution:** Z_semantic converges (Model B); Z_procedural and Z_pragmatic remain culturally distributed (Model A). For our argument — the Chomsky wall dissolves at the representation level — Z_semantic is the operative layer. But representation-level execution cannot ignore cultural stratification in other layers.

A further refinement: the Aristotelian critique (Gröger et al., 2026) shows that after controlling for confounders, global representational convergence largely disappears. What persists is *local neighborhood agreement* — models agree on relative distances, not absolute positions. Z may be better understood as a **topological structure** (preserved neighborhoods) rather than a geometric one (preserved distances).

### 3.6 The Convergence-Determinism Gap

PRH demonstrates statistical convergence: representations of equivalent inputs become *near* each other as scale increases. But program execution requires *identity*, not proximity. `sort([3,1,2])` must yield exactly `[1,2,3]`, not "something very close to `[1,2,3]`."

This creates a fundamental gap:

```
Convergence (PRH):      sim(E_NL("sort this list"), E_code(sort_fn)) → 1 - δ
Execution requirement:   E_NL("sort this list") must deterministically yield [1,2,3]
```

The gap δ → 0 as scale increases (PRH prediction), but δ = 0 is required for exact execution. Three possible resolutions:

1. **Discretization at Z.** Partition Z into equivalence classes, each mapping to a unique execution outcome. This is formally equivalent to constructing a new formal grammar over Z — the Chomsky hierarchy returns in a new guise.
2. **Verification after projection.** Execute at Z approximately, then verify the result against formal criteria. This accepts δ > 0 at Z but demands δ = 0 at the output.
3. **Probabilistic execution with guarantees.** Accept that δ > 0 and provide statistical guarantees (e.g., "this program produces the correct output with probability 1 - 10^{-9}"). This shifts the programming paradigm from deterministic to probabilistic — viable for some domains, unacceptable for others.

The convergence-determinism gap is the central technical obstacle to representation-level execution. Our evaluation criteria (§5.5) and predictions (§3.7) are designed to track progress on closing this gap.

### 3.7 Falsifiable Predictions

The framework generates five testable predictions:

**Prediction 1 (Scale-Convergence).** As model scale increases, mean cosine distance between NL descriptions and semantically equivalent code in shared representation space decreases monotonically. *Protocol:* Compare NL-code alignment (MRR on code search) across model scales from CodeBERT (125M) → UniXcoder (350M) → StarCoder (15B) → frontier models using a fixed benchmark.

**Prediction 2 (Cross-Lingual Semantic Invariance).** For purely computational operations, cross-lingual same-operation similarity exceeds within-language different-operation similarity. "정렬하라" (Korean: sort) and "sort this list" (English) should be closer in Z than "sort this list" and "reverse this list" in the same language. This should *not* hold for judgment-involving operations. *Protocol:* Embed operation descriptions in ≥5 languages; compute intra-operation vs. inter-operation cluster distances; compare computational vs. judgment domains.

**Prediction 3 (Stratification Separability).** Z_sem and Z_proc are separable via probing classifiers: a probe detecting *what is computed* generalizes across languages; a probe detecting *how it is derived* shows language-specific patterns. *Protocol:* Train probing classifiers on intermediate representations of multilingual code generation; measure cross-lingual transfer accuracy for semantic vs. procedural probes.

**Prediction 4 (Domain-Dependent Determinism).** Constrained decoding from Z to code achieves higher consistency (same NL → same code across runs) for purely computational tasks than for judgment-involving tasks. *Protocol:* Measure output determinism of temperature-zero constrained decoding across domain types; quantify determinism gap.

**Prediction 5 (Disambiguation Migration).** As systems move from Type-2 to representation-level input, total disambiguation effort is conserved — it migrates from syntax errors to semantic clarification queries. *Protocol:* Compare the rate of user clarification queries in a representation-level prototype vs. type/syntax errors in equivalent Type-2 tasks per unit of task complexity.

---

## 4. Reinterpreting the Chomsky Wall

### 4.1 The Wall Is Real — at the Surface

The Chomsky hierarchy is a mathematical fact about formal grammars:

| Type | Grammar | Parsing Complexity | Examples |
|------|---------|-------------------|----------|
| 3 | Regular | O(n) | Regex |
| 2 | Context-free | O(n³), deterministic | Python, C, Java, SQL |
| 1 | Context-sensitive | PSPACE-complete | Natural language |
| 0 | Unrestricted | Undecidable | Turing machines |

**The wall is real for parsers.** No deterministic algorithm resolves Type-1 ambiguity in polynomial time. Programming languages are Type-2 by necessity, not convention.

**The wall is also useful.** Type-2 constraints function as a *disambiguation mechanism*: every valid expression has exactly one parse. This eliminates pragmatic ambiguity endemic to NL. The question for representation-level execution is not only "can we dissolve the wall?" but "can we achieve equivalent disambiguation without it?"

### 4.2 The Wall Dissolves — at the Representation Level

LLMs do not parse. They map input sequences to high-dimensional representations via attention over the full context. In this representation space:

- "x가 5보다 크면" (Korean NL)
- `if x > 5:` (Python)
- `(> x 5)` (Lisp)
- The abstract predicate "x exceeds 5"

...all map to nearby points (per the Semantic Hub Hypothesis, Wu et al., ICLR 2025). The Type-1/Type-2 distinction exists in the surface form, not in the representation. Language identity is encoded via sparse dimensions orthogonal to semantic content (Zhong et al., 2025), demonstrating that "what language" and "what meaning" are separable.

**Key insight:** The Chomsky hierarchy describes the computational complexity of parsing surface forms. It says nothing about the complexity of representing meaning. LLMs bypass the hierarchy not by solving Type-1 parsing but by operating in a space where parsing is unnecessary.

### 4.3 Disambiguation Migration

The Chomsky wall's utility function is *disambiguation*. Removing the wall does not remove the need for disambiguation — it **migrates** the cost.

| | Surface-form programming | Representation-level execution |
|---|---|---|
| **Who disambiguates** | Programmer, at write time | System, at Z |
| **How** | Type-2 grammar constraints | Clarification, probabilistic ranking, or verification at Z |
| **Cost** | Learning curve, syntactic overhead | Inference latency, verification complexity, Z opacity |
| **Failure mode** | Syntax error (immediate, precise) | Semantic misinterpretation (delayed, diffuse) |

We conjecture a *disambiguation conservation principle*: any system accepting ambiguous input must resolve ambiguity somewhere, and the total disambiguation cost is approximately conserved across architectures. This is a design heuristic, not a formal theorem — but it suggests that representation-level systems should not promise to eliminate programming difficulty, only to *relocate* it from syntactic precision to semantic verification.

The practical question is whether the relocation is net beneficial: is it easier for a human to learn `if x > 5:` or to verify that the system correctly interpreted "큰 값을 제외하라" ("exclude large values")? For professional developers, the current paradigm may be more efficient. For the billions of people who can express computational intent in natural language but cannot program, the tradeoff favors representation-level systems even if disambiguation cost is conserved.

### 4.4 The Analogy Completes

```
QWERTY:
  Constraint (typewriter jams) → Design (key layout) → Constraint removed → Design persists

Programming syntax:
  Constraint (deterministic parsing) → Design (CFG syntax) → Constraint removed* → Design persists

*removed at the representation level by LLMs, but NOT at the toolchain level
```

The QWERTY analogy is instructive but incomplete. QWERTY persists because of human muscle memory — a soft constraint. Programming syntax persists because of the entire toolchain (compilers, type checkers, linters, IDEs, version control, code review) — a hard infrastructure constraint.

### 4.5 Why Translation Fails but Representation Might Work

Wave 3 (LLM-mediated translation) attempts to cross the Chomsky wall by: (1) accepting NL input (Type-1), (2) translating to code (Type-2), (3) executing deterministically. The problem is in step 2: translation is non-deterministic, and it does not preserve semantics (Cheung, 2025). Semantic label drift is amplified by LLMs in culturally sensitive domains (2025). Even mathematical reasoning degrades when cultural references change (2025).

A representation-level approach would instead: (1) accept NL input, (2) map to Z, (3) execute from Z directly or project to code deterministically via constrained decoding. Step 3 is the open research problem (§5).

---

## 5. Toward Representation-Level Execution

### 5.1 What Would It Look Like?

A system that executes at the representation level would:
1. **Accept any surface form** — NL, pseudocode, spec, existing code, or a mixture
2. **Map to Z** — the shared computational intent representation
3. **Verify at Z** — check completeness and unambiguity (replacing the Type-2 grammar's disambiguation role)
4. **Execute or project** — run in representation space (simple computations) or project to deterministic executable (complex systems)

### 5.2 Existing Pieces

- **Constrained decoding** (Outlines, Guidance, LMQL): Projection from Z → Type-2, made deterministic by grammar constraints. Solves non-determinism but requires specifying the target grammar.
- **Tool use / function calling**: NL intent → structured function calls. A narrow form of Z → execution.
- **Program synthesis with verification** (AlphaCode): Multiple candidates from NL, verified against formal spec. Hybrid Z-generation / Type-2-verification.
- **CoRE's LLM-as-interpreter**: Direct execution of structured NL. Closest to representation-level execution but inherits non-determinism.
- **Meta's Large Concept Models (LCM)**: Operates on sentence-level SONAR embeddings rather than tokens, predicting the next "concept" in language-agnostic representation space. SONAR supports 200 languages and multiple modalities. Architecturally closest to the proposed pipeline: SONAR embeddings serve as a concrete Z candidate. Key limitation: LCM performs generation, not execution.

### 5.3 Open Problems

1. **Determinism at Z.** How to guarantee same NL → same execution result? PRH convergence ≠ identity (§3.6). Temperature-zero decoding helps but doesn't eliminate stochasticity.

2. **Verification at Z.** Current verification requires Type-2 input. Can we verify properties of programs in continuous embedding space? Connects to neural program verification.

3. **The grounding problem.** PRH convergence is statistical. Programming requires exact grounding: `x > 5` means exactly "strictly greater than 5." What representational precision suffices?

4. **Compositionality.** Code composes via strict nesting (call graphs); NL composes via loose reference ("do that again but with the other file"). If execution requires decomposing NL into composable sub-intents, which compositional structure governs? Complex NL specifications may compose in ways without clean projection to executable sub-programs.

5. **The specification problem.** If the user writes NL → system maps to Z, the user cannot inspect Z. This is a new opacity: not "I can't read the code" but "I can't read the intent representation."

6. **Stratification and cultural invariance.** Execution must choose which Z layer to operate on. If Z_semantic carries cultural inflection for judgment-involving domains (§3.5), the "Platonic ideal" is a family Z(c), not a single Z.

### 5.4 A Possible Architecture

```
User input (any surface form, any language)
        │
        ▼
┌──────────────────────┐
│  Encoder to Z        │  e.g., SONAR encoder (200 languages, text/speech/image)
│  NL, code, spec →    │  Maps to sentence-level concept embeddings
│  shared representation│  (language-agnostic, modality-agnostic)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Verification at Z   │  Neural verifier: is Z complete? unambiguous?
│  Disambiguation      │  Replaces Type-2 grammar's disambiguation role (§4.3)
│  at Z level          │  Handles disambiguation migration
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Direct  │ │ Project  │  Constrained decoding to Type-2 target
│ execute │ │ to code  │  with provenance links back to Z
│ at Z    │ │ (Python, │  for bidirectional debugging
│ (LCM)   │ │  Rust..) │
└─────────┘ └──────────┘
```

LCM demonstrates that the encoder and partial Z-level execution are feasible. The critical missing pieces are the verification layer and deterministic projection — corresponding to Open Problems #1–3.

### 5.5 Evaluation Criteria

To assess progress toward representation-level execution, we propose five measurable criteria:

| Criterion | Metric | Current Baseline | Target |
|-----------|--------|-----------------|--------|
| **Convergence** | Mean NL-code cosine similarity | 0.77 MRR (UniXcoder, 2022) | > 0.95 MRR |
| **Determinism** | Same NL → same output rate (temp=0) | ~0.6 (estimated, GPT-4) | > 0.99 |
| **Cross-lingual invariance** | σ of NL-code sim. across NL languages | Not yet measured | σ < 0.05 |
| **Verification at Z** | % of properties verifiable without projection | 0% (no system exists) | > 50% |
| **Disambiguation efficiency** | Clarification queries per task | N/A | ≤ syntax errors per equivalent Type-2 task |

These criteria operationalize the open problems of §5.3. The convergence and determinism baselines provide concrete benchmarks; the cross-lingual invariance metric directly tests Prediction 2 (§3.7); and the disambiguation efficiency metric tests Prediction 5.

---

## 6. Discussion

### 6.1 Is the Chomsky Wall Actually a Wall?

- **For compilers and formal methods:** The wall is absolute. Deterministic parsing for Type-1 is PSPACE-complete.
- **For LLM-based systems:** The wall is irrelevant in practice. LLMs process NL and produce coherent code without parsing.
- **For hybrid systems:** The wall becomes a design choice — accept Type-1 input, process at Z, project to Type-2 output. The wall exists at the projection boundary, not the input boundary.

### 6.2 The Python Observation

Python occupies the closest point to NL within the Type-2 boundary:

```
Natural language  ←─── increasing ambiguity ───→  Assembly
          ╔═══════════════════════════════════╗
          ║         Type-2 boundary           ║
          ║   SQL ← Python ← C ← Assembly    ║
          ╚═══════════════════════════════════╝
```

The productive question is not "can we push further toward NL within Type-2?" (diminishing returns) but "can we accept NL input and project to Type-2 output via Z?" (a different architecture).

### 6.3 Implications for Language Design

If representation-level execution becomes practical, programming languages shift from **human-writable, machine-parseable specifications** to **machine-generated, machine-verified projections from Z**. Languages become output formats — analogous to object code. Humans specify intent; the system projects to a verifiable formal language.

**The debugging objection.** If code becomes machine-generated projection, how do humans debug? Two responses: (a) debugging shifts from code to Z — the intent representation — requiring new visualization tools; (b) projected code is annotated with provenance links back to Z for bidirectional navigation. The analogy to compiled code is instructive: few debug at assembly level, yet compiled languages dominate. The debugging abstraction can shift upward again.

### 6.4 The Cultural Boundary of Z

The Chomsky wall argument assumes Z is culture-invariant. §3.4–3.5 complicate this: a domain split emerges.

- **Purely computational domains** (sorting, arithmetic, data transformation): Z_semantic is culture-invariant. Representation-level execution is most tractable here.
- **Judgment-involving domains** (risk assessment, prioritization, summarization): Z carries cultural inflection. Representation-level execution must either make the cultural frame explicit — Z(c) rather than Z — or acknowledge irreducible cultural variation.

The sovereign AI movement (Korea's five-consortium initiative, Japan's ABCI 3.0, SEA-LION) implicitly recognizes this: if Z were truly Platonic, culturally-native foundation models would be unnecessary. The market is voting for Model A.

### 6.5 Implications for Non-English Programming

The framework has specific consequences for programming in non-English contexts. Korean programmers currently face a double translation: *thought* (Korean) → *code* (English-syntax language) → *execution*. Wave 1 languages (씨앗, Han, Nuri) reduce the first translation by reskinning keywords but preserve the second — the grammar remains English-derived. Representation-level execution would eliminate both translations: Korean computational intent → Z → execution, bypassing English-syntax intermediaries entirely.

This is not merely ergonomic. Research on reasoning-language effects (Li et al., 2026) suggests that forcing reasoning through a non-native language degrades performance — the "thinking in English" overhead is real. If developers could specify computational intent in their native language without translation, the global talent pool for software development could expand by an order of magnitude — limited only by the ability to express clear intent, not by mastery of English-derived syntax.

### 6.6 The Verification Paradox

A deeper tension emerges: if verification requires formal methods, and formal methods require Type-2 input, then the system must project from Z to Type-2 for verification — reintroducing the Chomsky wall at the verification stage.

```
NL input → Z → [verify at Z? → no current tools]
                      ↓
                 project to Type-2 → verify → execute
                                     ↑
                                     wall reappears
```

The wall is dissolved at input but reassembled at verification. Near-term systems will use a **hybrid architecture**: Z provides expressiveness; Type-2 projection provides verifiability. Full representation-level execution — input, verification, and execution all at Z — requires advances in neural verification, an early-stage field.

The paradox is not fatal but constrains the deployment path: the Chomsky wall retreats inward, from the user-facing boundary to the system-internal verification layer. Progress is measured by how far inward the wall can be pushed.

### 6.7 Limitations

- PRH is a hypothesis, not a theorem. Convergence evidence is statistical, and sufficiency for exact semantics is unclear.
- "Representation-level execution" is a research direction, not a working system. The open problems in §5.3 are substantial.
- The QWERTY analogy is suggestive but not rigorous. Path dependency dynamics may differ.
- This paper is primarily a position/perspective paper. Empirical validation — especially of the falsifiable predictions (§3.7) — requires future work.
- The Z stratification (§3.4) is a conceptual framework, not an empirically validated decomposition. Whether the layers are cleanly separable in practice is open.
- The cultural invariance discussion relies primarily on behavioral evidence. Internal mechanistic evidence is still emerging.
- The disambiguation conservation conjecture (§4.3) is informal and may not hold across all task types.

---

## 7. Related Work

**Platonic Representation Hypothesis.** Huh et al. (ICML 2024) established the convergence thesis. Ziyin et al. (2025) provided a formal proof for deep linear networks, identifying six conditions breaking convergence. Gröger et al. (2026) showed global geometric convergence is confounded by scale but local neighborhood agreement persists — a topological rather than geometric interpretation. An information-geometric analysis (NeurIPS 2025 Workshop) frames convergence as Bayesian posterior concentration, proving a "disunion theorem" for models with different approximation capabilities. Cross-domain extensions reach astronomy (NeurIPS 2025), interatomic potentials (2025), and neuroscience (2025).

**Multilingual Representations and Cultural Alignment.** The Semantic Hub Hypothesis (Wu et al., ICLR 2025) demonstrates shared middle-layer representations. Brinkmann et al. (NAACL 2025) show shared grammatical feature directions across typologically diverse languages. However, multilingual ≠ multicultural (Rystrøm et al., 2025): LLMs produce different moral judgments across languages (Vida et al., 2024; Agarwal, 2024; Li et al., 2026), display Cultural Frame Switching (COLING 2025), and reflect creator ideology (Buyl et al., 2025). Veselovsky et al. (2025) identify conserved cultural steering vectors. Naous and Xu (NAACL 2025) trace Western bias to training data. The transfer-localization tradeoff (Wendler et al., 2025) formalizes cross-lingual alignment's tension with cultural preservation.

**Natural Language Programming.** Beyond systems surveyed in §2, Cheung (UC Berkeley, 2025) argues LLM code translation preserves syntax but not semantics. IntentCoding (2026) addresses intent amplification via masked decoding. Meta's LCM (2024–2025) operates on SONAR embeddings, the closest system to representation-level execution. CultureManager (2026) introduces task-specific cultural adaptation.

**Pragmatic Meaning in NLP.** LLMs perform at or below chance on manner implicatures (Scientific Reports, 2024). Semantic label drift is amplified by LLMs' cultural knowledge in translation (2025). Mathematical reasoning degrades under cultural context changes (2025).

---

## 8. Conclusion

For seventy years, programming languages have been constrained to context-free grammars — not by choice but by the mathematical requirements of deterministic parsing. Three waves of natural-language programming have attempted to bridge the gap; all operate at the surface-form level and inherit the Chomsky wall's constraints.

The Platonic Representation Hypothesis suggests this wall is a property of surface forms, not of representations. Neural networks trained on language and code converge toward shared representations where the Type-1/Type-2 distinction is irrelevant. This reframes the problem from "how do we parse NL as code?" to "how do we execute at the representation level where NL and code are the same thing?"

But the shared representation Z is not monolithic. It stratifies into Z_semantic (convergent), Z_procedural (culturally mediated), and Z_pragmatic (culturally specific). Multilingual LLMs behave as code-switching polyglots, not Platonic oracles. The Aristotelian critique refines this further: what converges is local neighborhood topology, not global geometry. Representation-level execution is most tractable for purely computational domains; judgment-involving domains require cultural parameterization — Z(c) rather than Z.

The disambiguation cost currently borne by Type-2 grammars does not vanish when the wall dissolves — it migrates to the representation level. This creates a verification paradox: near-term systems must project back to Type-2 for formal verification, pushing the wall inward rather than eliminating it. Progress is measured by how far inward the wall retreats.

We have formalized these claims, derived five falsifiable predictions, and proposed evaluation criteria. The pieces exist — SONAR embeddings, Large Concept Models, constrained decoding, cultural steering vectors, program synthesis. They have not been unified under the representation-convergence framework that PRH provides. The research agenda is to unify them — while confronting the cultural stratification of Z that determines where representation-level execution can and cannot succeed.

Like QWERTY, formal syntax may be a vestige of a transcended constraint. Unlike QWERTY, the cost of persistence is not ergonomic inefficiency but a fundamental barrier: billions of people can express computational intent in natural language but cannot program. The wall need not fall everywhere — but where Z_semantic converges, it need not stand.

---

## References

[1] Huh, M., Cheung, B., Wang, T., and Isola, P. "The Platonic Representation Hypothesis." ICML 2024. arXiv:2405.07987.

[2] Chomsky, N. "Three Models for the Description of Language." IRE Transactions on Information Theory, 1956.

[3] CoRE / AIOS. "Natural Language Is All a Computer Needs." COLM 2025. arXiv:2405.06907.

[4] Elliott, E. "SudoLang: A Powerful Pseudocode Programming Language for LLMs." 2023.

[5] Fowler, M. "Spec-Driven Development: Tools." martinfowler.com, 2025.

[6] Brooker, M. "Natural Language Programming." brooker.co.za, 2025.

[7] Keles, A. "LLMs Could Be But Shouldn't Be Compilers." alperenkeles.com, 2025.

[8] Shumailov, I. et al. "AI Models Collapse When Trained on Recursively Generated Data." Nature, 2024.

[9] Jacob, R., Kerrigan, D., and Bastos, P. "The Chat-Chamber Effect: Trusting the AI Hallucination." Big Data & Society, SAGE, 2025.

[10] Shire. "AI Coding Agent Language." phodal/shire, GitHub, 2025.

[11] Xodn348. "Han: Korean Programming Language with LLVM." GitHub, 2025.

[12] Fowler, M., and Highsmith, J. "The Agile Manifesto." 2001.

[13] David, C. "The QWERTY Keyboard." Smithsonian Magazine, 2013.

[14] Du, X. et al. "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval." EMNLP 2025.

[15] Feng, J. et al. "Anchoring Bias in Large Language Models." J. Computational Social Science, 2026.

[16] Wu, W. et al. "The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities." ICLR 2025. arXiv:2411.04986.

[17] Brinkmann, M. et al. "Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages." NAACL 2025 (Oral).

[18] Gröger, F., Wen, S., and Brbic, M. "Revisiting the Platonic Representation Hypothesis: An Aristotelian View." arXiv:2602.14486, 2026.

[19] Vida, K., Damken, F., and Lauscher, A. "Decoding Multilingual Moral Preferences: Unveiling LLM's Biases through the Moral Machine Experiment." AAAI/ACM AIES, 2024.

[20] Agarwal, U. "Ethical Reasoning and Moral Value Alignment of LLMs Depend on the Language We Prompt Them In." LREC-COLING 2024.

[21] Li, N. et al. "Untangling Input Language from Reasoning Language: A Diagnostic Framework for Cross-Lingual Moral Alignment in LLMs." arXiv:2601.10257, 2026.

[22] Veselovsky, V. et al. "Localized Cultural Knowledge is Conserved and Controllable in Large Language Models." arXiv:2504.10191, 2025.

[23] Buyl, M. et al. "Large Language Models Reflect the Ideology of their Creators." npj Artificial Intelligence, 2025.

[24] Naous, T. and Xu, W. "On The Origin of Cultural Biases in Language Models: From Pre-training Data to Linguistic Phenomena." NAACL 2025.

[25] Rystrøm, S. et al. "Multilingual != Multicultural: Evaluating Gaps Between Multilingual Capabilities and Cultural Alignment in LLMs." arXiv:2502.16534, 2025.

[26] Ahn, J. et al. "Exploring the Impact of Language Switching on Personality Traits in LLMs." COLING 2025.

[27] Lu, J. G. and Zhang, L. D. "How Two Leading LLMs Reasoned Differently in English and Chinese." Harvard Business Review, December 2025.

[28] Ziyin, L. et al. "Proof of a Perfect Platonic Representation Hypothesis." arXiv:2507.01098, 2025.

[29] Aksoy, M. "Whose Morality Do They Speak? Unraveling Cultural Bias in Multilingual Language Models." arXiv:2412.18863, 2024.

[30] Wendler, C. et al. "Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs." arXiv:2510.26024, 2025.

[31] Shen, Y. et al. "Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?" arXiv:2503.18018, 2025.

[32] Chen, X. et al. "Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts." ACL 2025.

[33] Cassano, F. et al. "Scaling Laws for Code: Every Programming Language Matters." arXiv:2512.13472, 2025.

[34] Meta AI. "Large Concept Models: Language Modeling in a Sentence Representation Space." arXiv:2412.08821, 2024.

[35] SONAR. "Sentence-Level Multimodal and Language-Agnostic Representations." Meta AI, 2024.

[36] Cheung, A. "LLM-Based Code Translation Needs Formal Compositional Reasoning." UC Berkeley EECS-2025-174, 2025.

[37] Kim, S. et al. "Semantic Label Drift in Cross-Cultural Translation." arXiv:2510.25967, 2025.

[38] Hu, J. et al. "Manner Implicatures in Large Language Models." Scientific Reports (Nature), November 2024.

[39] Mancuso, P. et al. "An Information-Geometric View of the Platonic Hypothesis." NeurIPS 2025 Workshop.

[40] Zhong, Y. et al. "Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for LLMs." arXiv:2510.07213, 2025.

[41] IntentCoding. "Amplifying User Intent in Code Generation." arXiv:2602.00066, 2026.

[42] CultureManager. "Mind the Gap in Cultural Alignment: Task-Aware Culture Management for Large Language Models." arXiv:2602.22475, 2026.
