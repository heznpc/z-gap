# Beyond the Chomsky Wall: Platonic Representations as the Convergence Point of Natural Language and Code

> Draft v2 — 2026-03-18
> Target: COLM 2027 or ICML 2027 Workshop
> Format: 8 pages + references

---

## Abstract

Programming languages have remained syntactically formal for seven decades — not because of engineering convenience, but because deterministic parsing requires context-free (Type-2) grammars, while natural language is context-sensitive (Type-1 or beyond). Recent attempts to bridge this gap fall into three categories: keyword substitution (replacing `if` with native-language equivalents), structured natural language (constraining prose into parseable templates), and LLM-mediated translation (generating code from free-form text). All three operate at the surface-form level and inherit fundamental limitations: the first two cannot escape Type-2 constraints, and the third is non-deterministic. We argue that the Platonic Representation Hypothesis (Huh et al., ICML 2024) — which demonstrates that neural networks trained on different modalities converge toward a shared statistical model of reality — reframes this problem entirely. If natural language and code are projections of the same latent structure, the bottleneck is not translating between surface forms but operating at the representation level where the Chomsky distinction dissolves. We further propose that this shared latent structure (Z) is not monolithic but stratified: Z_semantic (computational result) converges across languages and cultures, while Z_procedural (derivation path) remains culturally mediated — even in code and mathematics. Drawing on cross-lingual moral divergence studies, cultural steering vector research, and the Aristotelian critique of PRH, we argue that multilingual LLMs behave more like code-switching polyglots than Platonic oracles, with implications for which domains can support representation-level execution. We survey current natural-language programming systems, analyze why each remains bound by surface-form constraints, and propose a research agenda for representation-level program execution — drawing an analogy to the QWERTY keyboard, where the original constraint vanished but the design persisted.

---

## 1. Introduction

### 1.1 The Persistence of Formal Syntax

Every mainstream programming language — from FORTRAN (1957) to Rust (2015) — uses a context-free grammar. This is not arbitrary. Compilers require deterministic parsing: the same source must produce the same parse tree every time. Context-free grammars (Chomsky Type-2) guarantee this in O(n) to O(n³) time. Natural language does not — and cannot, because identical sentences carry different meanings depending on context.

This constraint has shaped the entire history of programming language design. Even languages that appear "natural" — Python's `if x > 5:`, SQL's `SELECT * FROM users WHERE age > 5` — are carefully engineered to remain within Type-2 bounds. They read like English, but they parse like formal languages.

### 1.2 The Three Waves of Natural Language Programming

Attempts to make programming "natural" have come in three waves, each operating at the surface-form level:

**Wave 1: Keyword substitution (1960s–present).** Replace English tokens with native-language equivalents. COBOL's `ADD X TO Y GIVING Z` (1959), Korean languages like 씨앗 (1994) and Han (2025), and recent projects like Nuri (Haskell-inspired Korean syntax). The grammar remains context-free; only the terminal symbols change. `if` becomes `만약`, but the parse tree is identical.

**Wave 2: Structured natural language (2024–present).** Allow prose-like expressions within constrained templates. CoRE (Rutgers, COLM 2025) defines control flow in quasi-natural language with explicit delimiters (`:::`). SudoLang mixes natural language with programming constructs (`for each`, `if`). Shire (UnitMesh) interprets natural-language instructions within an IDE. All impose structure that renders the input parseable — effectively creating new formal languages that look like English.

**Wave 3: LLM-mediated translation (2023–present).** Free-form natural language → LLM → generated code → execution. Claude Code, Cursor, Lovable, Open Interpreter. This is the dominant paradigm as of March 2026 ("vibe coding"). It works, but the LLM acts as a non-deterministic compiler: the same input can produce different outputs. Formal verification is impossible. The stochastic nature is a feature for exploration but a defect for production.

### 1.3 The QWERTY Analogy

The QWERTY keyboard layout was designed in the 1870s to prevent typewriter jams by separating frequently co-occurring letter pairs. The mechanical constraint vanished with electric typewriters, then computers — but the layout persisted. Dvorak (1936) is demonstrably more efficient, yet adoption is negligible. The original constraint created path dependency that outlived the constraint itself.

Programming language syntax may be in a similar position. The original constraint — deterministic parsing by finite-state automata and pushdown automata — produced context-free grammars. LLMs have introduced a runtime that can process context-sensitive (even context-free-unbounded) input. But the entire toolchain (compilers, linters, type checkers, IDEs, developer muscle memory) is built on the assumption of formal syntax.

The question is whether LLMs merely relax the constraint (like an electric typewriter that still uses QWERTY) or eliminate it (like voice input that bypasses the keyboard entirely).

### 1.4 This Paper

We argue that the Platonic Representation Hypothesis offers a framework for understanding which of these two scenarios is occurring — and why the answer may be neither. Instead:

- Natural language and code are both projections of a shared latent structure (Z)
- The Chomsky hierarchy describes relationships between surface forms, not between representations
- LLMs operate at the representation level, where the Type-1/Type-2 distinction is irrelevant
- The productive direction is not to make natural language parseable (Wave 1–2) or to translate it stochastically (Wave 3), but to execute at the representation level directly

**Contributions:**
- A taxonomy of natural-language programming approaches organized by which level (surface vs. representation) they operate on (§2)
- An analysis of the Platonic Representation Hypothesis as it applies to the code–language divide, including engagement with the formal proof (Ziyin et al., 2025) and the Aristotelian critique (Gröger et al., 2026) (§3)
- A stratification of the shared latent structure Z into Z_semantic, Z_procedural, and Z_pragmatic, with evidence that only Z_semantic converges across cultures while the other layers remain culturally mediated (§3.4–3.5)
- An analysis of multilingual LLMs as code-switching polyglots rather than Platonic oracles, with implications for domain-dependent tractability of representation-level execution (§3.5, §6.4)
- A reinterpretation of the Chomsky wall as a surface-form artifact, not a fundamental barrier (§4)
- A research agenda for representation-level program execution, grounded in existing systems (LCM/SONAR) and six open problems (§5)

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

**What they share:** The grammar is isomorphic to an existing formal language. The parse tree does not change. This is reskinning, not rethinking.

**The fundamental limitation:** These languages cannot express anything that their base language cannot. They are syntactic sugar over Type-2 grammars — the Chomsky level is unchanged.

### 2.2 Structured Natural Language Systems

**CoRE / AIOS (Rutgers, COLM 2025)** — Unifies natural language, pseudocode, and flowcharts into a single "programming language" with LLM as interpreter. Defines control structures (sequence, branching, loops) in natural-language syntax with explicit delimiters. The LLM does not generate code; it interprets the structured natural language directly.

**SudoLang** — A pseudocode language designed for LLM execution. Mixes natural-language expressions with formal constructs. Any sufficiently capable LLM can "run" SudoLang without compilation. In practice, the LLM is the runtime.

**Shire (UnitMesh/Phodal)** — An "AI coding agent language" that interprets natural-language instructions within an IDE context. Evolved from AutoDev → DevIns → Shire.

**Spec-Driven Development (2025–)** — Natural-language specifications become the source of truth. GitHub Spec Kit, Kiro, Tessl. Martin Fowler's analysis positions this as the productization of "natural language as program."

**What they share:** All impose structure on natural language to make it interpretable. CoRE uses `:::` delimiters. SudoLang uses `for each`/`if` patterns. Spec-Driven uses YAML-like templates. The moment structure is imposed, a new formal grammar is born — and the Chomsky level resets to Type-2.

**The paradox:** Making natural language precise enough to execute requires removing precisely the properties that make it natural.

### 2.3 LLM-Mediated Translation

The dominant paradigm as of 2026. The user writes free-form natural language; the LLM generates executable code.

**Key systems:** Claude Code, Cursor, Lovable, Replit, Open Interpreter, MetaGPT, Devin.

**What works:** For exploration, prototyping, and one-off tasks, this is remarkably effective. Apple's Xcode 26.3 integrates Claude and Codex for natural-language app development.

**What doesn't work:**
- **Non-determinism:** Same prompt → different code across sessions. The "stochastic prior lock-in" problem (see Ploidy, 2026) applies here — whichever code the LLM generates first anchors the project.
- **Verification gap:** The MoltBook incident (2026) demonstrated that vibe-coded services can leak 1.5M API keys because security verification was skipped.
- **Cost and latency:** Every "execution" requires a full LLM inference pass. Marc Brooker (AWS VP): "The key is the conversational feedback loop, not one-shot conversion."

### 2.4 The RAG Parallel

Retrieval-Augmented Generation (RAG) shares the same architectural pattern:

```
RAG:          NL query → vector similarity search → relevant docs → LLM synthesis
NL→Code:      NL intent → [???] → executable result
```

Both map natural language to structured output via learned representations. The critical difference: RAG tolerates approximate matches (cosine similarity 0.95 is fine for retrieval), but code execution requires exact semantics (cosine similarity 0.95 is a bug).

This difference is why Wave 3 works for exploration but fails for production — and why a representation-level approach must address determinism, not just expressiveness.

---

## 3. The Platonic Representation Hypothesis

### 3.1 Overview

Huh et al. (ICML 2024) present evidence that neural networks trained on different data, objectives, and modalities are converging toward a shared statistical model of reality in their representation spaces. As vision models and language models scale, they measure distances between datapoints in increasingly similar ways. The hypothesis posits that there exists a "platonic" ideal representation — a shared latent structure (Z) of which images (X), text (Y), and other modalities are projections.

```
Vision data (X) ────┐
                     ├───→  Shared latent structure (Z)  ← "Platonic ideal"
Language data (Y) ──┘
Code data (?) ──────┘
```

The hypothesis has received partial formal support: Ziyin et al. (2025) prove that embedded deep linear networks trained with SGD become "Perfectly Platonic" — every pair of layers learns the same representation up to rotation, driven by entropy maximization. Crucially, they also identify six conditions that break convergence: weight decay, label transformation, convergence to saddle points, input heterogeneity, gradient flow (vs. SGD), and finite-step edge-of-stability effects. These breaking conditions are important for our argument: they suggest that PRH convergence is the default for well-behaved training, but can fail under specific technical conditions — a distinction that matters when we consider whether code-NL convergence is robust enough for execution (§5.3).

### 3.2 Evidence for Code as a Modality

While Huh et al. focus on vision and language, code fits naturally into the framework — and the evidence is increasingly quantitative:

- **Aligned embedding spaces.** Models trained jointly on natural language and code (CodeBERT, UniXcoder, StarCoder) develop representations where a code snippet and its natural-language description map to nearby points. UniXcoder achieves 77.1% MRR on NL-code search across six programming languages, indicating that NL-code alignment is not language-pair-specific but converges across multiple target languages toward a shared representation.
- **Cross-model transferability.** Concept representations are not model-specific: simple linear transformations can align the latent spaces of different LLMs, and steering vectors extracted from smaller models (e.g., 7B) successfully control behavior in larger models (e.g., 70B) (ACL 2025). This weak-to-strong transferability is a strong prediction of PRH — if representations converge toward a shared Z, the structure of Z should be recoverable across architecturally distinct models.
- **Translation emergence.** LLMs trained primarily on natural language can generate code without code-specific training, suggesting that code structure is partially recoverable from natural-language representations alone. Multilingual code scaling laws further reveal that cross-language synergies exist — training on one programming language improves performance on others, with the benefit scaling predictably (2025).
- **Bidirectional transfer.** Copilot-style models improve code completion by training on NL documentation, and improve documentation by training on code. This bidirectionality is predicted by PRH: if both modalities project from the same Z, enriching one projection should improve the other.

### 3.3 What Z Looks Like for Code

If natural language and code are both projections of Z, what is Z?

For code, Z is likely the **computational intent** — the abstract specification of what should happen, stripped of:
- Surface syntax (Python's `if x > 5:` vs. C's `if (x > 5)`)
- Implementation details (loop vs. recursion vs. vectorization)
- Language-specific idioms (`list comprehension` vs. `map/filter`)

This is consistent with the observation that LLMs can translate between programming languages with high accuracy — they are operating at a level where `for item in items: process(item)` (Python) and `items.forEach(item => process(item))` (JavaScript) are the same point in Z.

### 3.4 The Stratification of Z

The preceding analysis assumes Z is monolithic — a single point in latent space capturing computational intent. But there is reason to believe Z has internal structure.

Consider how different mathematical traditions derive the same theorem. The Bourbaki school proceeds via axiomatic abstraction; the Soviet tradition favors constructive methods; Indian mathematical traditions historically employed intuitive-inductive reasoning where the Euclidean tradition demands deduction. The theorem is identical — but the intermediate representations differ systematically. The same applies to code: `sort(array)` produces the same output regardless of whether a developer reaches for recursion (functional tradition) or iteration (imperative tradition), yet these paths reflect distinct cognitive habits at the community level.

This suggests Z has at least three layers:

- **Z_semantic**: The computational result — *what* is computed. Cross-lingual embedding studies show this layer converges across modalities and languages. LLMs develop a language-agnostic "semantic hub" in middle layers (Wu et al., ICLR 2025), and abstract grammatical concepts are encoded in feature directions shared across typologically diverse languages (Brinkmann et al., NAACL 2025).

- **Z_procedural**: The computational path — *how* the result is derived. This layer appears culturally and contextually mediated, even in domains conventionally treated as culture-invariant. The choice between axiomatic vs. constructive proof, recursive vs. iterative implementation, or object-oriented vs. functional decomposition reflects community-level reasoning patterns, not purely individual preference.

- **Z_pragmatic**: The communicative intent — *why* and *for whom* the result is presented. This layer governs how information is framed, what is emphasized or hedged, and which social conventions shape the output. A direct answer in German may require hedging in Japanese; a summary for a Western audience foregrounds different aspects than one for an East Asian audience. LLMs fail at pragmatic inference at a basic level — performing at or below chance on manner implicatures (Scientific Reports, 2024) — suggesting that Z_pragmatic is the least well-captured layer.

| | Result (what) | Process (how) | Framing (for whom) |
|---|---|---|---|
| Code/Math | Culture-invariant | **Culture-dependent** | **Culture-dependent** |
| Natural Language | Culture-dependent | Culture-dependent | **Culture-dependent** |

The three layers have decreasing degrees of cross-cultural convergence: Z_semantic converges robustly, Z_procedural partially, Z_pragmatic minimally. The Chomsky wall argument (§4) depends primarily on Z_semantic, where convergence is strongest.

### 3.5 Multilingual Convergence or Cultural Superposition?

PRH implies a single convergence point. But multilingual LLMs complicate this picture. Two models:

**Model A: The multilingual human.** The LLM has internalized multiple cultural-cognitive frameworks and code-switches between them depending on input language. Z is not a single Platonic ideal but a superposition of culturally-mediated representations — Korean input activates a Korean-inflected Z; English input activates an Anglo-American one.

**Model B: The Platonic oracle.** The LLM contains a culture-invariant Z, and cultural variations in output are artifacts of the decoding stage. Training data biases are noise, not signal.

Empirical evidence increasingly favors Model A for outputs involving human judgment:

- LLMs produce measurably different moral judgments depending on prompt language, with 9–56 percentage point gaps including complete preference reversals (Vida et al., AIES 2024; Agarwal, LREC-COLING 2024).
- GPT-4o displays Cultural Frame Switching analogous to human bilinguals — distinct "personalities" surface in different languages (COLING 2025).
- Reasoning-language effects contribute twice the variance of input-language effects, indicating the model's internal processing path is itself language-dependent (Li et al., 2026).
- Cultural knowledge is conserved inside LLMs but does not surface spontaneously; "cultural steering vectors" conserved across non-English languages can selectively activate alternative cultural world-models (Veselovsky et al., 2025).

However, for structural/formal representations, Model B finds support:

- Abstract grammatical concepts are encoded in shared feature directions across typologically diverse languages, even in English-dominant models (Brinkmann et al., NAACL 2025).
- LLMs develop a shared semantic hub in middle layers, processing multilingual input through a language-agnostic conceptual space (Wu et al., ICLR 2025).
- Language identity is encoded via a small, sparse set of dimensions separate from semantic content, enabling language switching while preserving meaning (2025).

The resolution aligns with the Z_semantic/Z_procedural distinction: Z_semantic converges (Model B), while Z_procedural and the pragmatic layer remain culturally distributed (Model A). For this paper's argument — that the Chomsky wall dissolves at the representation level — it is Z_semantic that matters. But a complete account of representation-level execution cannot ignore the cultural stratification of other layers.

A further refinement comes from the Aristotelian critique of PRH (Gröger et al., 2026): after controlling for confounders (model width, depth), the global representational convergence reported by Huh et al. largely disappears. What persists is local neighborhood agreement — models agree on *who is near whom* in representation space, even when global geometry diverges. This suggests Z may be better understood as a topological structure (preserved neighborhoods) rather than a geometric one (preserved distances): a shared relational fabric, not a shared coordinate system.

### 3.6 The Convergence Prediction

PRH predicts that as models scale:
1. Code representations and NL representations become more aligned
2. The "translation cost" between NL and code decreases in representation space
3. Eventually, the distinction between "understanding NL" and "understanding code" dissolves at the model level

This is already partially observed: Claude, GPT-4, and Gemini do not have separate "code modes" — they process code and language in the same transformer, with shared representations.

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

Natural language is at least Type-1: the meaning of "큰 값이 5보다 크면" depends on whether "큰" modifies "값" or introduces a separate condition. This ambiguity is not a bug in Korean — it is a structural property of context-sensitive grammars. English has the same issue: "Time flies like an arrow" has at least three valid parses.

**The wall is real for parsers.** No deterministic algorithm can resolve Type-1 ambiguity in polynomial time. This is why programming languages are Type-2 — it's not convention but necessity for deterministic compilation.

**The wall is also useful.** Type-2 constraints are not merely a technical limitation — they function as a *disambiguation mechanism*. By forcing authors to write in a context-free grammar, programming languages guarantee that every valid expression has exactly one parse. This eliminates the pragmatic ambiguity endemic to natural language. The question for representation-level execution is therefore not only "can we dissolve the wall?" but "can we achieve equivalent disambiguation without it?" Any system that accepts NL input and executes at Z must solve the disambiguation problem that Type-2 grammars currently solve for free — either through interactive clarification, probabilistic intent ranking, or formal verification at Z (§5.3, Open Problem #5).

### 4.2 The Wall Dissolves — at the Representation Level

But LLMs do not parse. They do not construct parse trees. They map input sequences to high-dimensional representations via attention over the full context. In this representation space:

- "x가 5보다 크면" (Korean NL)
- `if x > 5:` (Python)
- `(> x 5)` (Lisp)
- The abstract predicate "x exceeds 5"

...all map to nearby points. The Type-1/Type-2 distinction exists in the surface form, not in the representation.

This is not merely a theoretical claim. The Semantic Hub Hypothesis (Wu et al., ICLR 2025) provides direct empirical evidence: LLMs develop a shared representation space in their middle layers where semantically equivalent inputs across languages, code, arithmetic, and even visual modalities converge. In English-dominant models processing Korean or Chinese, the middle layers are dominated by English-aligned representations; the target language only reasserts dominance in the final layers. The implication is striking: the model's internal "thinking" occurs in a language-agnostic space — precisely the space where the Chomsky hierarchy is irrelevant. Language-specific surface form is a late-stage projection, not a deep-level constraint.

Furthermore, language identity itself is encoded via a small, sparse set of dimensions orthogonal to semantic content (2025). Manipulating these dimensions enables language switching while preserving meaning — demonstrating that "what language" and "what meaning" are separable in representation space.

**Key insight:** The Chomsky hierarchy describes the computational complexity of parsing surface forms. It says nothing about the complexity of representing meaning. LLMs bypass the hierarchy not by solving Type-1 parsing but by operating in a space where parsing is unnecessary.

### 4.3 The Analogy Completes

```
QWERTY:
  Constraint (typewriter jams) → Design (key layout) → Constraint removed → Design persists

Programming syntax:
  Constraint (deterministic parsing) → Design (CFG syntax) → Constraint removed* → Design persists

*removed at the representation level by LLMs, but NOT at the toolchain level
```

The QWERTY analogy is instructive but incomplete. QWERTY persists because of human muscle memory — a soft constraint. Programming syntax persists because of the entire toolchain (compilers, type checkers, linters, IDEs, version control diffs, code review) — a hard infrastructure constraint. Changing syntax requires changing every tool in the chain.

### 4.4 Why Translation Fails but Representation Might Work

Wave 3 (LLM-mediated translation) attempts to cross the Chomsky wall by:
1. Accepting NL input (Type-1)
2. LLM translates to code (Type-2)
3. Code is executed deterministically

The problem is in step 2: translation is non-deterministic. The same NL input produces different code across runs. This is because the LLM is mapping from a higher-entropy space (NL) to a lower-entropy space (code), and there are multiple valid projections. Cheung (UC Berkeley, 2025) formalizes a deeper issue: LLMs translate *syntax* but do not preserve *semantics*. A translated program may look correct — passing surface-level tests — while violating internal contracts, invariants, or non-functional properties (memory safety, timing). Principled compositional reasoning is required for trustworthy translation, but Wave 3 systems do not provide it.

The problem compounds across languages and cultures. Semantic label drift studies (2025) show that LLM translation amplifies cultural distortion in culturally sensitive domains — and counterintuitively, leveraging the model's cultural knowledge makes drift *worse*, not better. Even in mathematics, LLMs show performance degradation when cultural references in word problems change, despite identical underlying structure (2025). If NL → code translation cannot preserve meaning even for math, the gap for production software is substantial.

A representation-level approach would instead:
1. Accept NL input (Type-1)
2. Map to shared representation Z (neither Type-1 nor Type-2 — it's continuous)
3. Execute from Z directly, or project to code deterministically via constrained decoding

Step 3 is the open research problem. Constrained decoding (forcing the LLM to produce only syntactically valid code) is one approach; neurosymbolic execution (running programs in representation space) is another.

---

## 5. Toward Representation-Level Execution

### 5.1 What Would It Look Like?

A system that executes at the representation level would:

1. **Accept any surface form** — natural language, pseudocode, structured spec, existing code, or a mixture
2. **Map to Z** — the shared computational intent representation
3. **Verify at Z** — check that the representation is unambiguous and complete (not the surface form)
4. **Execute or project** — either run the program in representation space (for simple computations) or project to a deterministic executable (for complex systems)

### 5.2 Existing Pieces

Several existing systems implement fragments of this vision:

- **Constrained decoding** (Outlines, Guidance, LMQL): Forces LLM output to conform to a formal grammar. This is projection from Z → Type-2, made deterministic by grammar constraints. It solves the non-determinism problem of Wave 3 but still requires specifying the target grammar.

- **Tool use / function calling**: The LLM maps NL intent to structured function calls with typed parameters. This is a narrow form of Z → execution: the representation level "understands" the intent, and the projection is to a well-defined API.

- **Program synthesis with verification** (LLMLift, AlphaCode): Generate multiple code candidates from NL, then verify each against a formal specification. This uses Z for generation and Type-2 for verification — a hybrid approach.

- **CoRE's LLM-as-interpreter**: The LLM directly executes structured natural language without generating intermediate code. This is the closest to representation-level execution, but it inherits the LLM's non-determinism.

- **Meta's Large Concept Models (LCM)**: Perhaps the most significant existing fragment. LCM operates on sentence-level embeddings in SONAR space rather than tokens, predicting the next "concept" — a language-agnostic, modality-agnostic representation. SONAR supports 200 languages and multiple modalities (text, speech, images). LCM enables zero-shot cross-lingual and cross-modal generation without ever passing through a language-specific tokenization step. This is architecturally close to the "Encoder to Z → Execute/Project" pipeline proposed in §5.4: SONAR embeddings serve as a concrete candidate for Z, and LCM demonstrates that meaningful computation (next-concept prediction) can occur at the representation level. The key limitation is that LCM performs generation, not execution — it predicts what comes next, not what a program should do.

### 5.3 Open Problems

1. **Determinism at Z.** How do you guarantee that the same NL input always produces the same execution result? PRH shows representations converge, but convergence ≠ identity. Temperature-zero decoding helps but doesn't eliminate stochasticity from attention mechanisms.

2. **Verification at Z.** Current formal verification requires Type-2 (or at most Type-1) input. Can we verify properties of programs represented in continuous embedding space? This connects to neural program verification, an early-stage field.

3. **The grounding problem.** PRH shows that representations converge statistically. But programming requires exact grounding: `x > 5` must mean exactly "x is strictly greater than 5", not "x is approximately 5 or maybe a bit more." How much representational precision is sufficient for exact execution?

4. **Compositionality.** Natural language and code are both compositional, but their compositional structures differ in ways that may not align in Z. Code composes via strict nesting: a function calls a function that calls a function, and the semantics are determined entirely by the call graph. NL composes via loose reference: "do that again but with the other file" relies on discourse context, anaphora, and shared pragmatic knowledge that has no direct analogue in code's call graph. Consider: "Sort the users by age, then filter out minors, and email the rest" is a pipeline of three operations — compositionally clean. But "Handle the edge cases the way we discussed last time" composes an operation with a pragmatic reference that exists in Z_pragmatic, not Z_semantic. If representation-level execution requires decomposing NL intent into composable sub-intents, the system must resolve which compositional structure governs: code's strict nesting or NL's loose reference. The risk is that complex NL specifications compose in ways that have no clean projection to executable sub-programs.

5. **The specification problem.** Even if we can execute at Z, who specifies Z? If the user writes NL and the system maps to Z, the user cannot inspect Z directly. This creates a new kind of opacity — not "I can't read the code" but "I can't read the intent representation."

6. **Stratification and cultural invariance of Z.** If Z decomposes into Z_semantic (what) and Z_procedural (how), representation-level execution must choose which layer to operate on — and whether Z_procedural should be preserved or discarded. Furthermore, if Z_semantic itself carries cultural inflection (§3.5), then the "Platonic ideal" is not singular but a family of culturally-mediated ideals. Huh et al.'s convergence may hold at the level of local neighborhood topology (the Aristotelian refinement) rather than global geometry — which is sufficient for semantic equivalence but may be insufficient for exact execution. Empirical work on cross-lingual moral divergence (Vida et al., 2024; Li et al., 2026) and the cultural steering vector phenomenon (Veselovsky et al., 2025) suggest that even Z_semantic is not fully culture-invariant for domains involving human judgment, though it may be for purely computational domains.

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
│  Disambiguation      │  Replaces the Type-2 grammar's role as
│  at Z level          │  forced disambiguator (§4.1)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Direct  │ │ Project  │  Constrained decoding to Type-2 target
│ execute │ │ to code  │  with provenance links back to Z
│ at Z    │ │ (Python, │  for bidirectional debugging (§6.3)
│ (LCM)   │ │  Rust..) │
└─────────┘ └──────────┘
```

Meta's LCM demonstrates that the first box (Encoder to Z) and partial execution at Z are already feasible using SONAR embeddings. The critical missing pieces are the verification layer and the deterministic projection to executable code — corresponding to Open Problems #1–3 and #6.

---

## 6. Discussion

### 6.1 Is the Chomsky Wall Actually a Wall?

We have argued that the Chomsky hierarchy is a surface-form constraint, not a representation-level one. But this claim requires nuance:

- **For compilers and formal methods:** The wall is absolute. You cannot build a deterministic parser for Type-1 grammars in polynomial time. Any system that needs provable correctness must operate at Type-2 or below.
- **For LLM-based systems:** The wall is irrelevant in practice. LLMs routinely process NL (Type-1+) and produce coherent, contextually appropriate code. They do not "solve" Type-1 parsing — they bypass it entirely by operating in representation space.
- **For hybrid systems:** The wall becomes a design choice. You can accept Type-1 input, process at Z, and project to Type-2 output. The wall exists at the projection boundary, not at the input boundary.

### 6.2 The Python Observation

Python's success is partially explained by this framework. Python occupies the closest point to natural language within the Type-2 boundary:

```
Natural language  ←─── increasing ambiguity ───→  Assembly
                                                    │
          ╔═══════════════════════════════════╗      │
          ║         Type-2 boundary           ║      │
          ║   SQL ← Python ← C ← Assembly    ║      │
          ╚═══════════════════════════════════╝      │
```

SQL reads like English but parses like a formal language. Python reduces ceremony (no braces, no semicolons, significant whitespace) to approach the boundary. But neither can cross it without ceasing to be deterministically parseable.

The productive question is not "can we push further toward NL within Type-2?" (diminishing returns) but "can we accept NL input and project to Type-2 output via Z?" (a different architecture entirely).

### 6.3 Implications for Language Design

If representation-level execution becomes practical, the role of programming languages shifts:

- **Current role:** Human-writable, machine-parseable specification
- **New role:** Machine-generated, machine-verified projection from Z

In this future, programming languages become *output formats* — analogous to object code today. Humans specify intent in whatever surface form they prefer; the system projects to a verifiable formal language. The "programming language" is chosen by the machine for executability, not by the human for expressiveness.

This is the inverse of the current paradigm, where humans write in the machine's language.

**The debugging objection.** If programming languages become output formats, how do humans debug? Today, debugging works because the programmer can read the code, understand the control flow, and reason about state. If the executable is a machine-generated projection from Z, this readability vanishes. Two responses: (a) debugging shifts from inspecting code to inspecting Z — the intent representation — which requires new tooling for visualizing and querying continuous representations; (b) the projected code can be annotated with provenance links back to Z, enabling bidirectional navigation between intent and implementation. Neither response is fully satisfactory yet, but both are tractable research directions. The analogy to compiled code is instructive: few developers debug at the assembly level, yet compiled languages dominate. The debugging abstraction layer shifted upward once; it can shift again.

### 6.4 The Cultural Boundary of Z

Our argument that the Chomsky wall dissolves at Z assumes Z is culture-invariant. §3.4–3.5 complicate this assumption: even code and mathematics exhibit culturally-mediated derivation paths, and multilingual LLMs behave more like code-switching polyglots than Platonic oracles.

The practical consequence for representation-level execution is a domain split:

- **Purely computational domains** (sorting, arithmetic, data transformation): Z_semantic is plausibly culture-invariant. `sort([3,1,2])` → `[1,2,3]` regardless of the language or culture of the prompt. Representation-level execution is most tractable here.
- **Judgment-involving domains** (risk assessment, prioritization, summarization): Z carries cultural inflection. An LLM asked to "summarize the key points" will emphasize different aspects depending on prompt language (Lu and Zhang, HBR 2025), and moral reasoning shifts systematically across languages (Li et al., 2026). Representation-level execution in these domains must either (a) make the cultural frame explicit as part of the input specification, or (b) acknowledge that Z is parameterized by culture — Z(c) rather than Z.

This domain split suggests that the Chomsky wall dissolves cleanly for computation but only partially for communication. Since most real-world software involves both — computation embedded in socially-situated interfaces — representation-level systems will need to handle the boundary between culture-invariant Z_semantic and culture-dependent Z_pragmatic.

The sovereign AI movement (Korea's five-consortium initiative, Japan's ABCI 3.0, Southeast Asia's SEA-LION) can be read as an implicit recognition of this problem: if Z were truly Platonic, there would be no need for culturally-native foundation models. The market is voting for Model A.

### 6.5 Limitations

- PRH is a hypothesis, not a theorem. The convergence evidence is statistical, and it is unclear whether convergence is sufficient for exact program semantics.
- "Representation-level execution" is currently a research direction, not a working system. The open problems in §5.3 are substantial.
- The QWERTY analogy is suggestive but not rigorous. Path dependency in keyboard layouts and in programming language toolchains may have different dynamics.
- This paper is primarily a position/perspective paper. Empirical validation of the claims (especially §4.2 and §5.3) requires future work.
- The Z_semantic/Z_procedural distinction (§3.4) is proposed as a conceptual framework, not an empirically validated decomposition of representation space. Whether these layers are cleanly separable in practice is an open question.
- The cultural invariance discussion (§3.5, §6.4) relies primarily on behavioral evidence (output divergence across languages). Internal mechanistic evidence — whether the divergence originates in representation or decoding — is still emerging (Veselovsky et al., 2025; Gröger et al., 2026).

---

## 7. Related Work

**Platonic Representation Hypothesis.** Huh et al. (ICML 2024) established the convergence thesis. Ziyin et al. (2025) provided a formal proof for deep linear networks, identifying six conditions under which convergence breaks. The Aristotelian critique (Gröger et al., 2026) showed that global geometric convergence is confounded by scale, but local neighborhood agreement persists — proposing a topological rather than geometric interpretation. An information-geometric analysis (NeurIPS 2025 Workshop) frames convergence as a consequence of Bayesian posterior concentration, while also proving a "disunion theorem" for models with different approximation capabilities. Cross-domain applications extend PRH to astronomy (NeurIPS 2025), interatomic potentials (2025), and neuroscience (2025).

**Multilingual Representations and Cultural Alignment.** The Semantic Hub Hypothesis (Wu et al., ICLR 2025) demonstrates a shared middle-layer representation across languages and modalities. Brinkmann et al. (NAACL 2025) show grammatical concepts are encoded in shared feature directions across typologically diverse languages. However, a growing body of work demonstrates that multilingual capability does not imply multicultural alignment (Rystrøm et al., 2025): LLMs produce systematically different moral judgments across languages (Vida et al., 2024; Agarwal, 2024; Li et al., 2026), display Cultural Frame Switching (COLING 2025), and reflect creator ideology (Buyl et al., 2025). Veselovsky et al. (2025) identify conserved cultural steering vectors, and Naous and Xu (NAACL 2025) trace Western cultural bias to training data origins. The transfer-localization tradeoff (2025) formalizes the tension: cross-lingual alignment improves factual transfer but erases cultural localization.

**Natural Language Programming.** Beyond the systems surveyed in §2, Cheung (UC Berkeley, 2025) argues that LLM-based code translation preserves syntax but not semantics, requiring formal compositional reasoning for trustworthy translation. IntentCoding (2026) addresses the intent amplification problem via masked decoding. Meta's Large Concept Models (2024–2025) operate on sentence-level SONAR embeddings rather than tokens, representing the closest existing system to representation-level execution. The CultureManager framework (2026) introduces task-specific cultural adaptation with modular culture routers.

**Pragmatic Meaning in NLP.** LLMs remain weak at pragmatic inference: most models perform at or below chance on manner implicatures (Scientific Reports, 2024). Semantic label drift during cross-cultural translation is amplified, not reduced, by LLMs' cultural knowledge (2025). Even mathematical reasoning is affected by cultural context — LLMs struggle with math problems when cultural references change, despite identical underlying structure (2025).

## 8. Conclusion

For seventy years, programming languages have been constrained to context-free grammars — not by choice but by the mathematical requirements of deterministic parsing. Three waves of natural-language programming have attempted to bridge the gap: keyword substitution, structured natural language, and LLM-mediated translation. All three operate at the surface-form level and inherit the Chomsky wall's constraints.

The Platonic Representation Hypothesis suggests that this wall is a property of surface forms, not of representations. Neural networks trained on language and code converge toward shared representations where the Type-1/Type-2 distinction is irrelevant. This does not "solve" natural-language programming — it reframes the problem from "how do we parse NL as code?" to "how do we execute at the representation level where NL and code are the same thing?"

However, the shared representation Z is not as monolithic as the original PRH implies. We have argued that Z stratifies into Z_semantic (what is computed) and Z_procedural (how it is derived), and that multilingual LLMs behave more like code-switching polyglots than Platonic oracles — with culturally-mediated reasoning paths even in code and mathematics. The Aristotelian critique further refines this: what converges is local neighborhood topology, not global geometry. For representation-level execution, this means the approach is most tractable for purely computational domains where Z_semantic suffices, and faces additional challenges in judgment-involving domains where cultural parameterization — Z(c) rather than Z — becomes necessary. The sovereign AI movement implicitly confirms this: if Z were truly Platonic, culturally-native foundation models would be unnecessary.

Like QWERTY, formal syntax may be a vestige of a constraint that has been transcended at the representation level but persists at the toolchain level. Unlike QWERTY, the cost of persistence is not ergonomic inefficiency but a fundamental barrier to human-computer interaction: billions of people can express computational intent in natural language but cannot program, because programming requires expressing that intent in a Type-2 grammar they have never learned.

The research agenda is clear: deterministic execution at Z, verification at Z, disambiguation without Type-2 constraints, and graceful projection from Z to existing targets. The pieces exist — SONAR embeddings, Large Concept Models, constrained decoding, cultural steering vectors, program synthesis — but they have not been unified under the representation-convergence framework that PRH provides. The path forward requires not only engineering these pieces together, but also confronting the cultural stratification of Z that determines where representation-level execution can and cannot work.

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

[12] Fowler, M., and Highsmith, J. "The Agile Manifesto." 2001. (For historical context on spec-first approaches.)

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

[26] COLING 2025. "Exploring the Impact of Language Switching on Personality Traits in LLMs."

[27] Lu, J. G. and Zhang, L. D. "How Two Leading LLMs Reasoned Differently in English and Chinese." Harvard Business Review, December 2025.

[28] Ziyin, L. et al. "Proof of a Perfect Platonic Representation Hypothesis." arXiv:2507.01098, 2025.

[29] Aksoy, M. "Whose Morality Do They Speak? Unraveling Cultural Bias in Multilingual Language Models." arXiv:2412.18863, 2024.

[30] "Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs." arXiv:2510.26024, 2025.

[31] "Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?" arXiv:2503.18018, 2025.

[32] ACL 2025. "Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts."

[33] "Scaling Laws for Code: Every Programming Language Matters." arXiv:2512.13472, 2025.

[34] Meta AI. "Large Concept Models: Language Modeling in a Sentence Representation Space." arXiv:2412.08821, 2024.

[35] SONAR. "Sentence-Level Multimodal and Language-Agnostic Representations." Meta AI, 2024.

[36] Cheung, A. "LLM-Based Code Translation Needs Formal Compositional Reasoning." UC Berkeley EECS-2025-174, 2025.

[37] "Semantic Label Drift in Cross-Cultural Translation." arXiv:2510.25967, 2025.

[38] "Manner Implicatures in Large Language Models." Scientific Reports (Nature), November 2024.

[39] "An Information-Geometric View of the Platonic Hypothesis." NeurIPS 2025 Workshop.

[40] "Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for LLMs." arXiv:2510.07213, 2025.

[41] IntentCoding. "Amplifying User Intent in Code Generation." arXiv:2602.00066, 2026.

[42] CultureManager. "Mind the Gap in Cultural Alignment: Task-Aware Culture Management for Large Language Models." arXiv:2602.22475, 2026.
