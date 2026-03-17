# Beyond the Chomsky Wall: Platonic Representations as the Convergence Point of Natural Language and Code

> Draft v1 — 2026-03-17
> Target: COLM 2027 or ICML 2027 Workshop
> Format: 8 pages + references

---

## Abstract

Programming languages have remained syntactically formal for seven decades — not because of engineering convenience, but because deterministic parsing requires context-free (Type-2) grammars, while natural language is context-sensitive (Type-1 or beyond). Recent attempts to bridge this gap fall into three categories: keyword substitution (replacing `if` with native-language equivalents), structured natural language (constraining prose into parseable templates), and LLM-mediated translation (generating code from free-form text). All three operate at the surface-form level and inherit fundamental limitations: the first two cannot escape Type-2 constraints, and the third is non-deterministic. We argue that the Platonic Representation Hypothesis (Huh et al., ICML 2024) — which demonstrates that neural networks trained on different modalities converge toward a shared statistical model of reality — reframes this problem entirely. If natural language and code are projections of the same latent structure, the bottleneck is not translating between surface forms but operating at the representation level where the Chomsky distinction dissolves. We survey the current landscape of natural-language programming systems (CoRE, SudoLang, Shire, Spec-Driven Development), analyze why each remains bound by surface-form constraints, and propose a research agenda for representation-level program execution. We draw an analogy to the QWERTY keyboard: the original constraint (typewriter jamming) vanished, but the layout persisted — programming language syntax may be a similar vestige of a constraint (deterministic parsing) that LLMs have made obsolete at the representation level, though not at the surface level.

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
- An analysis of the Platonic Representation Hypothesis as it applies to the code–language divide (§3)
- A reinterpretation of the Chomsky wall as a surface-form artifact, not a fundamental barrier (§4)
- A research agenda for representation-level program execution (§5)

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

### 3.2 Evidence for Code as a Modality

While Huh et al. focus on vision and language, code fits naturally into the framework:

- **CodeBERT, UniXcoder, StarCoder:** Models trained jointly on natural language and code develop aligned representations — a code snippet and its natural-language description map to nearby points in embedding space.
- **Translation emergence:** LLMs trained primarily on natural language can generate code without code-specific training — suggesting that code structure is partially recoverable from natural-language representations alone.
- **Cross-modal transfer:** Copilot-style models improve code completion by training on natural-language documentation, and improve documentation generation by training on code. This bidirectional transfer is predicted by PRH if both modalities project from the same Z.

### 3.3 What Z Looks Like for Code

If natural language and code are both projections of Z, what is Z?

For code, Z is likely the **computational intent** — the abstract specification of what should happen, stripped of:
- Surface syntax (Python's `if x > 5:` vs. C's `if (x > 5)`)
- Implementation details (loop vs. recursion vs. vectorization)
- Language-specific idioms (`list comprehension` vs. `map/filter`)

This is consistent with the observation that LLMs can translate between programming languages with high accuracy — they are operating at a level where `for item in items: process(item)` (Python) and `items.forEach(item => process(item))` (JavaScript) are the same point in Z.

### 3.4 The Convergence Prediction

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

### 4.2 The Wall Dissolves — at the Representation Level

But LLMs do not parse. They do not construct parse trees. They map input sequences to high-dimensional representations via attention over the full context. In this representation space:

- "x가 5보다 크면" (Korean NL)
- `if x > 5:` (Python)
- `(> x 5)` (Lisp)
- The abstract predicate "x exceeds 5"

...all map to nearby points. The Type-1/Type-2 distinction exists in the surface form, not in the representation.

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

The problem is in step 2: translation is non-deterministic. The same NL input produces different code across runs. This is because the LLM is mapping from a higher-entropy space (NL) to a lower-entropy space (code), and there are multiple valid projections.

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

### 5.3 Open Problems

1. **Determinism at Z.** How do you guarantee that the same NL input always produces the same execution result? PRH shows representations converge, but convergence ≠ identity. Temperature-zero decoding helps but doesn't eliminate stochasticity from attention mechanisms.

2. **Verification at Z.** Current formal verification requires Type-2 (or at most Type-1) input. Can we verify properties of programs represented in continuous embedding space? This connects to neural program verification, an early-stage field.

3. **The grounding problem.** PRH shows that representations converge statistically. But programming requires exact grounding: `x > 5` must mean exactly "x is strictly greater than 5", not "x is approximately 5 or maybe a bit more." How much representational precision is sufficient for exact execution?

4. **Compositionality.** Natural language is compositional (the meaning of a sentence derives from its parts). Code is compositional (the behavior of a program derives from its modules). But are the compositional structures aligned in Z? If not, complex programs may not be expressible from NL via Z.

5. **The specification problem.** Even if we can execute at Z, who specifies Z? If the user writes NL and the system maps to Z, the user cannot inspect Z directly. This creates a new kind of opacity — not "I can't read the code" but "I can't read the intent representation."

### 5.4 A Possible Architecture

```
User input (any surface form)
        │
        ▼
┌──────────────────────┐
│  Encoder to Z        │  (LLM or specialized encoder)
│  NL, code, spec →    │
│  shared representation│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Verification at Z   │  (neural verifier: is Z complete? unambiguous?)
│  "Is this intent     │
│   fully specified?"  │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Direct  │ │ Project  │  (constrained decoding to Type-2 target)
│ execute │ │ to code  │
│ at Z    │ │ (Python, │
│ (simple)│ │  Rust..) │
└─────────┘ └──────────┘
```

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

### 6.4 Limitations

- PRH is a hypothesis, not a theorem. The convergence evidence is statistical, and it is unclear whether convergence is sufficient for exact program semantics.
- "Representation-level execution" is currently a research direction, not a working system. The open problems in §5.3 are substantial.
- The QWERTY analogy is suggestive but not rigorous. Path dependency in keyboard layouts and in programming language toolchains may have different dynamics.
- This paper is primarily a position/perspective paper. Empirical validation of the claims (especially §4.2 and §5.3) requires future work.

---

## 7. Conclusion

For seventy years, programming languages have been constrained to context-free grammars — not by choice but by the mathematical requirements of deterministic parsing. Three waves of natural-language programming have attempted to bridge the gap: keyword substitution, structured natural language, and LLM-mediated translation. All three operate at the surface-form level and inherit the Chomsky wall's constraints.

The Platonic Representation Hypothesis suggests that this wall is a property of surface forms, not of representations. Neural networks trained on language and code converge toward shared representations where the Type-1/Type-2 distinction is irrelevant. This does not "solve" natural-language programming — it reframes the problem from "how do we parse NL as code?" to "how do we execute at the representation level where NL and code are the same thing?"

Like QWERTY, formal syntax may be a vestige of a constraint that has been transcended at the representation level but persists at the toolchain level. Unlike QWERTY, the cost of persistence is not ergonomic inefficiency but a fundamental barrier to human-computer interaction: billions of people can express computational intent in natural language but cannot program, because programming requires expressing that intent in a Type-2 grammar they have never learned.

The research agenda is clear: deterministic execution at Z, verification at Z, and graceful projection from Z to existing Type-2 targets. The pieces exist — constrained decoding, tool use, program synthesis, neural verification — but they have not been unified under the representation-convergence framework that PRH provides.

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
