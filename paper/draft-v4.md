# Beyond the Chomsky Wall: Platonic Representations as the Convergence Point of Natural Language and Code

> Draft v4 — 2026-03-18
> Target: COLM 2027 or ICML 2027 Workshop
> Format: 8 pages + references

> **Changes from v3:** Added running example threading all sections (§1.5, throughout). Formalized disambiguation conservation via information theory (§4.3.1). Connected Z_semantic to denotational semantics (§3.2.1). Added pilot experiment design (§5.6). Added safety/alignment implications of culturally-stratified Z (§6.4.1). Formalized D_train as a variable shaping Z, showing Z_semantic convergence is an idealization contaminated by training distribution (§3.5.1). Expanded ethical analysis: computational colonialism, consent opacity, accountability gap, cultural feedback loops, and five mitigation paths (§6.4.2). Added Prediction 6 (D_train sensitivity) and cross-model experiment extension (§5.6). **Added tokenization bottleneck analysis: spacing typology across writing systems, Korean 띄어쓰기 problem, Chinese segmentation ambiguity, token tax, and spacing paradox for code (§4.2.1). Added Prediction 7 (spacing robustness) with protocol. Added symbols/emoji/Unicode analysis as representational directness spectrum — emoji as natural experiments in Z_semantic, cross-cultural emoji variation mirroring Z stratification, ZWJ composition paralleling compositionality problem (§4.2.2).** Expanded compositionality treatment with categorical framing (§5.3). Added naturalness hypothesis and neuro-symbolic connections to Related Work (§7). Consolidated contributions. Compressed §2 further.

---

## Abstract

Programming languages use context-free (Type-2) grammars because deterministic parsing requires them — not by convention but by mathematical necessity. Natural language is context-sensitive (Type-1 or beyond) and cannot be deterministically parsed. Three waves of natural-language programming — keyword substitution, structured NL, and LLM-mediated translation — all operate at the surface-form level and inherit this constraint. We argue that the Platonic Representation Hypothesis (Huh et al., ICML 2024) reframes the problem: if NL and code converge toward a shared latent structure Z in representation space, the Chomsky wall constrains only surface-form projections, not representations themselves. We propose that Z is stratified — Z_semantic (computational result) converges cross-culturally while Z_procedural (derivation path) and Z_pragmatic (communicative frame) remain culturally mediated — and that multilingual LLMs behave as code-switching polyglots, not Platonic oracles. We formalize the disambiguation cost migration from Type-2 grammars to the representation level using an information-theoretic argument, showing that specification complexity is conserved: ambiguity removed from syntax reappears as semantic verification cost. We connect Z_semantic to denotational semantics, grounding the convergence claim in programming language theory. We show that training data cultural composition (D_train) is an underexamined variable that contaminates even Z_semantic — the "purely computational" layer — and derive ethical implications including computational colonialism, consent opacity, and cultural feedback loops. We derive seven falsifiable predictions — including spacing robustness and training distribution sensitivity — propose a pilot experiment design, and identify a verification paradox that constrains the deployment path.

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
1. A formal framework connecting PRH to the code–language divide, including semi-formal definitions, connection to denotational semantics, and information-theoretic formalization of disambiguation conservation (§3.2, §3.2.1, §4.3.1)
2. Stratification of Z into Z_semantic, Z_procedural, Z_pragmatic, with evidence for differential convergence; analysis of D_train dependence showing that Z_semantic convergence is an idealization contaminated by training distribution (§3.4–3.5.1); ethical implications including computational colonialism, consent opacity, and accountability gaps (§6.4.1–6.4.2)
3. Seven falsifiable predictions (including D_train sensitivity and spacing robustness), evaluation criteria, and a concrete pilot experiment design (§3.7, §4.2.1, §5.5, §5.6)
4. Analysis of disambiguation cost migration and the resulting verification paradox (§4.3, §6.6)
5. A research agenda grounded in existing systems (LCM/SONAR), categorical compositionality analysis, and seven open problems including surface-form invariance (§5)

### 1.5 Running Example

Throughout this paper, we use a single example to ground abstract concepts:

> **"주어진 목록에서 가장 큰 값 세 개를 찾아라"**
> ("Find the three largest values in the given list")

This sentence is computationally precise yet linguistically ambiguous. "가장 큰" (largest) is unambiguous for numbers but ambiguous for composite objects (largest by what key?). The sentence lacks type information (what kind of list?), error handling (what if fewer than three elements?), and ordering specification (should the result be sorted?). Every programmer resolves these ambiguities — in Type-2 programming, by writing `sorted(lst, reverse=True)[:3]`, which implicitly answers all questions. We will track how each wave, each Z layer, and the verification paradox handle these ambiguities.

---

## 2. Current Landscape: Surface-Form Approaches

### 2.1 The Three Waves in Practice

Our running example across the three waves:

**Wave 1** (keyword substitution):
```
# Han (Korean Rust)
함수 상위셋(목록: 배열<정수>) -> 배열<정수> {
    정렬(목록, 역순=참)[..3]
}
```
The grammar is isomorphic to Rust. The ambiguities are resolved identically: the programmer writes Type-2 code in Korean tokens. Only terminal symbols change; the parse tree does not.

**Wave 2** (structured NL):
```
# CoRE-style
TASK: 상위 세 값 추출
INPUT: 목록 (정수 배열)
STEPS:
  1. 목록을 내림차순 정렬
  2. 처음 세 원소 반환
OUTPUT: 정수 배열
```
Structure is imposed on Korean NL to make it interpretable. A new formal grammar is born — and the Chomsky level resets to Type-2. **The paradox:** making NL precise enough to execute requires removing the properties that make it natural.

**Wave 3** (LLM-mediated translation):
```
User: "주어진 목록에서 가장 큰 값 세 개를 찾아라"
LLM → sorted(lst, reverse=True)[:3]  # Run 1
LLM → heapq.nlargest(3, lst)         # Run 2
LLM → [x for x in lst if x >= sorted(lst)[-3]]  # Run 3 (wrong for duplicates)
```
Same input, three outputs. Runs 1 and 2 are semantically equivalent; Run 3 has a subtle bug. The LLM acts as a non-deterministic compiler.

### 2.2 Formal Comparison

| Dimension | Wave 1: Keyword Sub. | Wave 2: Structured NL | Wave 3: LLM Translation | Proposed: Z-Level |
|-----------|---------------------|-----------------------|------------------------|-------------------|
| **Chomsky level** | Type-2 (unchanged) | Type-2 (re-imposed) | Type-1 input → Type-2 output | Type-1 input → Z → Type-2 output |
| **Determinism** | Full | Full | Non-deterministic | Deterministic (target) |
| **Expressiveness** | = base language | < natural language | ≈ natural language | = natural language |
| **Verification** | Standard (Type-2) | Standard (Type-2) | None (stochastic) | At Z (open problem) |
| **Cultural adaptation** | Terminal symbols only | Template structure | Implicit in LLM | Explicit via Z(c) |
| **Disambiguation** | By programmer (grammar) | By programmer (template) | By LLM (non-deterministic) | By system (at Z) |
| **What changes** | Keywords | Surface syntax | Generation process | Execution level |

### 2.3 The RAG Parallel

```
RAG:          NL query → vector similarity → relevant docs → LLM synthesis
NL→Code:      NL intent → [???] → executable result
```

Both map NL to structured output via learned representations. The critical difference: RAG tolerates approximate matches (cosine similarity 0.95 is fine), but code requires exact semantics (0.95 is a bug). Our running example: "가장 큰 값 세 개" must yield exactly three values, exactly the largest, in a deterministic order — not "approximately three values that are roughly the largest."

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

**Definition 3 (Shared Latent Structure).** Z is the limit space toward which encoder images converge: Z ≈ lim_{scale→∞} Im(E_i) for all modalities i. In practice, Z is approximated by the shared representation space of large-scale multimodal models (e.g., SONAR's 200-language sentence embeddings). Critically, Z is not only a function of scale but also of the training distribution: Z = Z(scale, D_train). Two models at the same scale trained on different cultural corpora may converge to *different* Z — a point we formalize in §3.5.1.

**Definition 4 (Z Stratification).** Z decomposes into three projections with decreasing convergence strength:

- **Z_sem : Z → S** (semantic layer) — the computational result (*what*). Cross-culturally convergent: ||ΔZ_sem||_cross-cultural < ε for purely computational domains.
- **Z_proc : Z → P** (procedural layer) — the derivation path (*how*). Culturally mediated: ||ΔZ_proc||_cross-cultural ≫ ε even for equivalent computations.
- **Z_prag : Z → Q** (pragmatic layer) — the communicative frame (*for whom*). Culturally specific: ||ΔZ_prag||_cross-cultural is maximal.

**Definition 5 (The Chomsky Wall).** The *wall* is the computational boundary between Type-2 (context-free, deterministically parseable) and Type-1 (context-sensitive, PSPACE-complete). The wall is *surface-level* if it constrains only parsing of surface forms, not operations on representations.

**Central Claim.** If PRH holds for code as a modality (Def. 2), then the Chomsky wall (Def. 5) constrains only the projection D_i : Z → M_i, not the representation Z itself. Operations at Z are not subject to Chomsky hierarchy constraints because Z is continuous, not discrete — it has no grammar to classify.

#### 3.2.1 Connection to Denotational Semantics

The Z stratification connects to classical programming language theory. *Denotational semantics* (Scott & Strachey, 1971) maps programs to mathematical objects in a semantic domain **D** — typically a complete partial order (CPO) or domain. The meaning of a program P is its denotation ⟦P⟧ ∈ **D**, independent of syntax.

We observe a structural parallel:

| | Denotational Semantics | PRH Framework |
|---|---|---|
| **Surface form** | Program text (syntax) | Modality-specific surface form |
| **Meaning** | ⟦P⟧ ∈ **D** | Z_sem(E(input)) |
| **Compositionality** | ⟦P₁; P₂⟧ = ⟦P₂⟧ ∘ ⟦P₁⟧ | Z_sem(E(s₁ + s₂)) ≈ Z_sem(E(s₂)) ∘ Z_sem(E(s₁)) (conjecture) |
| **Invariance** | ⟦P⟧ = ⟦P'⟧ for semantically equivalent P, P' | sim(E(x), E(y)) → 1 for equivalent x, y |

**Conjecture (Semantic Convergence).** If PRH holds, then Z_semantic is a neural approximation of the denotational semantic domain **D**: for programs P with denotation ⟦P⟧ and NL descriptions d of P:

    lim_{scale→∞} Z_sem(E(d)) ≈ ⟦P⟧   (up to continuous bijection)

This conjecture, if true, would explain three observations: (a) why Z_semantic converges cross-culturally — denotational semantics is culture-invariant by construction; (b) why the convergence is stronger for computational than judgment-involving domains — only the former have well-defined denotations; (c) why compositionality is the hard problem — denotational compositionality requires *homomorphic* structure, which NL does not guarantee (§5.3).

For our running example: the denotation of `sorted(lst, reverse=True)[:3]` is a function **List[Comparable] → List[Comparable]** that selects the three maximal elements. The PRH claim is that the Korean sentence "가장 큰 값 세 개를 찾아라" converges to the *same point* in Z_semantic — because both descriptions denote the same function.

### 3.3 Evidence for Code as a Modality

While Huh et al. focus on vision and language, code fits naturally into the framework:

- **Aligned embedding spaces.** Models trained jointly on NL and code (CodeBERT, UniXcoder, StarCoder) develop representations where code snippets and NL descriptions map to nearby points. UniXcoder achieves 77.1% MRR on NL-code search across six programming languages, indicating convergence across multiple target languages.
- **Cross-model transferability.** Linear transformations align latent spaces of different LLMs, and steering vectors from smaller models (7B) control behavior in larger ones (70B) (ACL 2025). This weak-to-strong transferability is a strong PRH prediction — if representations converge toward Z, the structure of Z should be recoverable across architecturally distinct models.
- **The Naturalness Hypothesis.** Hindle et al. (2012) established empirically that software is "natural" — it is repetitive and predictable in ways similar to natural language text. N-gram models over code approach the cross-entropy of English text. This is a *precondition* for PRH convergence: code is natural enough to share statistical structure with NL, which is why joint NL-code models succeed.
- **Translation emergence.** LLMs trained primarily on NL can generate code without code-specific training, suggesting code structure is partially recoverable from NL representations. Cross-language synergies in code follow predictable scaling laws (Cassano et al., 2025).
- **Bidirectional transfer.** Copilot-style models improve code completion by training on NL documentation, and improve documentation by training on code — predicted by PRH if both modalities project from the same Z.
- **Language-independent code semantics.** "Beyond Syntax: How Do LLMs Understand Code?" (ICSE 2025 NIER) provides direct empirical evidence: LLMs develop language-independent semantic representations of code, with syntax-specific components in early layers feeding into shared semantic components in middle layers. This is the strongest existing evidence that code fits into the PRH framework — though the authors do not frame it as such.

### 3.4 The Stratification of Z

The preceding analysis assumes Z is monolithic. But consider how different mathematical traditions derive the same theorem: the Bourbaki school proceeds via axiomatic abstraction; the Soviet tradition favors constructive methods; Indian mathematical traditions historically employed intuitive-inductive reasoning. The theorem is identical — the derivation paths differ systematically. The same applies to code: `sort(array)` produces the same output regardless of whether a developer reaches for recursion (functional tradition) or iteration (imperative tradition).

This motivates the Z stratification (Definition 4):

| Layer | Content | Convergence | Evidence |
|-------|---------|-------------|----------|
| Z_sem | What is computed | Cross-culturally convergent | Semantic Hub Hypothesis (Wu et al., ICLR 2025); shared grammatical features across diverse languages (Brinkmann et al., NAACL 2025) |
| Z_proc | How it is derived | Culturally mediated | Mathematical tradition differences; programming paradigm preferences; reasoning-language effects (Li et al., 2026) |
| Z_prag | For whom it is framed | Culturally specific | LLM pragmatic inference failures (Scientific Reports, 2024); Cultural Frame Switching (COLING 2025) |

**Running example.** "가장 큰 값 세 개를 찾아라" in Z:
- **Z_sem**: select top-3 by magnitude from a list (culture-invariant)
- **Z_proc**: a Korean developer may reach for a filtering approach ("큰 값을 제외하지 않는다" — "don't exclude large values"), while a Haskell-trained developer may think `take 3 . sortBy (flip compare)`. The derivation path differs; the result is identical.
- **Z_prag**: the sentence-final "~아라" is an imperative in Korean, implying a human-to-computer command frame. An equivalent English phrasing might use a softer "find..." or "return the...". The communicative register is culturally specific.

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

#### 3.5.1 Training Data as Cultural Prior: D_train Dependence

PRH's convergence claim assumes that scale will wash out training distribution effects — that sufficiently large models trained on *any* distribution converge to the same Z. But this assumption is empirically contested and has direct consequences for our framework.

**The problem.** Z is not determined by scale alone. It is jointly determined by scale and the training distribution D_train:

    Z = Z(scale, D_train)

If D_train is culturally skewed — as all current large models are, with English comprising 40–90% of training tokens — then Z inherits the cultural priors of the dominant culture, even in layers we expect to converge.

**Evidence that D_train shapes Z_semantic.** Even "purely computational" domains carry implicit cultural defaults:
- **Sort order**: "정렬하라" (sort) defaults to ascending in most contexts, but Korean lexicographic order differs from English — the "natural" sort is culture-dependent.
- **Null handling**: Whether null values sort first, last, or cause an error varies by programming tradition (SQL NULL semantics differ across databases, reflecting different design cultures).
- **Number formatting**: "큰 값" (large value) activates different thresholds depending on the cultural context of the training data (financial norms, scientific conventions).
- **Error semantics**: Whether an out-of-bounds access returns a default, throws an exception, or is undefined reflects PL culture (Python vs. C vs. Go), which is overrepresented in D_train proportional to each language's corpus share.

**Formal consequence.** Define the *training distribution divergence*:

    Δ_D(Z₁, Z₂) = ||Z(s, D₁) - Z(s, D₂)||  at fixed scale s

PRH claims Δ_D → 0 as s → ∞. But the Aristotelian critique suggests this convergence may be confounded: what disappears may be *geometric agreement* while *topological agreement* (local neighborhoods) persists. If so, two models trained on different D_train may agree on which operations are *similar* (topological) while disagreeing on *how to execute* them (geometric) — precisely the Z_semantic vs. Z_procedural split.

**Running example.** Consider "가장 큰 값 세 개를 찾아라" processed by:
- Model A (D_train: 80% English, 5% Korean): Z maps this to a Python-idiomatic representation (sorted + slice), defaulting to numeric magnitude comparison.
- Model B (D_train: 50% Korean, 20% English): Z maps to a potentially different representation where "가장 큰" carries additional connotations from Korean business/academic usage (e.g., "most significant" rather than strictly "numerically largest").

The models may agree that this is a "select top-k" operation (Z_semantic topology preserved) but disagree on the implicit comparison criterion (Z_semantic geometry diverges). This is not merely a Z_procedural difference — it is a **Z_semantic divergence induced by D_train**, even in a supposedly "purely computational" domain.

**Implication for the framework.** The clean separation between "Z_semantic converges" and "Z_procedural/Z_pragmatic diverge" is an idealization. In practice, D_train contaminates the boundary:

    Z_semantic(observed) = Z_semantic(ideal) + bias(D_train)

The bias term may be small for arithmetic operations but nontrivial for any operation involving implicit criteria, default behaviors, or edge case handling. This has direct ethical consequences (§6.4.2).

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

The framework generates seven testable predictions (Predictions 1–6 below; Prediction 7 in §4.2.1):

**Prediction 1 (Scale-Convergence).** As model scale increases, mean cosine distance between NL descriptions and semantically equivalent code in shared representation space decreases monotonically. *Protocol:* Compare NL-code alignment (MRR on code search) across model scales from CodeBERT (125M) → UniXcoder (350M) → StarCoder (15B) → frontier models using a fixed benchmark. See §5.6 for pilot experiment design.

**Prediction 2 (Cross-Lingual Semantic Invariance).** For purely computational operations, cross-lingual same-operation similarity exceeds within-language different-operation similarity. "정렬하라" (Korean: sort) and "sort this list" (English) should be closer in Z than "sort this list" and "reverse this list" in the same language. This should *not* hold for judgment-involving operations. *Protocol:* See §5.6.

**Prediction 3 (Stratification Separability).** Z_sem and Z_proc are separable via probing classifiers: a probe detecting *what is computed* generalizes across languages; a probe detecting *how it is derived* shows language-specific patterns. *Protocol:* Train probing classifiers on intermediate representations of multilingual code generation; measure cross-lingual transfer accuracy for semantic vs. procedural probes.

**Prediction 4 (Domain-Dependent Determinism).** Constrained decoding from Z to code achieves higher consistency (same NL → same code across runs) for purely computational tasks than for judgment-involving tasks. *Protocol:* Measure output determinism of temperature-zero constrained decoding across domain types; quantify determinism gap.

**Prediction 5 (Disambiguation Migration).** As systems move from Type-2 to representation-level input, total disambiguation effort is conserved — it migrates from syntax errors to semantic clarification queries. *Protocol:* Compare the rate of user clarification queries in a representation-level prototype vs. type/syntax errors in equivalent Type-2 tasks per unit of task complexity.

**Prediction 6 (Training Distribution Sensitivity).** For models at comparable scale, Z_semantic divergence between models is positively correlated with D_train cultural divergence — and this effect persists even for "purely computational" operations involving implicit criteria (e.g., default sort order, null handling, edge case semantics). *Protocol:* Compare Z embeddings of identical computational task descriptions across models with different D_train compositions (e.g., GPT-4 vs. HyperCLOVA X vs. Qwen); measure Z_semantic divergence as a function of D_train overlap (Jaccard similarity of training corpus cultural composition). Prediction 2's invariance should weaken as D_train divergence increases.

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

#### 4.2.2 Symbols, Emoji, and the Spectrum of Representational Directness

Between natural language and continuous representations, there exists a spectrum of surface forms with varying degrees of *representational directness* — how closely the surface form approximates the underlying concept without linguistic mediation.

**The spectrum:**

```
NL              Emoji/Pictographs    Mathematical Symbols    Code Operators     Z
"더 크다"        📈 > ⬆️             >  ≥  ≫                 >, >=, >>         [continuous]
(linguistic,     (iconic,             (formal,               (formal,           (no surface
culturally       semi-universal)      universal)             universal)         form)
mediated)
```

Each step rightward reduces linguistic mediation and cultural dependence:

| Surface Form | Example | Chomsky Level | Cultural Dependence | Ambiguity |
|-------------|---------|:---:|:---:|:---:|
| NL sentence | "가장 큰 값 세 개를 찾아라" | Type-1 | High | High |
| Emoji sequence | 🔝3️⃣📊 ("top 3 in chart") | None (no grammar) | Medium | Medium |
| Math notation | max₃(L) | Type-2 (formal) | Low | Low |
| Code | `sorted(L)[-3:]` | Type-2 (formal) | None | None |
| Z (representation) | E("top 3") ∈ ℝ^d | None (continuous) | None (target) | None (target) |

**Emoji as natural experiments in Z_semantic.** Unicode emoji (😀, 📊, 🔝, ⚠️) are pictographic symbols that encode concepts without linguistic structure. They bypass the Chomsky hierarchy entirely — there is no grammar of emoji, yet they communicate meaning cross-linguistically. In this sense, emoji are the closest *existing surface form* to Z: discrete tokens that directly represent abstract concepts, largely independent of the user's language.

The Unicode Consortium's emoji standardization is, in effect, a manually curated surface-form projection from a consensus Z_semantic. When Unicode defines 🔥 as "fire," it assigns a single codepoint to an abstract concept that maps to "불" (Korean), "fire" (English), "fuego" (Spanish), "火" (Chinese) — precisely the cross-lingual convergence that PRH predicts for Z_semantic.

**But emoji also exhibit Z stratification.** Despite their apparent universality, emoji carry cultural variation that mirrors the Z_sem / Z_proc / Z_prag decomposition:

| Layer | Emoji Example | Convergence |
|-------|--------------|-------------|
| Z_sem | 📊 = "chart/data" | Near-universal |
| Z_proc | 🙏 = prayer (Western) vs. gratitude/greeting (East Asian) vs. high-five (some contexts) | Culturally divergent |
| Z_prag | 👍 = positive (Western) vs. potentially offensive (Middle East, parts of West Africa) | Culturally specific |

Research on cross-cultural emoji interpretation confirms this pattern: Lu et al. (2016) found that the same emoji received sentiment ratings differing by up to 3 points (on a 5-point scale) across cultures; Barbieri et al. (2016) showed emoji semantics vary significantly across languages even on the same platform. Emoji are a microcosm of the Z stratification thesis: the *referent* converges (Z_sem), the *connotation* diverges (Z_proc), the *social register* is culturally specific (Z_prag).

**Emoji composition parallels the compositionality problem.** Unicode ZWJ (Zero Width Joiner) sequences compose emoji: 👩 + 💻 = 👩‍💻 (woman technologist). This compositional structure is *partially* systematic (gender + profession = gendered profession) but breaks down for complex compositions — mirroring the NL compositionality problem (§5.3). Code composition is strict and deterministic (`sort` + `[:3]` = exactly "sort then take first three"); emoji composition is loose and context-dependent (🔥 + 📊 = "trending chart"? "chart on fire"? "urgent data"?).

**Mathematical and code symbols as Z_semantic surface forms.** At the formal end of the spectrum, mathematical symbols (`>`, `∑`, `∫`, `→`) and code operators (`==`, `!=`, `&&`, `->`) are *already* operating near Z_semantic: they are cross-culturally unambiguous, language-independent, and map directly to computational operations. The symbol `>` means "greater than" in every mathematical tradition and every programming language. These symbols achieved Z_semantic convergence *before* neural networks, through millennia of mathematical standardization.

This suggests a refinement: the Chomsky wall is not a binary. Surface forms exist on a continuum of representational directness, from fully linguistic (NL, high cultural mediation, Type-1) to fully formal (code/math, no cultural mediation, Type-2) to iconic (emoji, medium cultural mediation, no grammar). LLMs and PRH predict that the continuous representation Z is the limit point of this spectrum — the fully decontextualized, fully universal representation that all surface forms approximate to varying degrees.

**Running example.** Our example "가장 큰 값 세 개를 찾아라" can be expressed at each point on the spectrum:
- NL: "주어진 목록에서 가장 큰 값 세 개를 찾아라"
- Emoji-augmented: "📋➡️🔝3️⃣"
- Math: top₃(L) = {x ∈ L : |{y ∈ L : y > x}| < 3}
- Code: `heapq.nlargest(3, lst)`
- Z: E(any of the above) → nearby points in ℝ^d

The emoji version (📋➡️🔝3️⃣) is more ambiguous than code but more universal than Korean NL — it demonstrates both the promise and the limit of non-linguistic representation: sufficient for simple intent, insufficient for precise specification.

#### 4.2.1 The Tokenization Bottleneck: Spacing and the Surface-Z Boundary

The claim that the wall dissolves at Z assumes a clean mapping E : surface_form → Z. In practice, this mapping is mediated by tokenization: E(tokenize(surface_form)). The tokenizer is a surface-form processor — and spacing conventions are a surface-form property that *leaks through* the tokenizer into Z.

**Typology of spacing systems.** Natural languages vary fundamentally in how they delimit word boundaries:

| Type | Languages | Tokenization Behavior |
|------|-----------|----------------------|
| Space-delimited, stable | English, Spanish | Word boundaries align with tokens; BPE is effective |
| Space-delimited, complex rules | Korean (띄어쓰기), German | Spacing errors common in practice; compounds challenge tokenizers |
| No word boundaries | Chinese (中文), Japanese (日本語), Thai (ภาษาไทย) | Requires segmentation model; segmentation is itself ambiguous |
| Agglutinative + spaced | Turkish, Finnish, Hungarian | Single words carry sentence-level meaning; aggressive subword splitting |

Code, by contrast, has *no spacing ambiguity* — the grammar determines tokenization. `sorted(lst,reverse=True)[:3]` tokenizes identically regardless of whitespace preferences, because the lexer rules are unambiguous. This is another manifestation of the Type-2 advantage: formal grammars resolve token boundaries deterministically.

**English is not immune.** English is often treated as the "well-spaced" baseline, but spacing degradation occurs systematically in real-world input pipelines:
- **Hyphenation at line breaks**: "pro-\ngramming" is read by OCR as two tokens "pro-" and "gramming" — a word boundary is hallucinated where none exists.
- **Justified text**: Variable inter-word spacing in typeset documents confuses OCR word boundary detection, producing outputs like "the  function   returns" with inconsistent spacing.
- **Multi-column layouts**: Newspaper and academic paper columns cause OCR engines to read across columns ("The function Section 3 returns") rather than within them.
- **Degraded scans**: Low-resolution or noisy scans produce intra-word gaps ("cl ose" for "close") or merge adjacent words ("isTrue" for "is True").
- **PDF text extraction**: Programmatic text extraction from PDFs — a realistic input for specification-level systems — frequently loses or distorts spacing, especially for justified, multi-column, or non-standard-font documents (Nguyen et al., 2021).

These are not edge cases. OCR and document processing are a primary input pathway for representation-level systems: a user scanning a specification document, extracting text from a PDF, or voice-dictating (where spacing must be inferred) will produce spacing-degraded input. The "stable spacing" assumption for English is an idealization that breaks under real-world conditions.

**The Korean spacing problem.** Korean 띄어쓰기 is notoriously difficult — even native speakers regularly violate the rules, and the rules themselves are debated. Our running example admits at least four spacing variants:

```
(a) "주어진 목록에서 가장 큰 값 세 개를 찾아라"     (correct spacing)
(b) "주어진목록에서 가장큰값 세개를 찾아라"         (common informal spacing)
(c) "주어진 목록 에서 가장 큰값 세개 를 찾아라"     (spacing errors)
(d) "주어진목록에서가장큰값세개를찾아라"             (no spacing)
```

All four express identical computational intent — and critically, **all four are readable by a Korean human speaker**. Korean's agglutinative morphology (조사 and 어미 as grammatical markers) provides sufficient structural cues for humans to recover word boundaries without spaces. "주어진목록에서가장큰값세개를찾아라" is disfluent but unambiguous: the particles -에서, -를, and the verb ending -아라 mark syntactic boundaries.

English lacks this property. "findthethreelargestvaluesinthegivenlist" is substantially harder for English speakers to parse, because English relies on spaces as the primary word boundary signal — morphological cues are weaker. This asymmetry means that **spacing removal degrades information differently across languages**: Korean loses *fluency* but preserves *parsability*; English loses *both*.

But BPE tokenizers produce different token sequences for each variant:
- (a) → ~12 tokens (well-aligned with training data)
- (d) → ~8 tokens (treated as a single long string, split at subword boundaries that may not align with morphemes)

If E(tokenize(a)) ≠ E(tokenize(d)), the wall has *not* fully dissolved — surface-form spacing variation produces different representations for identical meaning.

**The Chinese segmentation problem.** Chinese has no spaces at all. "找到列表中最大的三个值" (find the three largest values in the list) requires a word segmentation model to determine boundaries:
- Segmentation A: "找到 / 列表 / 中 / 最大 / 的 / 三个 / 值" (7 words)
- Segmentation B: "找到 / 列表中 / 最大的 / 三 / 个 / 值" (6 words, different grouping)

Different segmentations → different token sequences → potentially different Z. The segmentation model is itself trained on data with culturally-specific boundary conventions — another instance of D_train dependence (§3.5.1) operating at the tokenization level.

**The token tax.** Non-space-delimited and non-Latin languages systematically require more tokens to express the same semantic content — the "token tax" (Petrov et al., 2024). For our running example:

| Language | Text | Approx. tokens (GPT-4) | Token tax ratio |
|----------|------|------------------------|-----------------|
| English | "Find the three largest values in the given list" | ~10 | 1.0× (baseline) |
| Korean | "주어진 목록에서 가장 큰 값 세 개를 찾아라" | ~16 | 1.6× |
| Chinese | "找到列表中最大的三个值" | ~12 | 1.2× |
| Thai | "หาค่าที่มากที่สุดสามค่าในรายการที่กำหนด" | ~25 | 2.5× |

More tokens = longer path through the encoder = more opportunity for representation drift. If PRH convergence assumes equivalent token-level information density, the token tax introduces a systematic asymmetry: English inputs reach Z via a shorter, more efficient path than Thai inputs expressing the same content.

**Implications for Z convergence.** Spacing variation introduces noise at the tokenization stage that should be irrelevant at the semantic level. This creates a testable prediction:

**Prediction 7 (Spacing Robustness).** For a fixed semantic content, Z_semantic distance between spacing variants within a language should be smaller than Z_semantic distance between semantically different inputs with identical spacing patterns. Formally: for spacing variants a, b of the same sentence and a semantically different sentence c with the same spacing as a:

    d(E(a), E(b)) < d(E(a), E(c))

This should hold *if* the encoder successfully abstracts over spacing — but may fail for tokenizers with poor spacing robustness, especially for languages where spacing is irregular (Korean) or absent (Chinese, Thai).

*Protocol:* Take the 50 computational descriptions from §5.6 in Korean; generate four spacing variants each (correct, informal, erroneous, no-space); embed all variants; measure intra-meaning d(a,b) vs. inter-meaning d(a,c). Repeat for Chinese with alternative segmentation models. Compare spacing robustness across encoder models (SONAR, StarCoder-2, frontier models).

**The spacing paradox for code.** An ironic asymmetry emerges: code has *perfect* spacing robustness (the lexer resolves all ambiguity), while NL has imperfect robustness (the tokenizer is spacing-dependent). This means the Chomsky wall, which we argue dissolves at Z, partially *reconstitutes* at the tokenization layer — one more surface-form constraint that leaks into representation space.

**The ideal encoder behaves like a Korean speaker, not a BPE tokenizer.** A Korean reader effortlessly recovers meaning from "주어진목록에서가장큰값세개를찾아라" — using morphological knowledge to reconstruct word boundaries. The ideal encoder E should do the same: map all spacing variants of the same meaning to the same Z, using learned morphological and semantic structure rather than surface-level whitespace. This is exactly what representation-level processing promises — but current tokenizers, trained predominantly on well-spaced English text, fall short.

Full wall dissolution requires not only representation-level convergence but also **spacing-invariant tokenization** — an encoder that maps all spacing variants to the same Z. Morpheme-aware tokenizers (e.g., Korean morphological analyzers like Mecab-Ko, Kiwi), character-level encoders, and byte-level models (ByT5, CANINE) offer partial solutions. Byte-level models are particularly promising because they bypass tokenization entirely — but at the cost of longer sequences and higher compute. The tradeoff between spacing robustness and computational efficiency is itself a design constraint for representation-level systems.

### 4.3 Disambiguation Migration

The Chomsky wall's utility function is *disambiguation*. Removing the wall does not remove the need for disambiguation — it **migrates** the cost.

| | Surface-form programming | Representation-level execution |
|---|---|---|
| **Who disambiguates** | Programmer, at write time | System, at Z |
| **How** | Type-2 grammar constraints | Clarification, probabilistic ranking, or verification at Z |
| **Cost** | Learning curve, syntactic overhead | Inference latency, verification complexity, Z opacity |
| **Failure mode** | Syntax error (immediate, precise) | Semantic misinterpretation (delayed, diffuse) |

**Running example.** "가장 큰 값 세 개를 찾아라" contains at least four ambiguities:

1. *Type ambiguity*: What type of values? (integers, floats, strings, objects?)
2. *Comparison ambiguity*: "가장 큰" — largest by what criterion? (magnitude, absolute value, custom comparator?)
3. *Boundary ambiguity*: What if the list has fewer than three elements?
4. *Output ambiguity*: Should the result be sorted? In what order?

In Type-2 programming, the programmer resolves all four by writing `sorted(lst, reverse=True)[:3]` — the type system handles (1), the comparison operator handles (2), slicing handles (3) silently, and the output order is determined by the sort. In representation-level execution, these ambiguities must be resolved at Z — either by the system (inferring from context) or by querying the user.

#### 4.3.1 Information-Theoretic Formalization

We formalize the disambiguation conservation conjecture using Kolmogorov complexity.

**Definition 6 (Specification Complexity).** For a computational task T with unique intended behavior B(T), the *specification complexity* K(T) is the Kolmogorov complexity of the minimal description that uniquely determines B(T).

**Theorem (Informal — Disambiguation Conservation).** For any interface I that accepts input S and produces execution of task T:

    K(T) ≤ H(S) + H(T|S) + O(1)

where H(S) is the Shannon entropy of the surface form and H(T|S) is the residual ambiguity of the task given the input. The total specification burden K(T) is invariant across interfaces — what changes is the *partition* between H(S) (syntactic precision) and H(T|S) (remaining ambiguity requiring disambiguation).

**Corollary.** Type-2 interfaces maximize H(S) (forcing the programmer to be syntactically precise) and minimize H(T|S) (leaving little ambiguity). NL interfaces minimize H(S) (allowing natural expression) but face H(T|S) > 0 (residual ambiguity that must be resolved somewhere).

This connects to Piantadosi et al. (Cognition, 2012), who argue that ambiguity in natural language is a *feature* of efficient communication, not a bug — context resolves ambiguity at low cost. Our conservation principle extends this insight to the NL-code boundary: ambiguity is efficient for NL communication, but the resolution cost cannot be avoided when exact computation is required.

*Sketch of argument.* K(T) is a property of the task, not the interface. By the data processing inequality, no encoding of T can contain more information about B(T) than K(T) bits. A Type-2 surface form S₁ encodes K(T) bits directly via syntactic constraints: the programmer writes `sorted(lst, reverse=True)[:3]`, which leaves H(T|S₁) ≈ 0. An NL surface form S₂ = "가장 큰 값 세 개를 찾아라" carries fewer bits of specification: H(S₂) < H(S₁), so H(T|S₂) > 0. The remaining H(T|S₂) bits must be provided through disambiguation — context inference, clarification queries, or default assumptions.

**Implication.** Representation-level systems do not reduce the total information needed to specify a program. They *redistribute* it: less from the user upfront (lower cognitive load on syntax), more from the system at Z (higher inference and verification cost). The question is whether this redistribution is net beneficial — and for whom.

This connects to Rate-Distortion Theory (Shannon, 1959): there is a minimum rate (specification density) below which the distortion (semantic error) exceeds acceptable thresholds. Type-2 grammars operate at zero distortion with high rate; NL interfaces operate at lower rate with nonzero distortion that must be corrected downstream.

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

**Running example.** The user says "주어진 목록에서 가장 큰 값 세 개를 찾아라" — whether typed (with or without correct spacing), spoken (zero spacing, no punctuation), or scanned from a handwritten spec (OCR artifacts). The system:
1. Normalizes the surface form (spacing recovery, OCR correction — §4.2.1)
2. Encodes to Z (SONAR-like encoder, language-agnostic, modality-agnostic)
3. At Z, detects residual ambiguity: type and comparison criterion are underspecified (H(T|S) > 0)
4. Checks cultural context: is "가장 큰" purely numerical (Z_semantic) or contextually loaded (Z_proc)? (§3.5.1, §6.4.1)
5. Either: (a) infers from context (if a typed variable is in scope), (b) asks "어떤 기준으로 비교할까요?" (By what criterion?), or (c) applies a default (numeric magnitude) with an annotation that the default was assumed
6. Projects to `heapq.nlargest(3, lst)` with provenance link back to Z

Note that SONAR already supports speech as an input modality — "주어진목록에서가장큰값세개를찾아라" as continuous speech can be encoded directly to the same representation space as text, bypassing both spacing and tokenization entirely. This is the strongest case for representation-level execution: the spacing problem (§4.2.1) disappears when the input modality has no spacing to begin with.

### 5.2 Existing Pieces

- **Constrained decoding** (Outlines, Guidance, LMQL): Projection from Z → Type-2, made deterministic by grammar constraints. Solves non-determinism but requires specifying the target grammar.
- **Tool use / function calling**: NL intent → structured function calls. A narrow form of Z → execution.
- **Program synthesis with verification** (AlphaCode): Multiple candidates from NL, verified against formal spec. Hybrid Z-generation / Type-2-verification.
- **CoRE's LLM-as-interpreter**: Direct execution of structured NL. Closest to representation-level execution but inherits non-determinism.
- **Meta's Large Concept Models (LCM)**: Operates on sentence-level SONAR embeddings rather than tokens, predicting the next "concept" in language-agnostic representation space. SONAR supports 200 languages and multiple modalities. Architecturally closest to the proposed pipeline: SONAR embeddings serve as a concrete Z candidate. Key limitation: LCM performs generation, not execution.
- **Latent execution systems**: LaSynth (Meta, NeurIPS 2021) learns latent representations to approximate execution of partially generated programs. Latent Program Network (LPN, NeurIPS 2025 spotlight) learns a latent space of implicit programs and executes via neural decoder — the closest existing system to Wave 4. COCONUT / Chain of Continuous Thought (Meta, 2024) bypasses language tokens entirely, reasoning in continuous latent space. Code World Models (CWM, Meta FAIR, 2025) models program execution trajectories, tracking variable states and predicting outputs — but operates at the token level, not in SONAR space. These systems demonstrate that execution in representation space is *feasible* in limited domains; the gap is unifying them with PRH-predicted convergence and cross-lingual intent.
- **Neuro-symbolic systems** (DeepProbLog, Scallop, Lobster): Combine neural perception with symbolic reasoning. Lobster (ASPLOS 2026) achieves 3.9× speedup over Scallop on GPU. These systems bridge continuous representations and discrete logic — directly relevant to the Z → verification pipeline. The key insight from this literature: the neural-symbolic interface requires a *differentiable relaxation* of discrete constraints, which may inform how verification at Z could work without projecting fully to Type-2.

### 5.3 Open Problems

1. **Determinism at Z.** How to guarantee same NL → same execution result? PRH convergence ≠ identity (§3.6). Temperature-zero decoding helps but doesn't eliminate stochasticity.

2. **Verification at Z.** Current verification requires Type-2 input. Can we verify properties of programs in continuous embedding space? Connects to neural program verification and neuro-symbolic reasoning. The neuro-symbolic literature suggests a path: define verification predicates as differentiable functions over Z, using techniques from neural theorem proving (Loos et al., 2017) and differentiable logic programming (Manhaeve et al., 2018).

3. **The grounding problem.** PRH convergence is statistical. Programming requires exact grounding: `x > 5` means exactly "strictly greater than 5." What representational precision suffices?

4. **Compositionality.** Code composes via strict nesting (call graphs); NL composes via loose reference ("do that again but with the other file"). If execution requires decomposing NL into composable sub-intents, which compositional structure governs?

    This can be framed categorically. Programs form a category **Prog** (objects = types, morphisms = functions, composition = function composition). NL descriptions of computations form a looser structure — at best a *semicategory* (composition is not always defined: "sort the list, then do it again but differently" does not compose cleanly). The question is whether there exists a functor F : **NL_comp** → **Prog** that preserves the compositional structure of the computational subcategory of NL. The denotational semantics connection (§3.2.1) suggests that such a functor exists for the fragment of NL that has well-defined denotations — but this fragment may be smaller than expected, especially for complex multi-step specifications.

    Practically, this means representation-level execution is most feasible for *compositionally transparent* NL — instructions where the compositional structure is explicit ("first sort, then take the top three") — and hardest for *compositionally opaque* NL ("make it work like the old version but faster").

5. **The specification problem.** If the user writes NL → system maps to Z, the user cannot inspect Z. This is a new opacity: not "I can't read the code" but "I can't read the intent representation."

6. **Stratification and cultural invariance.** Execution must choose which Z layer to operate on. If Z_semantic carries cultural inflection for judgment-involving domains (§3.5), the "Platonic ideal" is a family Z(c), not a single Z.

7. **Surface-form invariance.** The encoder must map all surface-form variants of the same meaning to the same Z — including spacing variants (§4.2.1), OCR artifacts, ASR outputs, and typographic variation. Current tokenizers fail this requirement for many languages. Byte-level and morpheme-aware encoders are partial solutions, but a general spacing-invariant encoder for all writing systems is an open problem.

### 5.4 A Possible Architecture

```
User input (any surface form, any language, any modality)
        │
        ▼
┌──────────────────────┐
│  Surface-form        │  Spacing normalization (§4.2.1)
│  normalization       │  Morpheme-aware tokenization or byte-level encoding
│                      │  OCR post-correction, ASR boundary recovery
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Encoder to Z        │  e.g., SONAR encoder (200 languages, text/speech/image)
│  NL, code, spec →    │  Maps to sentence-level concept embeddings
│  shared representation│  (language-agnostic, modality-agnostic, spacing-invariant)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Cultural context    │  Detect D_train bias (§3.5.1); apply Z(c) if needed (§6.4.1)
│  + Verification at Z │  Neural verifier: is Z complete? unambiguous?
│  + Disambiguation    │  Cost: H(T|S) bits of clarification (§4.3.1)
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

To assess progress toward representation-level execution, we propose seven measurable criteria:

| Criterion | Metric | Current Baseline | Target |
|-----------|--------|-----------------|--------|
| **Convergence** | Mean NL-code cosine similarity | 0.77 MRR (UniXcoder, 2022) | > 0.95 MRR |
| **Determinism** | Same NL → same output rate (temp=0) | ~0.6 (estimated, GPT-4) | > 0.99 |
| **Cross-lingual invariance** | σ of NL-code sim. across NL languages | Not yet measured | σ < 0.05 |
| **Verification at Z** | % of properties verifiable without projection | 0% (no system exists) | > 50% |
| **Disambiguation efficiency** | Clarification queries per task | N/A | ≤ syntax errors per equivalent Type-2 task |
| **Spacing robustness** | Z distance between spacing variants / Z distance between different meanings | Not yet measured | > 10× (spacing variation ≪ semantic variation) |
| **D_train invariance** | Cross-model Z_sem agreement for computational tasks | Not yet measured | > 0.9 cosine similarity |

The spacing robustness criterion (§4.2.1) and D_train invariance criterion (§3.5.1) are novel: no existing benchmark measures them, yet they are critical for representation-level execution across writing systems and model ecosystems.

### 5.6 Pilot Experiment Design

We propose a concrete experiment to test Predictions 1 and 2 simultaneously.

**Stimuli.** Two sets of operation descriptions:
- **Computational set (C):** 50 operations with unambiguous denotational semantics (sort ascending, filter positives, compute mean, reverse list, find maximum, count elements, ...).
- **Judgment set (J):** 50 operations requiring cultural or contextual judgment (prioritize tasks, evaluate quality, summarize appropriately, select the best option, assess risk, ...).

Each operation is described in five typologically diverse languages: English, Korean (한국어), Mandarin Chinese (中文), Arabic (العربية), and Spanish (Español). Descriptions are produced by native-speaker computational linguists to ensure naturalness (not translation artifacts). Total: (50 + 50) × 5 = 500 descriptions.

**Models.** Three embedding models at different scales:
- SONAR (Meta, sentence-level, 200 languages)
- StarCoder-2 Embeddings (code-specialized, 600+ languages)
- Frontier model embeddings (e.g., Claude or GPT-4 internal representations via probing)

**Protocol.**
1. Embed all 500 descriptions in each model's representation space.
2. For each operation o_i, compute the *intra-operation cross-lingual centroid distance*:
   d_intra(o_i) = mean pairwise cosine distance among the 5 language embeddings of o_i.
3. For each language l_k, compute the *inter-operation within-language distance*:
   d_inter(l_k) = mean pairwise cosine distance among all 100 operation embeddings in l_k.
4. Compute the *discriminability ratio*: R = d_inter / d_intra. R > 1 means cross-lingual same-operation similarity exceeds within-language different-operation similarity.

**Predictions.**
- **P1 (Scale-Convergence):** R increases monotonically with model scale across all three models.
- **P2 (Cross-Lingual Semantic Invariance):** R_C > R_J — the discriminability ratio is higher for the computational set than the judgment set.
- **Stratification signal:** d_intra should be near-zero for computational operations in Z_semantic but nonzero for judgment operations, reflecting cultural Z_proc/Z_prag divergence.

**Controls.**
- Scrambled descriptions (preserve words, destroy semantics) as a lower bound.
- Literal translations (Google Translate) vs. native descriptions to detect translation artifacts.
- Within-language paraphrases to establish an upper bound on d_intra for semantic-preserving variation.

**Extension for Prediction 6 (D_train Sensitivity).** Repeat the protocol above using embedding models with known D_train divergence:
- HyperCLOVA X (Korean-dominant D_train)
- Qwen (Chinese-dominant D_train)
- GPT-4 / Claude (English-dominant D_train)

For computational operations with implicit criteria (e.g., "정렬하라" / sort — where default order may vary), measure whether d_intra across *models* exceeds d_intra across *languages within the same model*. If D_train matters, cross-model divergence will be nonzero even for the "purely computational" set — directly testing whether Z_semantic convergence holds across D_train or only within a given D_train.

**Extension for Prediction 7 (Spacing Robustness).** For the Korean subset (50 computational + 50 judgment descriptions), generate four spacing variants per description: (a) correct 띄어쓰기, (b) common informal spacing, (c) spacing errors, (d) no spacing. Total: 100 × 4 = 400 Korean variants. For the Chinese subset, generate two segmentation variants per description using different segmentation models (jieba vs. pkuseg). Measure:
- d_spacing(o_i) = mean Z distance between spacing variants of the same operation
- d_semantic(l_k) = mean Z distance between different operations in the same spacing condition
- Spacing robustness ratio: R_spacing = d_semantic / d_spacing. R_spacing > 1 means the encoder is spacing-robust.

Compare R_spacing across models: byte-level models (ByT5) should exhibit higher R_spacing than BPE-based models, confirming that tokenization architecture mediates spacing robustness. Additionally test with OCR-degraded English inputs (scanned document with justified text → Tesseract OCR output) to verify that English spacing robustness also varies with input quality.

**Extension for speech/voice input.** Speech transcription introduces a zero-spacing condition: ASR (automatic speech recognition) output may omit or misplace word boundaries entirely, especially for agglutinative languages. A supplementary condition uses Whisper transcriptions of spoken descriptions (no punctuation, no spacing) as input. If the encoder is truly spacing-invariant, ASR output should map to the same Z as well-formatted text. This tests the extreme case of the spacing robustness prediction.

**Feasibility.** This experiment requires no new models or training — only inference through existing embedding APIs. Estimated cost: < $500 in API calls (base experiment), ~$1,500 with all extensions (D_train, spacing, voice). Estimated timeline: 3–4 weeks including stimulus preparation and voice recording.

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

#### 6.4.1 Safety and Alignment Implications

Culturally-stratified Z creates a specific alignment challenge for representation-level execution. If a system executes computational intent at Z without explicit cultural parameterization, it will *silently impose* the cultural defaults encoded in its Z_proc and Z_prag layers — predominantly those of its English-language training data.

**Running example.** "주요 고객을 선별하라" ("Select key customers"). Z_semantic converges: the operation is *filter*. But "주요" ("key/important") in Korean business culture may weight 관계 (relationship/guanxi) — longstanding partnership, referral networks, personal trust — while the same operation in an American context may weight metrics (revenue, growth rate, engagement score). A system executing at Z without cultural awareness will default to whichever criterion is more prevalent in training data — likely the American metric-based approach, even when processing Korean input.

This is not a hypothetical concern. Existing evidence shows:
- LLMs produce different moral judgments in different languages (Vida et al., 2024)
- Cultural steering vectors can selectively activate different world-models (Veselovsky et al., 2025)
- The creator ideology of the training corpus propagates through model outputs (Buyl et al., 2025)

**Implication.** Representation-level execution systems must either: (a) make cultural parameterization Z(c) explicit and user-configurable, (b) detect when a task crosses from Z_semantic (culture-invariant) to Z_proc/Z_prag (culture-dependent) and flag it, or (c) restrict representation-level execution to the purely computational domain where Z_semantic convergence holds. Option (c) is the safest near-term path; option (a) is the goal.

#### 6.4.2 Ethical Implications of D_train-Dependent Z

The D_train dependence of Z (§3.5.1) transforms the cultural boundary from a theoretical nuance into a concrete ethical problem with five dimensions.

**1. Computational colonialism.** If representation-level execution becomes widespread, the D_train of the dominant models effectively *defines* what computation means globally. A model trained predominantly on English-language code and documentation encodes English-language computational norms — variable naming conventions, library idioms, error handling patterns, implicit default behaviors — as the "neutral" baseline of Z. Non-English computational traditions are represented as *deviations* from this baseline rather than as equally valid alternatives. This is not merely a bias in outputs; it is a structural bias in the *representation of computational meaning itself.*

The parallel to linguistic imperialism (Phillipson, 1992) is direct: just as English became the "default" language of international communication — not because of intrinsic superiority but because of historical power dynamics — English-derived computational norms are becoming the default of Z through D_train composition, not through any principled claim about computational universality.

**2. The "purely computational" boundary is itself culturally situated.** Our framework distinguishes between Z_semantic (culture-invariant for "purely computational" tasks) and Z_proc/Z_prag (culturally mediated). But §3.5.1 shows that D_train contaminates this boundary: what counts as "purely computational" is itself a judgment that varies across traditions. For example:
- Is **privacy** a computational concern or a cultural one? European GDPR norms treat data minimization as a computational requirement; other traditions treat it as a policy choice.
- Is **fairness in sorting** a computational property? If "sort employees" implicitly means "sort by rank," and rank systems differ across cultures (seniority-based vs. performance-based), the "neutral" sort is culturally loaded.
- Is **error handling** purely computational? The choice between "fail loudly" (exception) and "fail silently" (default value) reflects engineering culture (Erlang's "let it crash" vs. Go's explicit error returns).

The claim that Z_semantic is culture-invariant may be an artifact of analyzing Z through the lens of one specific computational culture.

**3. Consent and transparency.** In Type-2 programming, the cultural assumptions are visible in the code: a programmer can inspect `sorted(employees, key=lambda e: e.rank)` and see that rank is the sort criterion. In representation-level execution from NL, the system resolves "직원을 정렬하라" ("sort the employees") by applying implicit criteria encoded in Z — which are shaped by D_train but invisible to the user. The user has *no mechanism* to inspect which cultural prior was applied, let alone consent to it.

This creates a new form of opacity: not "I can't read the code" (the existing opacity of compiled languages), but **"I can't see which cultural assumptions shaped the interpretation of my intent."** Unlike code opacity, which can be resolved by reading source code, Z-level cultural opacity has no corresponding inspection mechanism — the assumptions are distributed across billions of parameters.

**4. Accountability gap.** When a representation-level system misinterprets culturally-loaded intent, accountability is diffuse:
- The **model developer** chose D_train but cannot predict all cultural interactions.
- The **user** expressed valid intent in their native language but has no control over how Z resolves ambiguity.
- The **system** applied its best interpretation based on D_train but has no model of its own cultural blind spots.

This three-way gap is analogous to the "responsibility gap" in autonomous systems (Matthias, 2004) but specific to cultural interpretation: the harm is not a malfunction but a *correct execution of culturally misaligned intent*.

**5. Feedback loop risks.** If representation-level execution becomes the primary way people specify computation, the outputs (code, behaviors, systems) generated from Z will re-enter D_train for future models. This creates a self-reinforcing loop:

```
D_train (culturally skewed) → Z (culturally biased) → outputs → new D_train → Z' (more biased)
```

This is a cultural analogue of model collapse (Shumailov et al., 2024): just as training on model-generated data degrades statistical diversity, training on Z-projected computation may narrow the diversity of computational approaches — concentrating around the cultural modes most represented in the initial D_train.

**Mitigation paths.** These ethical concerns do not argue against representation-level execution but constrain its design:
- **(M1) Cultural auditing of Z.** Develop tools to probe Z for cultural bias along specific dimensions (comparison criteria, default behaviors, edge case handling). Prediction 6 (§3.7) provides a starting protocol.
- **(M2) Explicit cultural parameterization.** Require Z(c) specification for any operation crossing the computational-judgment boundary. The system should refuse to execute ambiguous intent without cultural context, rather than silently defaulting.
- **(M3) Pluralistic D_train.** Sovereign AI initiatives (Korea, Japan, EU) are independently valuable not only for performance but for creating *culturally diverse Z* that can serve as alternatives to English-dominant models.
- **(M4) D_train transparency.** Analogous to nutrition labels for food, models used for representation-level execution should disclose D_train cultural composition so users can make informed choices about which Z their computation is interpreted through.
- **(M5) Representation-level auditing.** Develop Z-level equivalents of algorithmic auditing: given the same NL input in multiple languages, verify that Z_semantic agrees and flag any divergence for review.

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

The wall is dissolved at input but reassembled at verification. Near-term systems will use a **hybrid architecture**: Z provides expressiveness; Type-2 projection provides verifiability. Full representation-level execution — input, verification, and execution all at Z — requires advances in neural verification. The neuro-symbolic literature (DeepProbLog, Scallop) offers partial tools: differentiable verification predicates that can operate on continuous representations without full projection to Type-2. But these are limited to simple properties; general program verification at Z remains open.

The paradox is not fatal but constrains the deployment path: the Chomsky wall retreats inward, from the user-facing boundary to the system-internal verification layer. Progress is measured by how far inward the wall can be pushed.

### 6.7 Limitations

- PRH is a hypothesis, not a theorem. Convergence evidence is statistical, and sufficiency for exact semantics is unclear.
- "Representation-level execution" is a research direction, not a working system. The open problems in §5.3 are substantial.
- The QWERTY analogy is suggestive but not rigorous. Path dependency dynamics may differ.
- This paper is primarily a position/perspective paper. Empirical validation — especially of the falsifiable predictions (§3.7) — requires future work, though the pilot experiment (§5.6) is designed to be immediately executable.
- The Z stratification (§3.4) is a conceptual framework, not an empirically validated decomposition. Whether the layers are cleanly separable in practice is open.
- The cultural invariance discussion relies primarily on behavioral evidence. Internal mechanistic evidence is still emerging.
- The disambiguation conservation argument (§4.3.1) relies on a Kolmogorov complexity framing that is uncomputable in general. The practical implications hold for typical programs, but edge cases (e.g., programs whose specification is shorter in NL than in code) may violate the informal bound.
- The denotational semantics connection (§3.2.1) is conjectural. Whether Z_semantic truly approximates the semantic domain **D** requires empirical verification — the pilot experiment (§5.6) is a first step.
- The D_train dependence analysis (§3.5.1) suggests that the clean separation of Z_semantic as "culture-invariant" may be an idealization. The extent to which D_train contaminates Z_semantic for computational operations is an open empirical question (Prediction 6).
- The ethical analysis (§6.4.2) identifies structural risks but does not propose solutions with formal guarantees. Mitigation paths M1–M5 are design heuristics, not proven safeguards.

---

## 7. Related Work

**Platonic Representation Hypothesis.** Huh et al. (ICML 2024) established the convergence thesis. Ziyin et al. (2025) provided a formal proof for deep linear networks, identifying six conditions breaking convergence. Gröger et al. (2026) showed global geometric convergence is confounded by scale but local neighborhood agreement persists — a topological rather than geometric interpretation. An information-geometric analysis (NeurIPS 2025 Workshop) frames convergence as Bayesian posterior concentration, proving a "disunion theorem" for models with different approximation capabilities. Cross-domain extensions reach astronomy (NeurIPS 2025), interatomic potentials (2025), and neuroscience (2025).

**Multilingual Representations and Cultural Alignment.** The Semantic Hub Hypothesis (Wu et al., ICLR 2025) demonstrates shared middle-layer representations. Brinkmann et al. (NAACL 2025) show shared grammatical feature directions across typologically diverse languages. However, multilingual ≠ multicultural (Rystrøm et al., 2025): LLMs produce different moral judgments across languages (Vida et al., 2024; Agarwal, 2024; Li et al., 2026), display Cultural Frame Switching (COLING 2025), and reflect creator ideology (Buyl et al., 2025). Veselovsky et al. (2025) identify conserved cultural steering vectors. Naous and Xu (NAACL 2025) trace Western bias to training data. The transfer-localization tradeoff (Wendler et al., 2025) formalizes cross-lingual alignment's tension with cultural preservation.

**Natural Language Programming and Vibe Coding.** Beyond systems surveyed in §2, Cheung (UC Berkeley, 2025) argues LLM code translation preserves syntax but not semantics. IntentCoding (2026) addresses intent amplification via masked decoding. Meta's LCM (2024–2025) operates on SONAR embeddings, the closest system to representation-level execution. CultureManager (2026) introduces task-specific cultural adaptation. The "vibe coding" paradigm (Wave 3) has generated substantial critical analysis: it is characterized as "fast but flawed" with QA frequently skipped (2025), reframed as a shift from deterministic instruction to probabilistic inference (2025), and its non-determinism has been shown to be fundamentally intractable within the current paradigm (ICSE 2026). The Moltbook incident (2026) — 1.5M API keys exposed through a vibe-coded service — validates these critiques empirically.

**Latent Execution and Representation-Level Computation.** LaSynth (Meta, NeurIPS 2021) learns latent representations to approximate program execution. LPN (NeurIPS 2025 spotlight) executes implicit programs via neural decoder in latent space — the closest existing system to Wave 4, achieving 3rd place at ARC Prize 2024. COCONUT (Meta, 2024) bypasses language tokens entirely for reasoning in continuous latent space. CWM (Meta FAIR, 2025) models code execution trajectories at the token level. "Beyond Syntax" (ICSE 2025 NIER) provides direct empirical evidence that LLMs develop language-independent semantic representations of code. These systems demonstrate that latent execution is feasible; our contribution is the PRH-based theoretical framework that unifies them.

**The Naturalness of Software.** Hindle et al. (2012) demonstrated that code is surprisingly repetitive and predictable — comparable to natural language text in its statistical properties. This "naturalness hypothesis" is a precondition for PRH convergence between code and NL: the statistical similarity that makes joint NL-code models possible exists because code, despite its formal grammar, exhibits natural-language-like regularities. Subsequent work (Allamanis et al., 2018) surveys the landscape of machine learning for code, establishing that code's dual nature — formally constrained yet statistically natural — is precisely what positions it as a bridge modality between NL and formal systems.

**Denotational Semantics and Program Meaning.** Scott and Strachey (1971) established that programs can be given meaning as mathematical objects in semantic domains, independent of their syntax. Our conjecture (§3.2.1) that Z_semantic approximates the denotational semantic domain connects PRH to this foundational PL theory. The connection to abstract interpretation (Cousot & Cousot, 1977) is also relevant: abstract interpretation provides a framework for reasoning about programs at different levels of precision, analogous to operations at Z being approximate versions of exact execution.

**Neuro-Symbolic Systems.** DeepProbLog (Manhaeve et al., 2018), Scallop (Li et al., 2023), and NeurASP (Yang et al., 2020) combine neural perception with symbolic reasoning, bridging continuous representations and discrete logic. The neural-symbolic interface problem — how to maintain differentiability across the continuous-discrete boundary — is directly relevant to verification at Z (§5.3, §6.6). Differentiable relaxations of discrete constraints (e.g., Gumbel-Softmax, straight-through estimators) may enable partial verification without full projection to Type-2.

**Tokenization, Spacing, and the Token Tax.** Petrov et al. (NeurIPS 2024) demonstrate that BPE tokenizers introduce systematic unfairness across languages: the same semantic content requires 2–15× more tokens in non-Latin, non-space-delimited languages — the "token tax." This directly affects representation quality and computational cost. Byte-level alternatives (ByT5, Xue et al., 2022; CANINE, Clark et al., 2022) bypass tokenization entirely but at higher compute cost. For Korean specifically, 띄어쓰기 (spacing) is a well-known challenge: morphological analyzers (Mecab-Ko, Kiwi) provide spacing-robust tokenization but are language-specific. Chinese word segmentation remains an active research area with segmenter-dependent outputs affecting downstream NLP tasks. OCR post-processing (Nguyen et al., 2021) addresses surface-form corruption but typically operates at the character level, not the representation level. These findings motivate our tokenization bottleneck analysis (§4.2.1) and Prediction 7 (spacing robustness).

**Emoji, Symbols, and Cross-Cultural Semantics.** Unicode emoji provide a natural laboratory for studying cross-cultural semantic convergence at the surface-form level. Lu et al. (UbiComp 2016) and Barbieri et al. (ACM Multimedia 2016) demonstrate that emoji semantics vary significantly across cultures and platforms — sentiment ratings for the same emoji differ by up to 3 points on a 5-point scale. This parallels our Z stratification: the referent (Z_sem) converges across cultures, while connotation (Z_proc) and social register (Z_prag) diverge. Mathematical notation history (Cajori, 1928–1929) shows that formal symbols achieved Z_semantic convergence through centuries of standardization — a manual process that PRH predicts neural networks accomplish automatically.

**Pragmatic Meaning in NLP.** LLMs perform at or below chance on manner implicatures (Scientific Reports, 2024). Semantic label drift is amplified by LLMs' cultural knowledge in translation (2025). Mathematical reasoning degrades under cultural context changes (2025).

---

## 8. Conclusion

For seventy years, programming languages have been constrained to context-free grammars — not by choice but by the mathematical requirements of deterministic parsing. Three waves of natural-language programming have attempted to bridge the gap; all operate at the surface-form level and inherit the Chomsky wall's constraints.

The Platonic Representation Hypothesis suggests this wall is a property of surface forms, not of representations. Neural networks trained on language and code converge toward shared representations where the Type-1/Type-2 distinction is irrelevant. This reframes the problem from "how do we parse NL as code?" to "how do we execute at the representation level where NL and code are the same thing?"

But the shared representation Z is not monolithic. It stratifies into Z_semantic (convergent), Z_procedural (culturally mediated), and Z_pragmatic (culturally specific). Multilingual LLMs behave as code-switching polyglots, not Platonic oracles. The Aristotelian critique refines this further: what converges is local neighborhood topology, not global geometry. Representation-level execution is most tractable for purely computational domains; judgment-involving domains require cultural parameterization — Z(c) rather than Z.

Moreover, Z is not only a function of scale but of the training distribution D_train. Even Z_semantic — the "purely computational" layer — carries bias from the cultural composition of the training corpus. The clean separation between "culture-invariant computation" and "culture-dependent judgment" is an idealization: default sort orders, null handling conventions, and implicit comparison criteria are culturally situated. This D_train dependence creates ethical risks — computational colonialism, consent opacity, accountability gaps, and cultural feedback loops — that constrain the design of representation-level systems.

The disambiguation cost currently borne by Type-2 grammars does not vanish when the wall dissolves — it migrates to the representation level. We formalize this as an information-theoretic conservation law: the specification complexity K(T) of a task is invariant across interfaces, and reducing syntactic precision (lower H(S)) necessarily increases semantic verification cost (higher H(T|S)). This creates a verification paradox: near-term systems must project back to Type-2 for formal verification, pushing the wall inward rather than eliminating it. Progress is measured by how far inward the wall retreats.

We connect Z_semantic to denotational semantics, conjecturing that representational convergence at scale yields a neural approximation of the semantic domain — explaining both why convergence is stronger for computational tasks and why compositionality is the hard problem. We derive seven falsifiable predictions — including D_train sensitivity, spacing robustness, and the token tax asymmetry across writing systems — propose a pilot experiment (§5.6) that can test several of them with existing tools at minimal cost, and identify concrete evaluation criteria for tracking progress.

The pieces exist — SONAR embeddings, Large Concept Models, constrained decoding, cultural steering vectors, neuro-symbolic verification, program synthesis. They have not been unified under the representation-convergence framework that PRH provides. The research agenda is to unify them — while confronting the cultural stratification of Z and the D_train dependence that determines where representation-level execution can and cannot succeed.

Like QWERTY, formal syntax may be a vestige of a transcended constraint. Unlike QWERTY, the cost of persistence is not ergonomic inefficiency but a fundamental barrier: billions of people can express computational intent in natural language but cannot program. The wall need not fall everywhere — but where Z_semantic converges, it need not stand. And we must ensure that the Z through which that convergence is measured does not silently privilege one culture's computational norms over all others.

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

[43] Hindle, A. et al. "On the Naturalness of Software." ICSE 2012.

[44] Allamanis, M. et al. "A Survey of Machine Learning for Big Code and Naturalness." ACM Computing Surveys, 2018.

[45] Scott, D. and Strachey, C. "Toward a Mathematical Semantics for Computer Languages." Oxford Programming Research Group Technical Monograph PRG-6, 1971.

[46] Cousot, P. and Cousot, R. "Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints." POPL 1977.

[47] Manhaeve, R. et al. "DeepProbLog: Neural Probabilistic Logic Programming." NeurIPS 2018.

[48] Li, Z. et al. "Scallop: A Language for Neurosymbolic Programming." PLDI 2023.

[49] Yang, Z. et al. "NeurASP: Embracing Neural Networks into Answer Set Programming." IJCAI 2020.

[50] Shannon, C. "Coding Theorems for a Discrete Source with a Fidelity Criterion." IRE National Convention Record, Part 4, 1959.

[51] Loos, S. et al. "Deep Network Guided Proof Search." arXiv:1701.06972, 2017.

[52] Phillipson, R. "Linguistic Imperialism." Oxford University Press, 1992.

[53] Matthias, A. "The Responsibility Gap: Ascribing Responsibility for the Actions of Learning Automata." Ethics and Information Technology, 6(3), 175–183, 2004.

[54] Petrov, A. et al. "Language Model Tokenizers Introduce Unfairness Between Languages." NeurIPS 2024. arXiv:2305.15425.

[55] Smith, R. "An Overview of the Tesseract OCR Engine." ICDAR 2007.

[56] Nguyen, T. T. H. et al. "Survey of Post-OCR Processing Approaches." ACM Computing Surveys, 2021.

[57] Xue, L. et al. "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." TACL, 2022.

[58] Clark, J. H. et al. "CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation." TACL, 2022.

[59] "Beyond Syntax: How Do LLMs Understand Code?" ICSE 2025 NIER. ACM, 2025.

[60] Macfarlane, J. and Bonnet, C. "Latent Program Network." NeurIPS 2025 (Spotlight). arXiv:2411.08706.

[61] Hao, S. et al. "Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)." arXiv:2412.06769, 2024.

[62] Chen, X., Song, D., and Tian, Y. "Latent Execution for Neural Program Synthesis Beyond Domain-Specific Languages (LaSynth)." NeurIPS 2021. arXiv:2107.00101.

[63] Meta FAIR. "Code World Models." arXiv:2510.02387, 2025.

[64] Durmus, E. et al. "Towards Measuring the Representation of Subjective Global Opinions in Language Models." NeurIPS 2023. arXiv:2307.07870.

[65] Piantadosi, S. T., Tily, H., and Gibson, E. "The Communicative Function of Ambiguity in Language." Cognition, 122(3), 280–291, 2012.

[66] Portelance, E. "On the Compatibility of Generative AI and Generative Linguistics." arXiv:2411.10533, 2024.

[67] "Vibe Coding as a Reconfiguration of Intent Mediation." arXiv:2507.21928, 2025.

[68] "Reflections on the Reproducibility of Commercial LLM." ICSE 2026. arXiv:2510.25506.

[69] Huang, J. and Yang, Z. "Lobster: GPU-Accelerated Neurosymbolic Framework." ASPLOS 2026. arXiv:2503.21937.

[70] "Beyond Language Boundaries: Uncovering Programming Language Families for Code Language Models." FSE 2026. arXiv:2512.19509.

[71] Lu, X. et al. "Learning from the Ubiquitous Language: An Empirical Analysis of Emoji Usage of Smartphone Users." UbiComp 2016.

[72] Barbieri, F. et al. "How Cosmopolitan Are Emojis? Exploring Emojis Usage and Meaning over Different Languages with Distributional Semantics." ACM Multimedia 2016.

[73] Unicode Consortium. "Unicode Technical Standard #51: Unicode Emoji." unicode.org, 2024.
