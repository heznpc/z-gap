# Experiment V2: Decoder Hidden State Analysis

> 2026-04-04. 근본적 고도화: sentence embeddings → decoder hidden states.
> 설계만. 구현/실행은 별도.

---

## 왜 이 전환이 필요한가

현재 논문은 "Platonic Representation Z"를 논하면서 sentence transformer의 pooled output을 측정한다.
이것은 Z의 proxy가 아니라 shadow다. 6개 구조적 약점이 모두 여기서 파생된다:

| 약점 | 원인 |
|------|------|
| 정체성 혼란 | Z를 직접 측정 못하니 framework 검증 불가 |
| P2 post-hoc | 보고 있는 게 Z가 아니라서 왜 틀렸는지 설명 불가 |
| Strategy A 모델의존 | encoder마다 Z approximation이 다름 |
| One-liner 한계 | sentence embedding이 한 문장만 처리 |
| 6-R 실패 | 방언 차이가 encoder resolution 이하 |
| Encoder-only | Decoder LLM의 hidden state가 실제 Z에 더 가까움 |

**전환 후 논문 정체성**: Empirical paper with theoretical framework.
- Sentence embedding 결과 → pilot/baseline (Section 5.1)
- Hidden state 결과 → main contribution (Section 5.2–5.4)

---

## 1. 모델 선정

### Primary Models (반드시 포함)

> **2026-04-04 업데이트**: CodeLlama-7B 제거 (2023년 모델, Llama 2 기반, 업데이트 없음). Qwen3-Coder 추가.

| Model | Params | Layers | Hidden dim | 특성 | GPU 요구 |
|-------|--------|--------|-----------|------|---------|
| **Llama-3.1-8B-Instruct** | 8B | 32 | 4096 | NL-dominant dense baseline | 1x A100 (16GB fp16) |
| **Qwen2.5-Coder-7B-Instruct** | 7B | 28 | 3584 | Code-specialized, 92 PL | 1x A100 |
| **Qwen3-Coder-30B-A3B-Instruct** | 30B (3B active) | TBD | TBD | 최신 code MoE (2025), expert routing 분석 | 1x A100 |
| **DeepSeek-Coder-V2-Lite-Instruct** | 16B (2.4B active) | 27 | 2048 | MoE, code+NL bilingual | 1x A100 |

### Secondary Models (시간 허용 시)

| Model | Params | 특성 | 이유 |
|-------|--------|------|------|
| Qwen2.5-7B-Instruct | 7B | NL-only Qwen | Code training 효과 분리 (Qwen2.5 vs Qwen2.5-Coder) |
| StarCoder2-15B | 15B | Code-only (no instruct) | Base vs instruct 비교 |
| Qwen3-Coder-Next (80B-A3B) | 80B (3B active) | 최신 agentic code model | Scale + MoE 효과 |

### 모델 선정 근거

**왜 이 4개인가:**
- Llama-3.1-8B: Dense NL baseline. Llama 4는 MoE만 있어 hidden state 비교가 복잡하므로 3.1 유지
- Qwen2.5-Coder vs Qwen3-Coder: 같은 팀, 세대 차이 → code training 발전이 Z에 미치는 영향
- DeepSeek-Coder-V2-Lite: MoE + code 특화. Coder 전용 최신은 V2가 마지막 (V3/V3.2는 general)
- ~~CodeLlama-7B~~: 제거. 2023년 Llama 2 기반으로 너무 구식. Llama-3.1과 아키텍처 비교 목적이 약해짐

**왜 GPT-4/Claude가 아닌가:**
- API로는 intermediate layer hidden states 추출 불가
- Open-weight 모델만 layer별 분석 가능

---

## 2. Stimuli 재설계

### 2.1 복잡도 3-tier 구조

현재 50개 one-liner를 유지하되, 2개 tier를 추가한다.

#### Tier 1: One-liner (기존 50개, baseline)
```
sorted(lst)
max(lst)
[x for x in lst if x > 0]
...
```
유지 이유: 기존 sentence embedding 결과와 직접 비교 가능.

#### Tier 2: Multi-step (신규 30개)
5-15줄 함수. 단일 알고리즘, 명확한 입출력.

| ID | Operation | Code sketch | 복잡도 |
|----|-----------|-------------|--------|
| algo_01 | Binary search | `def binary_search(arr, target): ...` | O(log n) |
| algo_02 | Bubble sort | `def bubble_sort(arr): ...` | O(n²) |
| algo_03 | Merge sort | `def merge_sort(arr): ...` | O(n log n) |
| algo_04 | BFS | `def bfs(graph, start): ...` | O(V+E) |
| algo_05 | DFS | `def dfs(graph, start): ...` | O(V+E) |
| algo_06 | Fibonacci (DP) | `def fib(n): ...` | O(n) |
| algo_07 | Two sum (hash) | `def two_sum(nums, target): ...` | O(n) |
| algo_08 | Palindrome check | `def is_palindrome(s): ...` | O(n) |
| algo_09 | Linked list reversal | `def reverse_linked_list(head): ...` | O(n) |
| algo_10 | Stack-based bracket matching | `def is_balanced(s): ...` | O(n) |
| algo_11 | Counting sort | `def counting_sort(arr): ...` | O(n+k) |
| algo_12 | Matrix rotation 90° | `def rotate_matrix(m): ...` | O(n²) |
| algo_13 | LRU cache (dict+doubly linked) | `class LRUCache: ...` | O(1) per op |
| algo_14 | Topological sort (Kahn's) | `def topo_sort(graph): ...` | O(V+E) |
| algo_15 | Dijkstra (priority queue) | `def dijkstra(graph, src): ...` | O(E log V) |
| algo_16 | Longest common subsequence | `def lcs(s1, s2): ...` | O(mn) |
| algo_17 | Knapsack 0/1 | `def knapsack(W, wt, val): ...` | O(nW) |
| algo_18 | Quick select (k-th element) | `def quick_select(arr, k): ...` | O(n) avg |
| algo_19 | Infix to postfix | `def infix_to_postfix(expr): ...` | O(n) |
| algo_20 | Trie insert + search | `class Trie: ...` | O(m) per op |
| algo_21 | Union-Find | `class UnionFind: ...` | O(α(n)) |
| algo_22 | Sliding window max | `def max_sliding_window(arr, k): ...` | O(n) |
| algo_23 | Interval merging | `def merge_intervals(intervals): ...` | O(n log n) |
| algo_24 | Run-length encoding | `def rle_encode(s): ...` | O(n) |
| algo_25 | Kadane's (max subarray) | `def max_subarray(arr): ...` | O(n) |
| algo_26 | Flood fill | `def flood_fill(image, sr, sc, color): ...` | O(mn) |
| algo_27 | Cycle detection (Floyd's) | `def has_cycle(head): ...` | O(n) |
| algo_28 | Heap sort | `def heap_sort(arr): ...` | O(n log n) |
| algo_29 | Binary tree inorder traversal | `def inorder(root): ...` | O(n) |
| algo_30 | Permutation generation | `def permutations(arr): ...` | O(n!) |

#### Tier 3: Compositional (신규 20개)
여러 알고리즘/자료구조를 조합. "실제 프로그래밍"에 가까움.

| ID | Operation | 구성 |
|----|-----------|------|
| comp_01 | Graph shortest path with negative edges | Bellman-Ford + edge relaxation |
| comp_02 | Word frequency top-k | Tokenize + Counter + heap |
| comp_03 | Expression evaluator | Tokenizer + parser + stack-based eval |
| comp_04 | CSV parser with quoted fields | State machine + string buffer |
| comp_05 | Simple regex matcher | Recursive descent / NFA |
| comp_06 | Task scheduler (dependency) | Topological sort + priority queue |
| comp_07 | In-memory key-value store with TTL | Dict + heap for expiry |
| comp_08 | Text similarity (edit distance) | DP matrix + backtrack |
| comp_09 | JSON path query | Recursive tree traversal + pattern match |
| comp_10 | Rate limiter (token bucket) | Time tracking + bucket logic |
| comp_11 | Balanced BST insertion (AVL) | Rotation + height balance |
| comp_12 | Producer-consumer queue | Thread-safe queue + signaling |
| comp_13 | File system tree (mkdir, ls, cd) | Tree + path parsing |
| comp_14 | Simple HTTP request parser | String splitting + header dict |
| comp_15 | Caching decorator with LRU | Closure + OrderedDict |
| comp_16 | Parallel merge sort | Divide + concurrent merge |
| comp_17 | Bloom filter | Multiple hash + bit array |
| comp_18 | Undo/redo system | Command pattern + two stacks |
| comp_19 | Schema validator (nested dict) | Recursive type checking |
| comp_20 | Event emitter (pub/sub) | Dict of callbacks + dispatch |

### 2.2 NL 기술문 작성 원칙

각 operation에 대해 5개 언어(en, ko, zh, ar, es) 기술문을 작성한다.

**Tier 1**: 기존 유지 (1문장).

**Tier 2**: 2-3문장. 알고리즘의 목적 + 핵심 메커니즘.
```
en: "Sort the array by repeatedly dividing it into halves, sorting each half,
     and merging the sorted halves back together."
ko: "배열을 반복적으로 반으로 나누고, 각 반을 정렬한 뒤,
     정렬된 반들을 다시 합쳐서 배열을 정렬하라."
```

**Tier 3**: 3-5문장. 시스템의 목적 + 구성 요소 + 동작 방식.
```
en: "Build a task scheduler that executes tasks respecting their dependency
     constraints. Each task may depend on other tasks that must complete first.
     The scheduler should detect circular dependencies and execute independent
     tasks in priority order."
ko: "의존성 제약을 지키며 작업을 실행하는 작업 스케줄러를 구축하라.
     각 작업은 먼저 완료되어야 하는 다른 작업에 의존할 수 있다.
     스케줄러는 순환 의존성을 감지하고 독립적인 작업을 우선순위 순서로 실행해야 한다."
```

**작성 방법**:
- en: 저자 직접 작성
- ko: 저자 직접 작성 (번역 아닌 독립 작성)
- zh, ar, es: 네이티브 스피커 or GPT-4o 생성 + 네이티브 검수
  - zh: GPT-4o 생성 가능 (기술 중국어 품질 높음)
  - ar: MSA 기술문은 GPT-4o 가능, 네이티브 검수 필수
  - es: GPT-4o 생성 가능

**Code 구현**: Python 3.10+, 표준 라이브러리만, docstring 없는 bare code.

### 2.3 총 자극 수

| Tier | Operations | Languages | Code | Total NL | Total stimuli |
|------|-----------|-----------|------|----------|--------------|
| 1 (one-liner) | 50 | 5 | 50 | 250 | 300 |
| 2 (multi-step) | 30 | 5 | 30 | 150 | 180 |
| 3 (compositional) | 20 | 5 | 20 | 100 | 120 |
| **Total** | **100** | — | **100** | **500** | **600** |

Judgment ops(기존 50개)는 Tier 2/3에서 제외 — hidden state 분석에서는 computational ops에 집중.
기존 judgment ops는 sentence embedding baseline에 잔류.

---

## 3. Hidden State 추출 파이프라인

### 3.1 입력 형식

각 stimulus를 decoder model에 넣을 때의 prompt template:

**NL descriptions:**
```
<|begin_of_text|>{description}<|end_of_text|>
```
또는 instruct 형식:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{description}<|eot_id|>
```

**Code snippets:**
```
<|begin_of_text|>{code}<|end_of_text|>
```

**중요**: Template는 모델 간 통일하되, tokenizer special tokens는 모델별로 맞춘다.
Instruct 모델은 instruct template, base 모델은 raw text.

### 3.2 추출 대상

각 입력에 대해 모든 transformer layer의 hidden state를 추출한다.

```
Model with L layers → L+1 representations per input:
  h_0: embedding layer output (token embeddings + positional)
  h_1: after layer 1
  h_2: after layer 2
  ...
  h_L: after final layer (pre-LM-head)
```

### 3.3 Token 위치 → Representation 변환

Decoder model은 token 단위로 hidden state를 생성한다.
문장 수준 representation으로 변환하는 전략:

| 전략 | 방법 | 장단점 |
|------|------|--------|
| **Last token** | h[last_token_pos] | Decoder의 autoregressive 특성상 마지막 토큰에 문장 전체 정보 집약. 가장 표준적 |
| Mean pooling | mean(h[all_tokens]) | Encoder 방식. Decoder에서는 초기 토큰이 전체 맥락을 모름 |
| EOS token | h[eos_pos] | Last token과 동일 (EOS가 마지막이므로) |
| **Weighted mean** | attention-weighted mean | Attention weight를 사용해 가중 평균. 계산 비용 높음 |

**Primary**: Last token.
**Secondary (robustness check)**: Mean pooling. 두 전략의 결과가 일치하면 token 선택에 robust.

### 3.4 추출 코드 구조 (pseudocode)

```python
class HiddenStateExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            output_hidden_states=True
        ).to(device)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers

    def extract(self, text: str, pooling: str = "last") -> np.ndarray:
        """Returns array of shape (n_layers+1, hidden_dim)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.hidden_states: tuple of (n_layers+1) tensors,
        # each of shape (1, seq_len, hidden_dim)
        all_layers = []
        for layer_hidden in outputs.hidden_states:
            if pooling == "last":
                vec = layer_hidden[0, -1, :]  # last token
            elif pooling == "mean":
                vec = layer_hidden[0].mean(dim=0)
            all_layers.append(vec.cpu().float().numpy())

        return np.stack(all_layers)  # (n_layers+1, hidden_dim)

    def extract_batch(self, texts: list[str], pooling: str = "last") -> np.ndarray:
        """Returns array of shape (n_texts, n_layers+1, hidden_dim)."""
        # Batch with padding, extract per-text last-token position
        ...
```

### 3.5 저장 형식

```
results/hidden_states/
  {model_name}/
    tier1_nl_{lang}.npz    # shape: (50, n_layers+1, hidden_dim)
    tier1_code.npz         # shape: (50, n_layers+1, hidden_dim)
    tier2_nl_{lang}.npz    # shape: (30, n_layers+1, hidden_dim)
    tier2_code.npz         # shape: (30, n_layers+1, hidden_dim)
    tier3_nl_{lang}.npz    # shape: (20, n_layers+1, hidden_dim)
    tier3_code.npz         # shape: (20, n_layers+1, hidden_dim)
    metadata.json          # model info, extraction params, timestamps
```

**총 저장량 추정** (Llama-3.1-8B 기준, 32 layers, 4096 dim, float32):
- Per tier 1 NL file: 50 × 33 × 4096 × 4 bytes = 27 MB
- Per tier 1 code file: 50 × 33 × 4096 × 4 bytes = 27 MB
- Total per model: (50+30+20) × 5 langs × 33 × 4096 × 4 + 100 × 33 × 4096 × 4 ≈ 3.5 GB
- Total 4 models: ~14 GB

→ float16으로 저장하면 절반. 관리 가능한 수준.

---

## 4. 분석 설계

### 4.1 Layer-wise Convergence Curve (핵심 분석)

**목적**: Z_sem / Z_proc / Z_prag 계층화를 empirically ground.

**방법**: 각 layer l에서 cross-lingual invariance를 측정하고, layer별 curve를 그린다.

```
PSEUDOCODE: layer_convergence_curve(hidden_states, ops, languages, n_layers)

    # hidden_states[op_id][lang] = array of shape (n_layers+1, hidden_dim)

    for l in 0..n_layers:
        # At this layer, compute R(l) = d_inter(l) / d_intra(l)

        d_intra_l = []  # same op, different languages
        for op in ops:
            vecs = [hidden_states[op.id][lang][l] for lang in languages]
            for a, b in combinations(vecs, 2):
                d_intra_l.append(cosine_distance(a, b))

        d_inter_l = []  # different ops, same language
        for lang in languages:
            vecs = [(op.id, hidden_states[op.id][lang][l]) for op in ops]
            for (id_a, a), (id_b, b) in combinations(vecs, 2):
                d_inter_l.append(cosine_distance(a, b))

        R[l] = mean(d_inter_l) / mean(d_intra_l)

    # Plot: x = layer, y = R
    # Expected shape: inverted U
    #   - Early layers (0-5): R low (language-specific features dominant)
    #   - Middle layers (10-20): R peaks (semantic convergence)
    #   - Late layers (25-32): R drops (generation-specific divergence)
```

**예상 결과와 해석**:

```
R(l)
 ^
 |          ****
 |        **    **
 |      **        **
 |    **            ***
 |  **                 ***
 | *                      ***
 +----------------------------> layer
 0    8    16    24    32

 |<-Z_prag->|<--Z_sem-->|<-Z_proc->|
 (언어고유)  (의미수렴)  (생성준비)
```

이 curve가 나오면:
- Z_sem은 더 이상 conceptual이 아니라 **특정 layer 범위에 localize된 empirical entity**
- Peak layer가 NL과 code에서 일치하면 → code는 Z_sem에 접근 가능
- Peak layer가 다르면 → modality별 Z가 다른 위치에 존재

### 4.2 Cross-modal Layer Alignment (NL ↔ Code)

**목적**: NL description과 code implementation이 어느 layer에서 가장 가까운지.

```
PSEUDOCODE: cross_modal_alignment(nl_states, code_states, comp_ids, languages, n_layers)

    for l in 0..n_layers:
        d_match_l = []   # same op: NL(lang, l) vs Code(l)
        d_mismatch_l = [] # diff op: NL(lang, l) vs Code(l)

        for op_id in comp_ids:
            code_vec = code_states[op_id][l]
            for lang in languages:
                nl_vec = nl_states[op_id][lang][l]
                d_match_l.append(cosine_distance(nl_vec, code_vec))

                # mismatch: sample 10 other ops
                for other_id in sample(comp_ids - {op_id}, 10):
                    d_mismatch_l.append(cosine_distance(nl_vec, code_states[other_id][l]))

        R_code[l] = mean(d_mismatch_l) / mean(d_match_l)

    # Plot: x = layer, y = R_code
    # Peak layer = where NL and code representations are most aligned
```

**핵심 질문**: R_code가 peak인 layer와 cross-lingual R이 peak인 layer가 같은가?
- 같으면: Z_sem이 NL-code 수렴과 cross-lingual 수렴을 동시에 매개 → PRH 강력 지지
- 다르면: 두 수렴이 독립적 → PRH 수정 필요

### 4.3 CKA (Centered Kernel Alignment)

**목적**: Layer 간 representational similarity를 모델 간 비교.

```
PSEUDOCODE: cross_model_cka(model_a_states, model_b_states, ops, lang)

    # For a fixed language and operation set,
    # compare layer l of model A with layer m of model B

    for l_a in 0..n_layers_a:
        for l_b in 0..n_layers_b:
            X = stack([model_a_states[op][lang][l_a] for op in ops])  # (n_ops, dim_a)
            Y = stack([model_b_states[op][lang][l_b] for op in ops])  # (n_ops, dim_b)
            cka_matrix[l_a, l_b] = linear_CKA(X, Y)

    # Plot: heatmap of CKA values
    # Diagonal pattern = layer correspondence between models
```

**비교 쌍**:
- Llama vs CodeLlama (same arch, code training 차이)
- Qwen2.5 vs Qwen2.5-Coder (same arch, code training 차이)
- Llama vs Qwen2.5 (다른 arch, 같은 용도)
- NL input vs Code input on same model (modality 차이)

### 4.4 P2 재검증 (Clean Test)

**목적**: Hidden state에서 P2 (cross-lingual invariance)가 성립하는지 layer별로 검증.

```
PSEUDOCODE: p2_per_layer(hidden_states, comp_ids, judg_ids, languages, n_layers)

    # 기존 P2: R_C > R_J (computational ops가 judgment ops보다 cross-lingual invariant)

    for l in 0..n_layers:
        R_C[l] = discriminability_ratio(hidden_states, comp_ids, languages, layer=l)
        R_J[l] = discriminability_ratio(hidden_states, judg_ids, languages, layer=l)

        # Permutation test at each layer
        p_value[l] = permutation_test(R_C[l] > R_J[l], n_perm=10000)

    # Plot: two curves (R_C, R_J) across layers
    # If R_C > R_J at peak semantic layer → P2 holds where it matters
    # If R_C > R_J only at certain layers → P2 is layer-dependent
```

**이것이 기존 P2와 다른 이유**:
- 기존: Sentence embedding (single vector) → R_C < R_J → FAIL
- 신규: Layer별 분석 → "어느 layer에서 P2가 성립하는가?"
- P2가 중간 layer에서만 성립하고 최종 layer에서 실패한다면:
  → "Z_sem 수준에서는 수렴하지만, Z_proc/Z_prag에서 발산" = 논문 thesis 직접 지지
  → Sentence embedding은 최종 layer에 가까우므로 P2 실패가 설명됨

### 4.5 Representational Similarity Analysis (RSA)

**목적**: 100개 operation 간 유사도 구조가 언어와 코드에서 일치하는지.

```
PSEUDOCODE: rsa_analysis(hidden_states, ops, languages, n_layers)

    for l in 0..n_layers:
        # Build RDM (Representational Dissimilarity Matrix) per language
        for lang in languages:
            vecs = [hidden_states[op.id][lang][l] for op in ops]
            RDM[lang][l] = pairwise_cosine_distance(vecs)  # (n_ops, n_ops)

        # Build RDM for code
        code_vecs = [code_states[op.id][l] for op in ops]
        RDM["code"][l] = pairwise_cosine_distance(code_vecs)

        # Cross-lingual RSA: correlation between RDMs
        for lang_a, lang_b in combinations(languages, 2):
            rsa_cross_lingual[l][(lang_a, lang_b)] = spearman(
                upper_tri(RDM[lang_a][l]), upper_tri(RDM[lang_b][l]))

        # NL-Code RSA: correlation between NL RDM and Code RDM
        for lang in languages:
            rsa_nl_code[l][lang] = spearman(
                upper_tri(RDM[lang][l]), upper_tri(RDM["code"][l]))

    # Plot: x = layer, y = Spearman ρ
    # High ρ_cross_lingual + high ρ_nl_code at same layer → Z_sem convergence
```

**RSA가 R ratio보다 나은 점**:
- R ratio: 평균 거리만 비교 (구조 무시)
- RSA: 전체 유사도 구조의 일치도 측정 (어떤 ops가 가까운지의 패턴)
- "Merge sort와 Quick sort가 가까운가?" 같은 의미적 구조가 언어 간 보존되는지를 본다

### 4.6 Linear Probing (P3 확장)

**목적**: 각 layer에서 operation category/identity를 예측할 수 있는지.

```
PSEUDOCODE: layer_probing(hidden_states, ops, languages, n_layers)

    for l in 0..n_layers:
        # Train: English representations at layer l
        X_train = [hidden_states[op.id]["en"][l] for op in ops]
        y_train = [op.category for op in ops]  # or op.id for fine-grained

        # Test: Other languages at layer l
        for lang in ["ko", "zh", "ar", "es"]:
            X_test = [hidden_states[op.id][lang][l] for op in ops]
            y_test = [op.category for op in ops]

            clf = LogisticRegression().fit(X_train, y_train)
            accuracy[l][lang] = clf.score(X_test, y_test)

    # Plot: x = layer, y = cross-lingual probing accuracy
    # Peak accuracy layer = where semantic structure is most language-invariant
```

**기존 P3 (sentence embedding, MiniLM 1개)와 비교**:
- 기존: 90% accuracy, single model, single representation
- 신규: layer별 accuracy curve × 4 models → probe가 어디서 transfer하는지 특정

### 4.7 Tier별 비교 분석

**목적**: 복잡도가 Z 수렴에 영향을 주는지.

```
PSEUDOCODE: tier_comparison(hidden_states, tier1_ops, tier2_ops, tier3_ops)

    for tier_name, ops in [("one-liner", tier1_ops),
                            ("multi-step", tier2_ops),
                            ("compositional", tier3_ops)]:
        R_curve[tier_name] = layer_convergence_curve(hidden_states, ops, ...)
        R_code_curve[tier_name] = cross_modal_alignment(...)

    # Plot: 3 curves overlaid
    # If all tiers show similar peak → complexity doesn't change convergence location
    # If tier 3 peaks later → complex ops need deeper processing
    # If tier 3 shows lower R → complexity hurts cross-lingual invariance
```

---

## 5. 통계 검정

### 5.1 Layer 선택의 다중 비교 문제

32개 layer를 모두 테스트하면 multiple comparison 문제가 발생한다.

**해결**: Layer를 독립 검정하지 않는다. 대신:
1. **Curve fitting**: R(l)에 Gaussian 또는 quadratic curve를 fit하고 peak layer와 CI를 보고
2. **Pre-registered layers**: 분석 전에 "early (0-25%), middle (25-75%), late (75-100%)" 3구간을 정의하고, 구간별 평균 R을 비교 → 3-way comparison only
3. **Permutation-based cluster test**: 연속 layer에서 유의한 차이가 나는 cluster를 찾는 방식 (neuroimaging에서 차용)

### 5.2 모델 간 비교

4개 모델의 peak layer 위치를 비교할 때:
- Bootstrap: 각 모델에서 ops를 resampling → peak layer distribution → CI
- 모델 간 peak layer 차이의 유의성: bootstrap difference test

### 5.3 Effect size 보고

모든 결과에 Cohen's d 또는 η² 보고. P-value만으로는 규모감 없음.

---

## 6. 예측 (Pre-registered)

실험 실행 전에 기록한다. 결과가 이와 다르면 post-hoc임을 명시.

### P-H1: Layer-wise Convergence Shape
**예측**: R(l) curve는 inverted-U shape.
- Early layers (0-25%): R < 1.5 (language-specific)
- Middle layers (25-75%): R > 3.0 (semantic convergence)
- Late layers (75-100%): R < 2.0 (generation-specific divergence)

**Falsification**: R이 monotonically increasing이면 Z_sem/Z_proc 분리가 없음.

### P-H2: NL-Code Peak Alignment
**예측**: R_code(l)의 peak layer와 cross-lingual R(l)의 peak layer는 ±3 layers 이내.

**Falsification**: Peak 차이 > 5 layers → 두 수렴이 독립적.

### P-H3: Code Training Effect
**예측**: CodeLlama의 R_code peak > Llama의 R_code peak (code training이 NL-code alignment 강화).
단, cross-lingual R peak 위치는 유사 (code training이 cross-lingual invariance를 바꾸지 않음).

**Falsification**: Llama의 R_code가 CodeLlama보다 높으면 → code training이 alignment에 불필요 → R_code는 lexical artifact일 수 있음.

### P-H4: P2 Layer-Dependency
**예측**: P2 (R_C > R_J)는 middle layers에서 성립하고 final layers에서 실패.
→ Sentence embedding (≈ final layer)에서 P2가 실패한 이유를 설명.

**Falsification**: P2가 모든 layer에서 실패 → computational/judgment 구분이 Z_sem에서도 없음.

### P-H5: Complexity Effect
**예측**: Tier 2/3의 R peak가 Tier 1보다 deeper layer에 위치.
복잡한 연산일수록 더 깊은 처리가 필요.

**Falsification**: 모든 tier에서 peak 동일 → 복잡도가 수렴 위치에 영향 없음 (놀라운 결과).

### P-H6: D_train Ranking 유지
**예측**: Per-language R_code ranking이 sentence embedding 결과와 일치
(en > es > zh > ko > ar).

**Falsification**: Ranking이 완전히 다르면 → sentence embedding의 D_train 효과가 decoder에서 재현 안 됨.

---

## 7. 논문 구조 변경

### Before (현재)

```
5. Toward Representation-Level Execution
   5.1 Method (sentence embeddings)
   5.2 Results (P1-P7)
   5.3 Strategy A, D, 6-R
```

### After (고도화)

```
5. Pilot: Sentence Embedding Analysis
   5.1 Method & Models
   5.2 Results (P2 failure, P3 success, P7, R_code)
   5.3 Limitations of Sentence Embeddings

6. Main Experiment: Layer-wise Hidden State Analysis
   6.1 Models & Stimuli (4 decoders, 100 ops × 3 tiers)
   6.2 Hidden State Extraction
   6.3 Layer-wise Convergence Curve
   6.4 Cross-modal NL-Code Alignment per Layer
   6.5 P2 Revisited: Layer-dependent Cross-lingual Invariance
   6.6 RSA: Structural Similarity across Languages and Code
   6.7 Complexity Effects (Tier Comparison)

7. Discussion
   7.1 Z_sem is Empirically Localizable
   7.2 Convergence ≠ Communicability: Layer-level Evidence
   7.3 The Verification Paradox Revisited
```

---

## 8. Compute 요구사항

| 항목 | 추정 |
|------|------|
| Hidden state 추출 (4 models × 600 stimuli) | ~8h on 1x A100 |
| 분석 (CKA, RSA, probing, statistics) | ~4h on CPU |
| 저장 공간 | ~14 GB (float32) or ~7 GB (float16) |
| GPU 요구 | 1x A100 40GB (또는 2x RTX 4090) |
| API 비용 | $0 (모든 모델 open-weight) |
| NL 기술문 작성 (Tier 2+3, 5 languages) | ~2 weeks |
| **총 calendar time (설계→논문)** | **4-6 weeks** |

### GPU 접근 방안

| 옵션 | 비용 | 비고 |
|------|------|------|
| 대학 클러스터 | $0 | 접근 가능 여부 확인 필요 |
| Lambda Labs A100 | ~$1.10/h × 8h = ~$9 | On-demand |
| RunPod A100 | ~$0.80/h × 8h = ~$6.4 | Community cloud |
| Google Colab Pro+ | $50/mo | A100 보장 안됨, L4만 가능할 수 있음 |
| 로컬 RTX 4090 | $0 (보유 시) | 16GB fp16으로 8B 모델 가능 |

---

## 9. 리스크 및 대응

| 리스크 | 확률 | 대응 |
|--------|------|------|
| R(l) curve가 flat (inverted-U 아님) | 중간 | → Z_sem 위치가 분산되어 있다는 해석. 여전히 layer별 P2 분석은 유효 |
| NL-Code peak과 cross-lingual peak이 완전히 다른 layer | 낮음 | → "두 종류의 수렴은 독립적" → PRH 수정 필요, 더 강한 contribution |
| P2가 모든 layer에서 실패 | 낮음 | → Computational/judgment 구분이 Z에 없음 → judgment ops를 다른 기준으로 재분류 |
| 8B 모델이 다국어 처리를 잘 못함 | 낮음 | → Qwen2.5가 다국어 강함. Llama-3.1도 100+ 언어 지원 |
| Tier 2/3 NL 기술문 품질 불균일 | 중간 | → 엄격한 검수 프로토콜. 한 언어라도 품질 미달이면 해당 언어 제외하고 보고 |

---

## 10. 기존 인프라와의 관계

### 유지

- `experiments/src/stimuli.py`: Tier 1 ops 유지. Tier 2/3 추가.
- `experiments/src/metrics.py`: cosine_distance, R ratio 함수 재사용. Layer 인자 추가.
- `experiments/src/predictions.py`: P2 test 구조 재사용. Layer loop 추가.
- `experiments/data/stimuli/`: JSON 구조 유지. `tier2/`, `tier3/` 디렉토리 추가.

### 신규

- `experiments/src/hidden_states.py`: HiddenStateExtractor class
- `experiments/src/layer_analysis.py`: Layer-wise convergence, CKA, RSA
- `experiments/src/probing.py`: Linear probing per layer
- `experiments/scripts/run_v2_extract.py`: Hidden state extraction pipeline
- `experiments/scripts/run_v2_analysis.py`: All layer-wise analyses
- `experiments/results/hidden_states/`: Layer-wise representation cache

### Deprecated (삭제 아닌 보존)

- Sentence embedding 결과 전체: pilot section으로 잔류
- Strategy A, D, 6-R: sentence embedding 맥락에서만 유효. Hidden state 분석에서는 대체됨

---

## 11. 실행 순서 (구현 시)

```
Phase 0: 설계 확정 (현재 문서) ← 여기까지 완료

Phase 1: Stimuli (2 weeks)
  1a. Tier 2 code 구현 (30개 Python 함수)
  1b. Tier 3 code 구현 (20개 Python 모듈)
  1c. Tier 2/3 NL 기술문: en, ko 직접 작성
  1d. Tier 2/3 NL 기술문: zh, es GPT-4o 생성
  1e. Tier 2/3 NL 기술문: ar 네이티브 작성/검수
  1f. 전체 검수 + JSON 저장

Phase 2: Infrastructure (1 week)
  2a. HiddenStateExtractor 구현
  2b. Layer-wise metrics 구현 (R, RSA, CKA)
  2c. Probing pipeline 구현
  2d. Visualization functions (layer curves, heatmaps)
  2e. 단위 테스트 (small model로 파이프라인 검증)

Phase 3: Extraction (1-2 days, GPU)
  3a. 4 models × 600 stimuli → hidden states 추출
  3b. 캐시 저장 + 무결성 검증

Phase 4: Analysis (1 week)
  4a. Layer-wise convergence curves
  4b. Cross-modal alignment per layer
  4c. P2 per layer
  4d. RSA per layer
  4e. CKA cross-model
  4f. Probing per layer
  4g. Tier comparison
  4h. Statistical tests + effect sizes

Phase 5: Paper rewrite (1-2 weeks)
  5a. Section 5 → pilot
  5b. Section 6 → main experiment
  5c. Figures + tables
  5d. Discussion update
  5e. Abstract/intro/conclusion revision
```
