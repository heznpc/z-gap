# TODO — Beyond the Chomsky Wall

> 2026-03-21 업데이트. ARR 마감일 정정, 신규 모델/인용 추가.

## Submission Target
- **Venue**: EMNLP 2026 (ARR submission)
- ~~**Cycle**: April 15, 2026~~ → **April 2026 ARR cycle 없음**
- **실제 다음 cycle: May 25, 2026** (March 16 이미 지남)
- **Commitment deadline**: August 2, 2026
- **EMNLP 2026**: October 24-29, Hungary

---

## Before May 25 (ARR May cycle) — 9주 남음

### 모델 확장 (최우선)
- [ ] **Qwen3-Embedding-8B** — MTEB Multilingual 1위 (70.58), 100+ 언어, Apache 2.0. 최강 다국어 baseline
- [ ] **BGE-M3** (BAAI) — multi-functionality/linguality/granularity. cross-lingual retrieval 강점. P2 재검증용
- [ ] **Codestral Embed** (Mistral, `codestral-embed-2505`) — code embedding SOTA. R_code 검증에 필수 (UniXcoder는 2022년)
- [ ] **E5-small / E5-base / E5-large** — 동일 패밀리 다른 스케일. P1 (scale-convergence) 결정적 테스트
- [ ] **jina-embeddings-v3** — 570M, 8192 tokens, multilingual. v2→v3 비교로 P1 보강
- [ ] **OmniSONAR** (Meta, 2026.03) — text+speech+code+math 단일 공간, 1000+ 언어. Zsem 수렴 직접 테스트
- [ ] (선택) Qodo-Embed-1, Code-Embed — code-specific 보조 baseline

### 기존 모델 확장 유지
- [ ] Llama-3.1, Qwen2.5, Gemma-2, Aya-23 (다국어)
- [ ] CodeLlama, DeepSeek-Coder-v2 (코드)

### Operation 확장
- [ ] 100 → 300-500 ops (하위 유형 분석 통계적 검정력 확보)

### P1 retest
- [ ] E5-small/base/large로 clean scale-convergence 테스트 (기존 P1 inconclusive 해결)
- [ ] P6: cross-model D_train sensitivity (GPT-4 vs HyperCLOVA X vs Qwen)

---

### P2 FAIL 해석 보강 — 5가지 전략

**전략 1: English-Pivot 설명** (CLSD, EMNLP 2025 Findings)
- [ ] 인용 추가: arXiv:2502.08638
- [ ] MiniLM 등 retrieval-tuned 모델이 cross-lingual matching 시 English pivot
- [ ] judgment ops = 추상 어휘 ("evaluate", "prioritize") → English gloss 동일 → R_J 높음
- [ ] computational ops = 도메인 어휘 ("transpose", "jeongnyeol") → 다양한 English 형태 → R_C 낮음
- [ ] "P2 failure = English-vocabulary alignment 측정, Zsem convergence가 아님"

**전략 2: Sparse Language Dimensions** (Zhong et al., arXiv:2510.07213)
- [ ] 언어 정체성이 semantics와 직교하는 sparse dimensions에 인코딩
- [ ] computational descriptions가 더 많은 language-specific dimensions 활성화
- [ ] **실험**: non-language subspace에 프로젝션 후 R_C 재계산 → R_C 증가하면 measurement artifact

**전략 3: Description-Execution Split** (SemCoder, NeurIPS 2024, arXiv:2406.01006)
- [ ] SemCoder가 3 levels 모델링: high-level description, local execution, I/O behavior
- [ ] Level (1) description은 발산, Level (2)(3) execution은 수렴
- [ ] "P2 failure은 Zsem이 execution level에서 작동한다는 확인"

**전략 4: Aristotelian k-NN 재분석** (Groger et al., arXiv:2602.14486)
- [ ] global geometric convergence → calibration 후 사라짐, local neighborhood topology 유지
- [ ] distance ratio (R_C, R_J) 대신 **k-NN accuracy**로 P2 재분석
- [ ] "sort" 5개 언어가 mutual k-nearest-neighbors이면 local convergence 존재

**전략 5: Dtrain Mediation** (Semantic Hub, ICLR 2025, arXiv:2411.04986)
- [ ] English-dominant 모델이 다른 언어도 English space에서 처리
- [ ] Zsem(observed) = Zsem(ideal) + bias(Dtrain) — 논문에 이미 있는 수식의 실증적 뒷받침

**전략 6: Modality Completeness + Dialect Continuum** (고맥락 + 비언어적 맥락 + 방언)
- [ ] Code = modality-complete (텍스트 = 전체 의미) → embedding이 의미 전부 포착 → P3 수렴 성공
- [ ] NL = modality-incomplete (텍스트 ≠ 전체 의미) → 억양, 표정, 사회적 위계, 주어 생략 등 비언어적 채널 누락
- [ ] 고맥락 언어(한/일/중)일수록 비언어 의존도 높음 → cross-lingual alignment 더 어려움
- [ ] 의성어/의태어 = 감각/신체 경험 인코딩 → 영어 embedding 공간에 깔끔한 대응 없음 (예: "살금살금")
- [ ] **핵심 논점**: convergence는 명시적 텍스트 채널에서 발생, communicability는 비언어적 채널까지 요구 → convergence ≠ communicability의 직접 근거
- [ ] 관련 인용 조사: high-context communication (Hall, 1976), embodied cognition in NLP
- [ ] **방언 실험**: 언어 경계가 이산적이 아닌 연속적임을 실증
  - 한국어: 표준어 / 경상도 / 전라도 / 제주
  - 아랍어: MSA / 이집트 / 걸프 / 레반트
  - 중국어: 普通话 / 粤语 / 闽南语
  - 영어: American / British / Indian / AAVE
  - 스페인어: Castilian / Mexican / Rioplatense
  - 100 ops × 방언 변이 → R_C, R_J를 언어 간 + 언어 내 이중 계산
  - P2 prediction 자체를 cross-lingual + cross-dialectal로 확장

---

### 신규 인용 추가

**최우선:**
- [ ] **OmniSONAR** (arXiv:2603.16606, 2026.03) — text+code+speech+math 단일 embedding. Zsem 수렴 직접 증거
- [ ] **Semantic Hub Hypothesis** (arXiv:2411.04986, ICLR 2025) — middle layers = modality-agnostic semantic hubs. Zsem/Zprag 분리 지지
- [ ] **Sparse Language Dimensions** (arXiv:2510.07213) — 언어 정체성 sparse encoding. Zsem/Zprag 분리 실증
- [ ] **CLSD** (arXiv:2502.08638, EMNLP 2025 Findings) — cross-lingual semantic discrimination. P2 failure English-pivot 설명
- [ ] **SemCoder** (arXiv:2406.01006, NeurIPS 2024) — description vs execution semantics. P2 description-level 한계 설명

**추가:**
- [ ] **Programming Language Families** (arXiv:2512.19509) — LLM이 PL phylogenetic clustering 학습. code modality 내부 수렴 구조
- [ ] **Byte Latent Transformer** (arXiv:2412.09871, ACL 2025) — tokenizer-free. P7 spacing robustness 극대화 예측
- [ ] **MMTEB** (arXiv:2502.13595) — 500+ tasks, 250+ langs. 모델 선택 validation
- [ ] **e5-omni** (arXiv:2601.03666) — cross-modal alignment 기법
- [ ] **Convergence Without Correspondence** (Sergent, 2026, PhilArchive) — PRH 철학적 비판. communicability 논증 강화

### P7 강화
- [ ] BLT (Byte Latent Transformer) 또는 ByT5로 P7 spacing robustness 테스트
- [ ] tokenizer-free 모델에서 R_spacing >> 2.9 예측 검증

---

### Paper Polish
- [ ] **ARR Responsible NLP Checklist** (필수 — 부정확 기재 시 desk reject)
  - B: 모든 모델 명시 (이름, 라이선스, 의도된 사용)
  - C: computational budget, hyperparameters, 결과 단일 실행 vs 집계 여부
  - E: AI assistant 사용 여부 공개
- [ ] Add 1-2 running examples (date parsing, NER 등)
- [ ] **Double-column appendix 형식 확인** (2025.07부터 위반 시 desk reject)
- [ ] EMNLP 2026 Special Theme "New Missions for NLP Research" — communicability 프레이밍 포지셔닝 검토

---

## After Review (August cycle)
- [ ] Address reviewer feedback
- [ ] Commitment deadline: August 2, 2026
