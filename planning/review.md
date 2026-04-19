# z-gap — Review (2026-04-11)

## 1. 커밋 톤이 주장을 일관되게 지지하는가?

**판정: 매우 일관됨, 본 survey 21개 paper 중 *과학적 정직성과 진화 깊이*가 함께 가장 강한 케이스 (19 commits, 2026-03-18 → 04-07).**

```
577f198 chore: add .zenodo.json for DOI minting                              (2026-04-07)
5f7a05d feat: add 8 missing citations from academic landscape review
ce22836 feat: lexical overlap control for R_code + honest P2 framing
2d28658 feat: Strategy D (20/20 NL-code alignment) and Strategy 6-R (dialect continuum)
53307a2 feat: P2 failure interpretation with vocabulary mediation and language-pair decomposition
2e30f77 feat: revise P1/P2 analysis, add disambiguation bound, prepare COLM 2026 submission
b81524a fix: correct bootstrap sampling to guarantee target sample sizes
8e4e2e6 feat: P2 failure interpretation, date parsing example, NLP checklist
6ef3883 feat: AI-AI communication corollary + math purity revision
618895e rename to Z-Gap: update README, untrack TODO, fix experiments README
3458b59 v7: modality completeness, dialect continuum, model expansion
e002393 chore: clean up TODO and gitignore
06d9a7f v6: EMNLP 2026 position paper — convergence ≠ communicability
f0e98f5 v5: LaTeX migration, pilot results (P2/P3/P7 + NL-code alignment), reference fixes
e6d3746 pilot experiment: P2 cross-lingual invariance + P7 spacing robustness
2a08a0f submission-ready paper: 8-page body + 6 appendices (A-F)
d6c0030 draft v3-v4: formal framework, D_train dependence, spacing analysis, emoji spectrum
5955cc9 draft v2: Z stratification, cultural superposition, and Aristotelian critique
b84a53e Initial commit: Beyond the Chomsky Wall paper                       (2026-03-18)
```

진화 패턴:
- **Phase 1 (3/18 ~ 3/21)**: v1 → v2 → v3-v4 (Z stratification, cultural superposition, formal framework, D_train, spacing analysis, emoji spectrum). 4 draft 진화.
- **Phase 2 (3/22)**: 8-page body + 6 appendices (A-F)로 *submission-ready* 정착.
- **Phase 3 (3/22 ~ 3/27)**: pilot experiment (P2 cross-lingual + P7 spacing) → v5 LaTeX migration → v6 EMNLP position paper → v7 modality completeness + dialect continuum + model expansion.
- **Phase 4 (3/27 ~ 4/4)**: rename to Z-Gap → AI-AI communication corollary → math purity revision → bootstrap sampling fix → P2 failure interpretation 4 strategies → Strategy D 20/20 NL-code alignment → lexical overlap control + **honest P2 framing** → 8 missing citations.
- **Phase 5 (4/7)**: Zenodo DOI.

톤 일관성:
- 핵심 주장(Z = Platonic representation 수렴점 / 4 phenomena: P1 scale, P2 cross-lingual, P3 probing transfer, P7 spacing / NL-code alignment / convergence ≠ communicability)이 *진화 단계마다 정직히 update*.
- **P2 가설이 실패했다는 사실을 paper 안에서 정직히 인정**: report.md에 "P2 (NL-NL cross-lingual): FAIL — description-level only". commit message에 "honest P2 framing", "P2 failure interpretation". **자기 가설 reject되는 데이터를 paper에 통합**. ploidy와 같은 *negative result honesty*.
- **`AUDIT_P2_STRATEGIES.md` (88줄)**: 4개 P2 해석 전략(English-pivot / sparse language dimensions / Aristotelian k-NN / dialect continuum) 각각의 *치명적 결함*을 자기 비판. **본 survey 21개 paper 중 *experiment design audit document를 정식 공개한* 두 번째 케이스 (canary EXPERIMENT-AUDIT.md와 함께)**.
- audit document가 *Process 실패 원인*까지 5개 점으로 분류:
  1. 기존 코드 미검증 위임
  2. 설계 에이전트에 adversarial 조건 미포함
  3. 도메인 상식 수준 검증 누락
  4. "6/6 완성" 확증 편향
  5. 에이전트 출력 무비판 전달
- *meta paper와 narcissus paper의 *empirical anchor*로 직접 사용*. cross-repo data sharing.
- **3개 venue 동시 준비**: main.tex (general), main_colm.tex (COLM 2026), main_emnlp.tex (EMNLP 2026). venue-specific submission 정착.

## 2. 부가 서비스 품질

**판정: 본 survey 21개 paper 중 *학술 실험 코드 인프라*가 *production-grade*인 케이스. ploidy/canary와 함께 코드 TOP 3.**

서비스 구성 (코드 ~7,800줄):
- **`experiments/src/` (4,640줄)** — core library:
  - `stimuli.py` (891) — 100 operations × 5 languages stimulus 데이터셋
  - `metrics.py` (707) — d_intra, d_inter, R, HSD 등 거리 메트릭
  - `vocab_internationality.py` (471) — 어휘 국제성 분석
  - `vocab_mediation.py` (467) — vocabulary mediation 분석
  - `hidden_state_analysis.py` (392) — Llama hidden state 분석
  - `predictions.py` (297) — 4 prediction model
  - `visualize.py` (305), `hidden_state_visualize.py` (196)
  - `analysis.py` (131), `code_alignment.py` (239), `embeddings.py` (146)
  - `hidden_states.py` (236), `report.py` (162)
- **`experiments/scripts/` (~6,000줄)** — runner scripts:
  - `run_all.py` (260), `run_p1_p3.py` (319), `run_punctuation.py` (169)
  - `run_strategy1_vocab.py` (266), `run_strategy2_langpair.py` (677), `run_strategy4_knn.py` (387), `run_strategy4_prereq.py` (549), `run_strategy_6r_dialect.py` (325), `run_strategy_a_vocab.py` (317), `run_strategy_d_code_alignment.py` (277)
  - `run_code_alignment.py` (184), `run_code_alignment_significance.py` (271), `run_rcode_token_control.py` (322)
  - `run_v2_extract.py` (133), `run_v2_quick.py` (262), `run_v2_analyze.py` (265)
  - `run_sparse_language_dims.py` (886) — 가장 큰 단일 분석 script
  - `run_cross_experiment_synthesis.py` (373)
- **`experiments/results/`** — 14개 JSON results + 1 figures dir + embeddings + hidden_states + latex + report.md
- **`experiments/notebooks/v2_colab.ipynb`** — Colab notebook for hidden state extraction
- **`experiments/data/`** — dialect_stimuli.json + stimuli/
- **3 paper variants**: main.tex, main_colm.tex (COLM 2026), main_emnlp.tex (EMNLP 2026, 880줄). 각각 빌드 PDF 존재.
- **`paper/references.bib`** — 673줄. 본 survey 21개 paper 중 가장 많을 가능성.
- **`AUDIT_P2_STRATEGIES.md`** — 4 strategy critical flaw 보고서 (88줄)

품질 평가:
- ploidy / canary와 함께 *production-grade scientific code*. 본 survey 21개 paper 중 *학술 실험 코드 = paper data*의 일치도 TOP 3.
- **20/20 NL-code alignment**, **R_code = 1.04 (UniXcoder)**, **R_code = 1.18 (MiniLM)** — *NL-code alignment는 본 paper의 가장 강력한 finding*.
- **P3 cross-lingual probing**: 90% category transfer + 86% operation transfer (5 languages × 100 ops).
- **P7 spacing robustness**: R = 2.90, p < 0.001.
- **honest negative result**: P1 (scale) NOT SUPPORTED, P2 (NL-NL cross-lingual) FAIL, P6 partial. **paper의 *4개 prediction 중 2개가 실패*하는 상황을 그대로 publish**. 학술적 정직성 매우 강함.
- **3 venue-specific paper**: main_emnlp.tex가 ARR May 25 cycle deadline에 정렬. 매우 진지한 publication strategy.

## 3. 고도화 가능 파트

높은 우선순위:
1. **EMNLP 2026 ARR May 25 submission 마무리** — README가 명시. ~6주 남음. main_emnlp.tex이 880줄로 가장 큼. 8 missing citations(4/4 commit) 반영 확인.
2. **P2 failure의 *deeper diagnosis***: 4 strategies가 모두 fail. AUDIT 문서에 따르면 "전략 수보다 전략 질 우선 (6→2-3개 견고한 전략)". *어떤* 새 전략이 P2 failure를 정직하게 다룰 수 있는지 결정.
3. **Hidden state extraction (V2)** — `EXPERIMENT_V2_HIDDEN_STATES.md` 명시. babel paper의 §5.3 layer-wise diversity가 *이 데이터에 의존*. cross-repo dependency 해결.
4. **Sparse language dimensions의 *재실행***: 384→1024 차원의 더 큰 모델로 effect size 확보. 현재 4/384 → 5.7° 변화는 노이즈 수준.
5. **Cross-model expansion**: 현재 MiniLM + UniXcoder. ploidy처럼 4 model family (Anthropic/OpenAI/Google/xAI) 추가. 모든 결과의 *cross-model 검증*.

중간 우선순위:
6. **Disambiguation bound formalization** — paper §3 정도에 명시되는 듯. *수학적 bound*가 z-gap의 unique theoretical contribution. 더 강하게 정착.
7. **AI-AI communication corollary 강화** — 본 paper의 가장 *novel claim*. AI 끼리의 communication에서도 z-gap이 발생하는지의 검증 실험.
8. **Cross-repo paper trilogy 완성**: z-gap (data) + babel (theory) + ploidy (intervention) + meta (reflexive). 4-paper trilogy의 인용 surface area.
9. **Dialect continuum의 *진짜 dialect data*** — 제주어/민난어/걸프 아랍어 등 GPT-4o가 생성 못 하는 방언을 *native speaker corpus*로 대체.
10. **AUDIT 문서의 *Process 실패 원인*을 meta paper에 통합** — z-gap의 audit가 narcissus/meta의 데이터로 사용 가능.

낮은 우선순위:
11. arXiv preprint.
12. 한국어 abstract.
13. CHI/CSCW 백업 venue.

## 4. 학술적 / 시장 가치 (글로벌, 2026-04-11 기준)

### 학술적 가치: **상위권 (working paper 기준 top ~5%, EMNLP/COLM 한정 시 top 3-5%)**

차별점:
- **"Z-Gap" naming + Beyond the Chomsky Wall**: 인용 가능한 새 vocabulary anchor. *convergence ≠ communicability*라는 1-line 명제.
- **NL-code alignment 강한 결과**: R_code = 1.04 (UniXcoder), R_code = 1.18 (MiniLM). 20/20 NL-code 매핑. **이 finding이 paper의 가장 강력한 anchor**. NL과 code가 같은 representation space에 존재한다는 강한 명제.
- **P3 cross-lingual probing 강한 결과**: 90% category, 86% operation transfer across 5 languages. *Platonic representation hypothesis*의 강한 empirical anchor.
- **P7 spacing robustness**: R=2.90, p<0.001. paper의 가장 깔끔한 결과.
- **Honest P2 failure**: 4개 prediction 중 1개만 정직히 fail. **negative result를 paper에 통합한 *학술적 정직성 모범 사례***. ploidy와 동급.
- **AUDIT_P2_STRATEGIES.md**: experiment design audit를 *공개*. 본 survey 21개 paper 중 canary와 함께 audit 공개 2개 뿐. reviewer 신뢰도 +2.
- **Production-grade experimental code (~7,800줄)**: 본 survey 21개 paper 중 *재현 가능 학술 코드 TOP 3*.
- **3 venue submission strategy**: ARR May 25 (EMNLP), COLM 2026, general. 학술 publication 전략 매우 진지.
- **673줄 references.bib**: 본 survey 21개 paper 중 *가장 많을 가능성*. NL-code interface + Platonic representation + cross-lingual probing 분야의 *광범위* literature integration.
- **babel/meta/narcissus/ploidy와의 cross-repo synergy**: z-gap이 4-paper trilogy의 *data anchor*. babel §5.3 layer-wise diversity, meta paper의 reflexive evidence, narcissus의 citation analysis 모두 z-gap data 인용.

위험:
- **P1 NOT SUPPORTED + P2 FAIL** — 4개 prediction 중 2개 실패. paper를 *positive results paper*로 reframe하면 안 됨. *honest framing*이 가능한지 venue별 차이.
- **P2 4 strategies 모두 critical flaw** — AUDIT 문서가 명시. P2 failure의 *deeper diagnosis*가 paper의 *open question*. reviewer가 짚을 수 있음.
- **Independent researcher 단독 저자** — EMNLP/COLM은 multi-author가 표준. ML/NLP top venue.
- **MiniLM과 UniXcoder만 사용** — cross-model 확장이 약함. ploidy처럼 4 family 비교가 reviewer 신뢰도 +1.
- **Hidden state extraction (V2)**가 미실행. paper의 후속 단계 약속.

게재 전망:
- **EMNLP 2026** (ARR May 25): **realistic, 35-45%**. position paper 트랙으로 적합. *honest negative result + production-grade code*가 reviewer 호의적. P2 failure의 *honest framing*이 venue에 맞으면 통과.
- **COLM 2026**: **35-45%**. ploidy와 함께. NL/cross-lingual 트랙.
- **NeurIPS 2026 / ICLR 2027**: 30-40%. system paper 트랙.
- **TACL** (Transactions of ACL): 35-45%.
- **ACL 2026**: 30-40%.
- arXiv preprint 우선 권장.

### 시장 가치: **상위 (NLP eval + AI safety 영역에서 강함)**

- **NLP eval 회사**: Hugging Face Eval, Stanford CRFM, MLCommons가 *cross-lingual probing benchmark*로 사용 가능. P3 90% finding은 즉시 cross-model evaluation의 baseline.
- **Multilingual LLM 회사**: Anthropic, OpenAI, Google, Cohere가 *cross-lingual representation* 평가에 인용 가능. *5 languages × 100 ops* dataset 그대로 reuse.
- **AI safety 영역**: *convergence ≠ communicability* 명제는 AI safety alignment 담론의 anchor. P2 failure는 *AI-AI communication*의 한계.
- **언론**: NYT/Atlantic/Wired/MIT Tech Review가 좋아할 톤. "AI는 같은 representation space에 살지만 서로 의사소통하지 못한다" 헤드라인.
- **Education / linguistics**: 한국 KAIST, Tsinghua, MILA 같은 NLP lab의 *후속 연구 anchor*. cross-lingual probing 분야의 standard reference.
- **Audit 문서가 *학술 정직성 표준*으로 사용 가능**: 다른 학자가 자기 작품의 audit를 작성할 때 z-gap의 AUDIT을 template으로 reuse.

### 종합 평점 (2026-04-11)

| 축 | 점수 | 코멘트 |
|---|---|---|
| Originality of construct | 9/10 | Z-gap + convergence ≠ communicability |
| Empirical contribution | 9/10 | P3 90%, NL-code R=1.04, P7 R=2.90 |
| Honest negative result | 10/10 | P1/P2 fail을 그대로 publish |
| Code quality | 10/10 | ~7,800줄 production-grade |
| Self-criticism (AUDIT) | 10/10 | 4 strategy critical flaw 공개 |
| Repo health | 10/10 | 19 commits, 3 venue-specific tex, Zenodo |
| Cross-repo synergy | 10/10 | babel/meta/narcissus/ploidy의 data anchor |
| Submission readiness | 9/10 | EMNLP/COLM/general 3 venue 동시 준비 |
| Theoretical depth | 9/10 | disambiguation bound + AI-AI corollary |
| Practical applicability | 9/10 | NLP eval 즉시 사용 |
| Timing | 9/10 | Cross-lingual + Platonic representation 정점 |
| **Overall (research paper)** | **9.0/10** | **본 survey 21개 paper 중 canary/ploidy와 동급 1-3위** |

핵심 격언: **"본 survey 21개 paper 중 *production-grade 학술 실험 코드 + 정직한 negative result + 자기 비판 audit 공개*의 동시 만족도 최상위. EMNLP 2026 ARR May 25 cycle 6주 안에 submission 가능."** canary가 *DSR + service*라면, ploidy가 *empirical system + MCP*라면, z-gap은 *empirical NLP + cross-lingual probing*. **본 survey 21개 paper 중 *technical excellence trilogy*의 마지막 기둥**. Honest P2 failure는 학술적 정직성의 모범이며, AUDIT 문서는 다른 학자에게 *self-criticism template*으로 reuse 가능.
