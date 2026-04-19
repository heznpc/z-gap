# P2 Failure Interpretation — Experiment Design Audit Report

> 2026-04-04. 설계 감사 결과 기록. 재설계 전 참조용.

## 배경

P2 예측 (R_C > R_J)이 실패함 (MiniLM: R_C=1.91, R_J=3.79). TODO.md에 6가지 해석 전략이 나열됨.
4개 전략(#1, #2, #4, #6)에 대해 실험 설계를 진행했으나, 독립 감사에서 **전 전략에 치명적 결함** 발견.

## 전략별 치명적 결함

### Strategy 1: English-Pivot (어휘 국제성)

- **비라틴 스크립트 간 메트릭 무의미**: Token overlap, romanization similarity 모두 ko-zh, ko-ar, zh-ar 쌍에서 ≈ 0 (스크립트 비공유). 5개 언어 중 3개가 비라틴 → 메트릭의 60%가 노이즈.
- **English-pivot cosine ↔ d_intra 순환성**: en-X 쌍이 d_intra의 40%를 구성. English-pivot cosine과 d_intra의 상관은 부분적 항등식.
- **"artifact" vs "genuine phenomenon" 혼동**: 어휘 차이가 P2를 설명해도, 그것 자체가 communicability gap의 실제 발현일 수 있음. "artifact"라는 결론은 non sequitur.
- **Baron & Kenny mediation 부적절**: n=100 비독립 관측, binary predictor, Sobel test 파워 부족.
- **다중비교 미보정**: 8개 검정, FWER ≈ 0.34.

### Strategy 2: Sparse Language Dimensions (non-language subspace 투사)

- **효과 크기 문제**: 384차원에서 최대 4개 제거 → cosine distance 변화 ≈ arccos(√(380/384)) ≈ 5.7°. 측정 노이즈보다 작을 가능성.
- **R_J - R_C gap 보존 미검증**: R_C, R_J 모두 상승하면 P2 실패 그대로. 통과 조건에 gap 축소 여부 없음.
- **Semantic control 1차원**: comp/judg 이진 분류 1개 방향 제거 → 100-way 의미 구조 파괴 불가 → negative control 실패.
- **Double-dipping**: 전체 데이터로 분류기 학습 + 같은 데이터로 평가.
- **Zhong et al. 전이 미검증**: 원 논문은 LLM hidden state 분석. Sentence embedding에 적용 가능 여부 불명.

### Strategy 4: Aristotelian k-NN 재분석

- **메트릭 비독립**: d_intra_J < d_intra_C → kNN_J > kNN_C 자동. kNN은 R과 같은 기저 거리에 의존. "재분석"이 아니라 "재측정".
- **Gröger et al. 범주 오류**: 원 논문은 서로 다른 모델 간 내부 표상 비교. 본 실험은 단일 모델 내 단일 공간 분석 → "Aristotelian" 근거 부재.
- **순열 검정 교환가능성 위반**: 카테고리 내 연산별 난이도 차이 무시. 기술 길이, 어휘 특성이 교란변수.
- **언어 쌍 효과 미분리**: en-es 축이 결과 지배 가능.
- **Hubness 보정 없음**: 고차원 embedding에서 hub point가 kNN 왜곡.

### Strategy 6: Dialect Continuum

- **데이터 0건**: dialect_descriptions 필드에 tags가 잘못 할당됨 (dataclass 생성자 위치 인수 버그). 타입 자체가 list[str] ≠ dict[str, dict[str, str]].
- **`dialectal_continuum` 메트릭 오정의**: "within-dialect"이 실제로는 d_inter (다른 연산 간 거리). 메트릭 정의 자체가 틀림.
- **p_value=0.0 하드코딩**: bootstrap CI 미구현. 검정 불가능.
- **GPT-4o 방언 생성 불가**: 제주어(UNESCO 위기언어), 민난어(비표준화 문어), 걸프 아랍어 — 학습 데이터 부족.
- **임베딩 모델 방언 미학습**: OOV 토큰 → 노이즈.
- **패러프레이즈 대조군 없음**: 방언 거리 vs 단순 재기술 거리 구분 불가.
- **방언/언어 분류 논쟁**: 광동어-보통화, 이집트 아랍어-MSA는 언어학적으로 별개 언어 수준.

## 프로세스 실패 원인

1. **기존 코드 미검증 위임**: Grep 결과만으로 "인프라 ready" 판단 → 설계 에이전트가 잘못된 전제 위에 설계.
2. **설계 에이전트에 adversarial 조건 미포함**: "실패 시나리오"를 요구하지 않음.
3. **도메인 상식 수준 검증 누락**: 비라틴 스크립트 overlap, 4/384 차원 효과 크기, kNN-R 메트릭 독립성 등.
4. **"6/6 완성" 확증 편향**: 전략 수 채우기에 집중, 각 전략의 근본 타당성 미점검.
5. **에이전트 출력 무비판 전달**: 종합 단계에서 sanity check 없이 그대로 사용자에게 보고.

## 재설계 방향

- 전략 수보다 전략 질 우선 (6개 → 2-3개 견고한 전략)
- 설계 전 선행 검증 (메트릭 독립성, 효과 크기 파워)
- 각 전략에 falsification condition 명시
- 기존 코드 직접 검증 후 설계 진행
