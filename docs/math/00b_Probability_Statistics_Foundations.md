ㅎㅎ# 00b. 확률과 통계 기초 (Probability & Statistics Foundations)

이번 단원에서는 머신러닝에 필요한 **확률과 통계 기초**를 배워보겠습니다.

## 학습 목표
- 확률 분포 (Probability Distribution)의 개념과 종류
- 조건부 확률 (Conditional Probability)과 베이즈 정리 (Bayes' Theorem)
- 최대우도 추정 (Maximum Likelihood Estimation, MLE)
- 손실 함수 (Loss Function)와 확률 분포의 연결
- 정규화와 통계의 관계

## 왜 이 단원이 필요한가?

머신러닝은 본질적으로 **확률론적 모델링**입니다. 모델은 데이터의 확률 분포를 학습하고, 예측은 확률로 표현됩니다.

- **CrossEntropy Loss**: 왜 분류 문제에서 사용하는가? → 다항 분포 (Multinomial Distribution)와 MLE
- **MSE (Mean Squared Error) Loss**: 왜 회귀 문제에서 사용하는가? → 정규 분포 (Normal Distribution)와 MLE
- **정규화**: 왜 배치 정규화 (Batch Normalization)가 작동하는가? → 통계적 정규화
- **불확실성**: 모델의 예측이 얼마나 확실한가? → 확률 해석

이 단원을 이해하면 손실 함수가 단순한 수식이 아니라 **확률론적 의미**를 가진다는 것을 알게 됩니다.

## 이 단원을 배우기 전에

**이전 단원 (00a) 복습**: 미분, 경사하강법 (Gradient Descent)의 개념을 이해하셨나요?

## 이 단원 다음에는

**다음 단원 (01)**: 수학 기초를 다졌으니, 이제 PyTorch의 텐서 (Tensor)를 배워봅시다.

---
# 1. 직관적 이해 (Why)
---

## 1.1 확률이란 무엇인가?

### 실생활 비유: 날씨 예보

- **확실성**: "내일 해가 뜬다" (확률 = 1)
- **불확실성**: "내일 비가 올 확률은 70%" (확률 = 0.7)
- **확률**: 불확실한 사건이 일어날 가능성을 0~1 사이의 숫자로 표현

### 머신러닝에서의 확률

- **분류 모델**: "이 이미지가 고양이일 확률은 0.85"
- **예측의 불확실성**: 확률이 높을수록 모델이 확신함
- **데이터의 패턴**: 데이터가 따르는 확률 분포를 학습

### 왜 딥러닝에 필요한가?

1. **손실 함수 (Loss Function)의 의미**: CrossEntropy는 확률 분포 (Probability Distribution) 간의 차이
2. **모델의 출력**: 소프트맥스 (Softmax)는 확률 분포로 변환
3. **학습의 목표**: 실제 데이터 분포를 모델이 근사

---
# 2. 수학적 기초 (What)
---

## 2.1 확률의 기본 개념 (고등학교 복습)

### 단계별 이해

#### Step 1: 확률의 정의

사건 A가 일어날 확률:

$$P(A) = \frac{\text{A가 일어나는 경우의 수}}{\text{전체 경우의 수}}$$

**예시**: 주사위를 던질 때
- 6이 나올 확률: $P(6) = \frac{1}{6}$ (1가지 경우 / 6가지 경우)
- 짝수가 나올 확률: $P(\text{짝수}) = \frac{3}{6} = \frac{1}{2}$ (2,4,6 / 6가지)

#### Step 2: 확률의 성질 (3가지 필수 규칙)

1. **0 ≤ P(A) ≤ 1**: 확률은 0과 1 사이 (0% ~ 100%)
2. **P(전체) = 1**: 모든 가능한 경우의 확률 합은 1
3. **P(A ∪ B) = P(A) + P(B) - P(A ∩ B)**: 합사건 확률

**직관**: 확률은 "가능성을 숫자로 표현"한 것입니다.

#### Step 3: 머신러닝에서의 확률

- **분류 모델**: 각 클래스에 속할 확률 예측
- **소프트맥스 (Softmax) 출력**: 모든 클래스 확률의 합 = 1
- **불확실성**: 확률이 낮으면 모델이 불확신

### 확률의 공리적 정의 (Axioms of Probability)

**왜 이 3가지 규칙인가?** 확률은 수학적으로 콜모고로프(Kolmogorov)의 3가지 공리로 정의됩니다:

1. **비음성 (Non-negativity)**: $0 \leq P(A) \leq 1$ - 확률은 0과 1 사이
2. **정규화 (Normalization)**: $P(\Omega) = 1$ - 표본공간 $\Omega$의 확률은 1
3. **가산성 (Countable Additivity)**: $P(A \cup B) = P(A) + P(B)$ (A와 B가 상호 배타적일 때)

**표본공간 (Sample Space)**이란? 가능한 모든 결과의 집합이에요. 주사위를 던질 때 표본공간은 {1, 2, 3, 4, 5, 6}이죠.

### 확률변수 (Random Variable)

**확률변수 $X$란 무엇인가요?** 확률 실험의 결과를 숫자로 표현한 거예요.

**중요한 구분**:
- **확률변수 $X$의 값**: 실험 결과에 따라 정해지는 숫자
- **확률변수가 특정 값을 가질 확률 $P(X = x)$**: 그 숫자가 나올 가능성

**예시 1: 주사위 던지기**
- 표본공간: {1, 2, 3, 4, 5, 6} (결과 자체가 이미 숫자)
- 확률변수 $X$: 주사위 결과를 그대로 사용 (특별한 매핑 없음)
  - 주사위가 1이 나오면: $X = 1$
  - 주사위가 2가 나오면: $X = 2$
  - ...
- **확률**: $P(X = 1) = \frac{1}{6}$, $P(X = 2) = \frac{1}{6}$, ...

**예시 2: 동전 던지기** (매핑이 필요한 경우)
- 표본공간: {앞면, 뒷면} (숫자가 아님!)
- 확률변수 $X$: 앞면을 1로, 뒷면을 0으로 매핑
  - 앞면이 나오면: $X = 1$
  - 뒷면이 나오면: $X = 0$
- **확률**: $P(X = 1) = \frac{1}{2}$ (앞면), $P(X = 0) = \frac{1}{2}$ (뒷면)

**핵심 정리**:
- 확률변수 $X$는 실험 결과를 숫자로 변환하는 함수
- $X$의 값과 $P(X = x)$는 다릅니다!
  - 주사위 예: $X = 1$ (값)이고, $P(X = 1) = \frac{1}{6}$ (확률)

**머신러닝에서의 의미**:
- 데이터 포인트 $x$의 값을 관측하는 것 = 확률변수 $X$의 관측값
- 모델의 출력 $y$도 확률변수
- 확률변수의 분포 $P(X)$를 학습하는 것이 머신러닝

### 기댓값 (Expected Value)과 분산 (Variance)

**기댓값 $E[X]$란?** 확률변수 $X$의 평균값이에요. "확률을 고려한 가중평균"이라고 생각하시면 돼요.

**이산 확률변수**:
$$E[X] = \sum_{x} x \cdot P(X = x)$$

**연속 확률변수**:
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

여기서 $f(x)$는 확률 밀도 함수 (PDF)예요.

**예시**: 주사위를 던질 때
- $E[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + \cdots + 6 \cdot \frac{1}{6} = 3.5$

**분산 $\text{Var}(X)$란?** 확률변수가 평균에서 얼마나 퍼져있는지를 측정하는 거예요.

### 분산을 왜 제곱으로 측정하나요?

**문제**: 평균에서 떨어진 정도를 측정하고 싶어요.

**왜 절댓값이나 평균 편차를 쓰지 않나요?**

평균 편차를 계산해봅시다:
$$E[X - \mu] = E[X] - E[\mu] = \mu - \mu = 0$$

**결과**: 항상 0이에요! 양수와 음수가 서로 상쇄되어 버립니다.

**해결책**: 제곱을 사용하면 양수와 음수가 모두 양수가 되고, 상쇄되지 않아요:
$$\text{Var}(X) = E[(X - \mu)^2]$$

### 분산 공식의 유도

**두 가지 표현이 같다는 것**: $E[(X-\mu)^2] = E[X^2] - (E[X])^2$

**유도 과정**:

$$\text{Var}(X) = E[(X - \mu)^2]$$

전개하면:
$$= E[X^2 - 2X\mu + \mu^2]$$

기댓값의 선형성 사용 ($E[aX + b] = aE[X] + b$):
$$= E[X^2] - 2\mu E[X] + \mu^2$$

$E[X] = \mu$이므로:
$$= E[X^2] - 2\mu \cdot \mu + \mu^2$$
$$= E[X^2] - 2\mu^2 + \mu^2$$
$$= E[X^2] - \mu^2$$
$$= E[X^2] - (E[X])^2$$

**결론**: 
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

두 공식이 수학적으로 동일해요!

**예시: 주사위 분산**

$\mu = E[X] = 3.5$이므로:

방법 1: $E[(X-\mu)^2]$ 계산
$$\text{Var}(X) = \frac{1}{6}[(1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2]$$
$$= \frac{1}{6}[6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25] = \frac{17.5}{6} \approx 2.92$$

방법 2: $E[X^2] - (E[X])^2$ 계산
$$E[X^2] = \frac{1}{6}(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6}$$
$$\text{Var}(X) = \frac{91}{6} - (3.5)^2 = \frac{91}{6} - 12.25 = \frac{91 - 73.5}{6} = \frac{17.5}{6} \approx 2.92$$

**의미**: 분산이 크면 값들이 평균에서 많이 흩어져 있고, 분산이 작으면 평균 주변에 모여있어요.

**머신러닝 연결**: 
- 정규분포에서 분산 $\sigma^2$은 퍼진 정도를 나타냄
- 배치 정규화는 분산을 1로 맞춰서 안정화

### 독립성 (Independence)

**두 사건 $A$와 $B$가 독립이란?** 한 사건의 발생이 다른 사건의 확률에 영향을 주지 않는다는 거예요.

$$P(A \cap B) = P(A) \cdot P(B)$$

**조건부 독립성**: $P(A|B) = P(A)$ - B가 일어났어도 A의 확률이 변하지 않음

**MLE에서의 중요성**: 

데이터 포인트들이 독립이라고 가정하면:

$$P(x_1, x_2, ..., x_n | \theta) = \prod_{i=1}^n P(x_i | \theta)$$

**이 공식의 의미**:
- **왼쪽**: 모든 데이터 $(x_1, x_2, ..., x_n)$가 동시에 나올 확률
- **오른쪽**: 각 데이터 포인트가 나올 확률들을 모두 곱한 값
- $\prod$ (파이): 곱하기 기호 ($\sum$는 합, $\prod$는 곱)
- $\theta$ (세타): 모델의 파라미터

**왜 곱셈인가?**
- 독립 사건이 동시에 일어날 확률 = 각 확률의 곱
- 예: 동전 3번 던져 모두 앞면 = $\frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} = \frac{1}{8}$
- 마찬가지로 모든 데이터가 동시에 나올 확률 = 각 데이터 확률의 곱

**실제 예시**:
데이터 3개가 있다고 가정: $x_1 = 2.1, x_2 = 3.5, x_3 = 1.8$

모델 파라미터 $\theta$가 주어졌을 때:
- $P(x_1 | \theta) = 0.3$ (첫 번째 데이터가 나올 확률)
- $P(x_2 | \theta) = 0.5$ (두 번째 데이터가 나올 확률)
- $P(x_3 | \theta) = 0.4$ (세 번째 데이터가 나올 확률)

**독립 가정 하에서**: 세 데이터가 모두 동시에 나올 확률
$$P(x_1, x_2, x_3 | \theta) = P(x_1 | \theta) \times P(x_2 | \theta) \times P(x_3 | \theta) = 0.3 \times 0.5 \times 0.4 = 0.06$$

**이 가정의 효과**: 
- 우도 함수를 곱셈으로 계산할 수 있어요
- 각 데이터 포인트의 확률을 독립적으로 계산하고 곱하면 전체 우도를 구할 수 있어요

### 확률 밀도 함수 (PDF)의 의미

**PDF $f(x)$는 확률이 아닙니다!** 중요한 점은 PDF는 "밀도"라는 거예요.

**이산 vs 연속**:
- **이산**: $P(X = x)$ - 정확히 그 값이 나올 확률 (확률 질량 함수, PMF)
- **연속**: $P(a \leq X \leq b) = \int_a^b f(x) dx$ - 구간의 확률을 계산 (확률 밀도 함수, PDF)

**PDF의 성질**:
$$\int_{-\infty}^{\infty} f(x) dx = 1$$

**직관**: 
- PDF는 "단위당 확률" - 높이가 높을수록 그 근처 값이 나올 확률이 높음
- 하지만 $f(x)$ 자체는 확률이 아니라 밀도예요
- 확률을 구하려면 구간을 적분해야 해요

**예시**: 정규분포에서 $f(0) = 0.4$라고 해서 "0이 나올 확률이 0.4"가 아니라, "0 근처의 확률 밀도가 0.4"라는 의미예요.

### 조건부 확률 (Conditional Probability)

**조건부 확률이란?** 어떤 조건이 주어졌을 때의 확률이에요.

사건 B가 일어났을 때 A가 일어날 확률:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**직관적 이해**: "B라는 조건 하에서 A가 일어날 확률은?"

**비유**: 
- 비가 올 확률 $P(비)$는 일반적 확률
- 하늘이 구름으로 가득할 때 비가 올 확률 $P(비|구름)$은 조건부 확률
- 구름이 있으면 비 올 확률이 높아지죠!

**머신러닝 연결**: 
- 이미지 $x$가 주어졌을 때 클래스 $y$일 확률: $P(y|x)$
- 이것이 분류 모델이 학습하는 거예요

### 베이즈 정리 (Bayes' Theorem)

**베이즈 정리가 뭔가요?** 새로운 정보를 얻었을 때 확률을 업데이트하는 공식이에요.

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**각 항의 의미**:
- $P(A)$: **사전 확률 (Prior)** - 아무 정보도 없을 때의 확률
- $P(A|B)$: **사후 확률 (Posterior)** - 정보 B를 얻은 후의 확률
- $P(B|A)$: **우도 (Likelihood)** - A가 일어났을 때 B가 나올 확률

### 우도 vs 사후 확률: 방향의 차이

**중요한 차이점**: 우도와 사후 확률은 **방향이 반대**예요!

#### 사후 확률 $P(A|B)$: 결과 → 원인 추정
- **질문**: "데이터 $B$를 봤을 때, 가설 $A$가 맞을 확률은?"
- **방향**: 결과(데이터)를 보고 원인(가설)을 추정
- **예시**: "가운을 입은 사람을 봤어. 이 사람이 의사일 확률은?"
  $$P(\text{의사} | \text{가운 입음}) = ?$$

#### 우도 $P(B|A)$: 원인 → 결과 확률
- **질문**: "가설 $A$가 맞다고 가정하면, 데이터 $B$가 나올 확률은?"
- **방향**: 원인(가설)을 가정하고 결과(데이터)의 확률을 계산
- **예시**: "이 사람이 의사라고 가정하면, 가운을 입을 확률은?"
  $$P(\text{가운 입음} | \text{의사}) = ?$$

**핵심**: 
- **우도**는 가설을 가정하고 데이터 확률을 보는 것 (반대 방향)
- **사후 확률**은 데이터를 보고 가설 확률을 추정하는 것 (일반적인 추론)

**베이즈 정리의 역할**: 
- 우도(반대 방향)를 알고 있으면 사후 확률(원하는 방향)을 계산할 수 있어요!
- $P(\text{가운 입음} | \text{의사})$를 알면 → $P(\text{의사} | \text{가운 입음})$을 구할 수 있음

**직관적 이해**: 
처음에는 $P(A)$라는 믿음이 있었는데, 정보 $B$를 얻으니 확률이 $P(A|B)$로 업데이트되는 거예요.

**비유**: 
- 처음 생각: "이 사람이 의사일 확률은 10%" (사전 확률)
- 새로운 정보: "이 사람이 백가드를 입고 있음" (우도가 높음)
- 업데이트: "의사일 확률이 80%로 올라감" (사후 확률)

**머신러닝 연결**: 

#### 베이지안 최적화 (Bayesian Optimization)

**하이퍼파라미터 튜닝에서의 활용**: 학습률, 정규화 강도 등을 어떻게 선택할까요?

**전통적 방법 (Grid Search)**: 모든 조합을 시도 → 비효율적

**베이지안 최적화**: 
1. **사전 확률**: 초기에는 모든 파라미터 값이 동일한 확률 (무정보 사전)
2. **실험**: 몇 가지 파라미터 조합을 시도하고 성능 관측
3. **사후 확률 업데이트**: 베이즈 정리로 "어떤 파라미터가 좋은 성능을 낼 가능성이 높은가?" 업데이트
4. **다음 실험 선택**: 높은 사후 확률을 가진 영역에서 더 시도

**예시**: 
- 처음: "학습률 0.1 vs 0.01 중 어느 게 좋을까?" → 사전 확률 50:50
- 실험 결과: 0.1이 더 좋은 성능 → 사후 확률 업데이트 (0.1이 더 유리)
- 다음 실험: 0.1 근처의 값들 (0.09, 0.11 등) 집중 탐색

#### MLE vs 베이지안 학습

**MLE (Maximum Likelihood Estimation)**:
- 목표: 하나의 "최적" 파라미터 값 찾기 (점 추정)
- 예: $\hat{\theta} = 0.5$ (이게 최적값이다!)
- 장점: 계산이 빠르고 단순함
- 한계: 불확실성을 표현하지 못함

**베이지안 학습**:
- 목표: 파라미터의 확률 분포를 추정 (분포 추정)
- 예: $\theta \sim \mathcal{N}(0.5, 0.1^2)$ (평균 0.5, 불확실함 정도는 0.1)
- 장점: 불확실성을 확률로 표현 (신뢰 구간 등)
- 한계: 계산이 복잡함 (베이지안 신경망)

**실전**:
- 대부분의 딥러닝은 MLE 사용 (빠르고 효과적)
- 불확실성이 중요한 응용 (의료, 자율주행 등)에서는 베이지안 방법 고려

## 2.2 확률 분포 (Probability Distribution)

**확률 분포가 뭔가요?** 어떤 값들이 나올 수 있는지, 그리고 각 값이 나올 확률이 얼마인지를 나타내는 거예요.

**비유**: 주사위를 던질 때
- 가능한 값: 1, 2, 3, 4, 5, 6 (각각 확률 1/6)
- 이것이 "주사위의 확률 분포"예요

**머신러닝에서의 의미**: 
- 데이터가 어떤 패턴을 따르는지
- 모델이 예측하는 값의 불확실성
- 손실 함수는 이 분포를 가정하고 만들어짐

### 이산 확률 분포 (Discrete Probability Distribution)

**이산이란?** 셀 수 있는 값들 (1, 2, 3, ... 또는 "고양이", "개", "새")

**확률 질량 함수 (Probability Mass Function, PMF)**: $P(X = x)$
- 각 값에 대해 정확히 그 값이 나올 확률

#### 1) 이항 분포 (Binomial Distribution)

$n$번 시행에서 성공 횟수의 분포:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**예시**: 동전을 10번 던져 앞면이 나오는 횟수

#### 2) 다항 분포 (Multinomial Distribution)

**다항 분포란?** 여러 개의 범주 중 하나를 선택하는 확률 분포예요.

여러 범주 중 하나를 선택하는 분포:

$$P(X_1=n_1, \ldots, X_k=n_k) = \frac{n!}{n_1! \cdots n_k!} p_1^{n_1} \cdots p_k^{n_k}$$

**직관적 이해**: 
- 동전 던지기는 2개 중 하나 (앞/뒤) → 이항 분포
- 주사위 던지기는 6개 중 하나 (1~6) → 다항 분포
- 분류 문제는 클래스 중 하나 선택 → 다항 분포!

**비유**: 
- 선택지가 3개인 시험 문제 (고양이/개/새)
- 각각 선택될 확률: $p_1, p_2, p_3$ (합은 1)
- 이것이 다항 분포예요

**조합론적 의미**:
- $\frac{n!}{n_1! \cdots n_k!}$: $n$개 중 $n_1$개, $n_2$개, ..., $n_k$개를 선택하는 방법의 수
- **예시**: 10번 시행에서 클래스 0이 3번, 클래스 1이 5번, 클래스 2가 2번 나오는 경우의 수
  - $\frac{10!}{3! \cdot 5! \cdot 2!} = \frac{3628800}{6 \cdot 120 \cdot 2} = 2520$가지 방법
- 이항 분포에서 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$도 같은 원리!

**머신러닝 연결**: 
- **주의**: 엄밀히 말하면, 단일 데이터 포인트의 분류는 **카테고리 분포 (Categorical Distribution)**입니다
- 다항 분포는 $n$번 시행에서 각 카테고리가 나온 횟수의 분포이고, $n=1$일 때가 카테고리 분포입니다
- 분류 문제에서는 각 데이터 포인트가 독립적으로 카테고리 분포를 따르고, 여러 데이터를 모으면 다항 분포와 연결됩니다
- 소프트맥스 (Softmax) 출력이 카테고리 분포의 확률 파라미터
- CrossEntropy Loss가 이 가정에서 MLE로 유도됨

### 연속 확률 분포 (Continuous Probability Distribution)

**확률 밀도 함수 (Probability Density Function, PDF)**: $f(x)$, $P(a \leq X \leq b) = \int_a^b f(x) dx$

#### 1) 정규 분포 (Normal Distribution)

**정규 분포란?** 가장 많이 쓰이는 확률 분포예요. 종 모양(bell curve)으로 생겼어요.

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**파라미터**:
- $\mu$ (뮤): **평균 (Mean)** - 분포의 중심점
- $\sigma^2$ (시그마 제곱): **분산 (Variance)** - 퍼진 정도
- 표기: $X \sim \mathcal{N}(\mu, \sigma^2)$

**특징**:
- **종 모양 (bell curve)**: 가운데가 높고 양쪽으로 갈수록 낮아짐
- **평균 주변에 집중**: 대부분의 값이 평균 근처에 있어요
- **중심극한정리**: 많은 확률변수를 더하면 정규분포에 가까워짐
- **68-95-99.7 규칙**: 
  - 평균 ± 1σ 안에 68% 데이터
  - 평균 ± 2σ 안에 95% 데이터
  - 평균 ± 3σ 안에 99.7% 데이터

**비유**: 
- 키, 몸무게, 시험 점수 등 자연 현상들이 보통 정규분포를 따라요
- 평균 키가 170cm라면, 160cm나 180cm는 많지만, 150cm나 190cm는 드물죠

**머신러닝 연결**: 
- **가중치 초기화**: Xavier, He initialization은 정규분포 사용
- **회귀 문제의 오차**: 예측 오차가 정규분포를 따른다고 가정 → MSE Loss
- **배치 정규화 (Batch Normalization)**: 데이터를 정규분포로 만들기 (평균 0, 분산 1)

## 2.2.5 정보 이론 (Information Theory)과 CrossEntropy

**왜 CrossEntropy라는 이름일까요?** "Entropy(엔트로피)"가 뭔지 모르면 이름 자체가 이해가 안 가요. 이 섹션에서는 정보 이론의 핵심 개념들을 배워봅시다.

### 엔트로피 (Entropy) - 불확실성의 척도

**엔트로피 $H(X)$란?** 확률 분포의 불확실성을 측정하는 거예요. 불확실성이 높을수록 엔트로피가 커져요.

$$H(X) = -\sum_{x} p(x) \log p(x)$$

**직관적 이해**:
- 확률이 균등할수록 (모든 값이 똑같이 나올 가능성) 엔트로피가 높음
- 하나의 값에만 확률이 집중되어 있으면 (확실함) 엔트로피가 낮음

**비유**:
- 동전 던지기 (앞/뒤 각 50%): 엔트로피 높음 (불확실함)
- "내일 해가 뜬다" (100%): 엔트로피 0 (완전히 확실)

**머신러닝에서의 의미**:
- 모델의 예측이 불확실할수록 엔트로피가 높음
- 엔트로피가 낮다는 것은 모델이 특정 예측에 확신한다는 뜻

### KL Divergence - 두 분포의 차이

**KL Divergence $D_{KL}(P||Q)$란?** 두 확률 분포 $P$와 $Q$가 얼마나 다른지를 측정하는 거예요.

$$D_{KL}(P||Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

**의미**:
- $P$는 "실제 분포" (예: 실제 데이터 레이블 분포)
- $Q$는 "예측 분포" (예: 모델이 예측한 확률 분포)
- KL Divergence가 작을수록 두 분포가 비슷함

**특징**:
- $D_{KL}(P||Q) \geq 0$ (항상 0 이상)
- $D_{KL}(P||Q) = 0$ 이면 $P = Q$ (두 분포가 완전히 같음)
- 비대칭: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

**머신러닝에서의 목적**: 실제 데이터 분포 $P$와 모델 예측 분포 $Q$의 차이를 줄이는 것이 학습 목표!

### CrossEntropy - 실제 분포와 예측 분포의 교차 엔트로피

**CrossEntropy $H(P, Q)$란?** 실제 분포 $P$를 사용하여 예측 분포 $Q$의 정보량을 측정하는 거예요.

$$H(P, Q) = -\sum_{x} p(x) \log q(x)$$

**핵심 관계식**:
$$\text{CrossEntropy} = \text{Entropy}(P) + \text{KL Divergence}(P||Q)$$

$$H(P, Q) = H(P) + D_{KL}(P||Q)$$

**의미**:
- $H(P)$: 실제 분포의 불확실성 (고정값, 최적화할 수 없음)
- $D_{KL}(P||Q)$: 실제와 예측의 차이 (이걸 줄이고 싶음!)
- 따라서 CrossEntropy를 최소화 = KL Divergence 최소화 = 두 분포의 차이 최소화

**왜 CrossEntropy라는 이름인가?**
- 실제 분포 $P$의 관점에서 예측 분포 $Q$의 정보량을 측정하므로 "교차(Cross)" 엔트로피
- 단순히 $Q$의 엔트로피가 아니라, $P$의 가중치를 사용하여 $Q$를 평가

**머신러닝 연결**:
- 분류 문제에서 실제 레이블(one-hot) $P$와 모델 예측(softmax) $Q$의 CrossEntropy를 최소화
- CrossEntropy Loss = $-\sum_{k} y_k \log \hat{p}_k$ (여기서 $y_k$는 실제 분포, $\hat{p}_k$는 예측 분포)

### 정보 이론 관점에서 보는 손실 함수

**핵심 통찰**: 손실 함수는 단순한 수식이 아니라, **분포 간 거리**를 측정하는 정보론적 도구예요!

- **MSE**: 오차 분포와 정규분포의 차이
- **CrossEntropy**: 실제 레이블 분포와 예측 확률 분포의 차이 (KL Divergence)

이제 CrossEntropy Loss가 왜 분류 문제에 적합한지 이해가 되시나요? 실제 분포와 예측 분포를 가깝게 만드는 것이 목표이기 때문이에요!

## 2.3 최대우도 추정 (Maximum Likelihood Estimation, MLE)

**MLE가 뭔가요?** 데이터를 가장 잘 설명하는 모델의 파라미터를 찾는 방법이에요. "이 데이터가 나올 가능성이 가장 높은 파라미터는 뭐야?"를 찾는 거예요.

### 단계별 이해: MLE의 직관

#### Step 1: 문제 상황

우리는 **데이터를 관측**했습니다: $x_1, x_2, \ldots, x_n$

**질문**: 이 데이터를 생성한 확률 분포는 무엇일까?  
→ 즉, 분포의 **파라미터 $\theta$**는 무엇일까?

#### Step 2: 우도 (Likelihood)의 개념

**우도 (Likelihood)**: 주어진 파라미터 $\theta$에서 데이터가 나올 확률

$$L(\theta | \mathbf{x}) = P(x_1, x_2, \ldots, x_n | \theta)$$

**의미**: "이 파라미터가 맞다면, 이 데이터가 나올 확률은?"

#### Step 3: MLE의 아이디어

**직관**: "이 데이터를 가장 잘 설명하는 파라미터를 찾자!"

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\theta | \mathbf{x})$$

**비유**: 시험 점수 [85, 90, 88, 92]를 본다면, 평균이 약 89인 정규분포가 가장 그럴듯함

#### Step 4: 로그 우도 (Log-Likelihood)로 변환

**왜 로그를 쓰나?**
- 확률은 곱셈: $P(x_1, x_2) = P(x_1) \times P(x_2)$
- 곱셈은 작은 값이면 0에 가까워짐 (수치 불안정)
- 로그를 취하면 **덧셈**으로 변환:

$$\log L(\theta | \mathbf{x}) = \sum_{i=1}^n \log P(x_i | \theta)$$

**장점**: 
- 곱셈 → 덧셈 (계산 안정)
- 작은 값도 안정적으로 계산

#### Step 5: 손실 함수 (Loss Function)와의 연결

**핵심 통찰**: 손실 함수를 최소화 = 로그 우도 (Log-Likelihood)를 최대화

$$\text{Loss} = -\log L(\theta | \mathbf{x})$$

이것이 **음의 로그 우도 (Negative Log-Likelihood, NLL)**입니다!

**이유**: 
- 최대화 문제를 최소화 문제로 변환 (최적화 알고리즘과 호환)
- 경사하강법 (Gradient Descent)는 최소화 문제를 해결

### MSE Loss의 완전한 유도 과정

**목표**: "오차가 정규분포를 따른다"는 가정에서 MSE Loss가 어떻게 나오는지 단계별로 보여드립니다.

#### Step 1: 독립 가정 (Independence Assumption)

데이터 포인트들이 서로 독립이라고 가정:

$$P(y_1, y_2, ..., y_n | x_1, ..., x_n, \theta) = \prod_{i=1}^n P(y_i | x_i, \theta)$$

**이 공식이 표현하는 것**: 
- 왼쪽: **모든 데이터 $(y_1, y_2, ..., y_n)$가 동시에 나올 확률**
- 오른쪽: **각 데이터 포인트가 나올 확률들을 모두 곱한 값**

**직관적 이해**:

**비유: 동전을 3번 던질 때**
- 동전 던지기는 서로 독립 (첫 번째 결과가 두 번째 결과에 영향 없음)
- 앞면이 나올 확률 = $\frac{1}{2}$

전체 확률 = 각 시행 확률의 곱:
$$P(\text{앞, 앞, 앞}) = P(\text{앞}) \times P(\text{앞}) \times P(\text{앞}) = \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} = \frac{1}{8}$$

**머신러닝에서의 의미**:

데이터가 3개 있다고 가정: $(x_1, y_1), (x_2, y_2), (x_3, y_3)$

**독립 가정**: 데이터 포인트들이 서로 영향을 주지 않는다
- $y_1$의 값이 $y_2$의 확률에 영향을 주지 않음
- 각 데이터는 독립적으로 발생

**공식의 의미**:
$$P(y_1, y_2, y_3 | x_1, x_2, x_3, \theta) = P(y_1 | x_1, \theta) \times P(y_2 | x_2, \theta) \times P(y_3 | x_3, \theta)$$

**왜 곱셈인가?**
- 독립 사건이 동시에 일어날 확률 = 각 확률의 곱
- 동전을 3번 던져 모두 앞면이 나올 확률 = $\frac{1}{2} \times \frac{1}{2} \times \frac{1}{2}$
- 마찬가지로, 모든 데이터가 동시에 나올 확률 = 각 데이터 확률의 곱

**기호 설명**:
- $\prod$ (파이, Pi): 곱하기 기호 (Σ는 합이었다면, $\prod$는 곱)
  - $\prod_{i=1}^n a_i = a_1 \times a_2 \times \cdots \times a_n$
- $\theta$ (세타, Theta): 모델의 파라미터 (가중치, 편향 등)
- $P(y_i | x_i, \theta)$: $x_i$가 주어졌을 때 모델 파라미터 $\theta$에서 $y_i$가 나올 확률

#### Step 2: 정규분포 가정 (Normal Distribution Assumption)

회귀 모델: $y_i = f(x_i; \theta) + \epsilon_i$, 여기서 오차 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

각 데이터 포인트의 확률 밀도 함수 (PDF):

$$P(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right)$$

**의미**: 모델이 $f(x_i; \theta)$를 예측할 때, 실제 값 $y_i$가 나올 확률

#### Step 3: 우도 함수 (Likelihood Function) 계산

모든 데이터에 대한 우도:

$$L(\theta | \mathbf{y}, \mathbf{x}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right)$$

#### Step 4: 로그 우도 (Log-Likelihood)로 변환

로그를 취하면 곱셈이 덧셈으로 변환:

$$\log L(\theta) = \sum_{i=1}^n \log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right) \right)$$

$$= \sum_{i=1}^n \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f(x_i; \theta))^2}{2\sigma^2} \right]$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f(x_i; \theta))^2$$

#### Step 5: 상수 제거 및 손실 함수로 변환

**MLE 목표**: $\hat{\theta} = \arg\max_\theta \log L(\theta)$

$\sigma^2$가 고정되어 있다면, 첫 번째 항 $-\frac{n}{2}\log(2\pi\sigma^2)$는 상수:

$$\arg\max_\theta \log L(\theta) = \arg\max_\theta \left[ -\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f(x_i; \theta))^2 \right]$$

음수 곱하기와 상수 $\frac{1}{2\sigma^2}$ 제거하면:

$$= \arg\min_\theta \sum_{i=1}^n (y_i - f(x_i; \theta))^2$$

**결론**: MSE Loss가 등장합니다!

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

여기서 $\hat{y}_i = f(x_i; \theta)$는 모델의 예측값이에요.

### CrossEntropy Loss의 완전한 유도 과정

**목표**: "레이블이 다항분포를 따른다"는 가정에서 CrossEntropy Loss가 어떻게 나오는지 보여드립니다.

#### Step 1: 독립 가정 및 다항 분포 가정

각 데이터 포인트의 레이블이 독립적이고, 다항 분포를 따른다고 가정:

$$P(y_1, ..., y_n | x_1, ..., x_n, \theta) = \prod_{i=1}^n P(y_i | x_i, \theta)$$

**이 공식의 의미** (MSE 유도 Step 1 참조):
- 모든 데이터 레이블 $(y_1, ..., y_n)$이 동시에 나올 확률
- 각 데이터 포인트의 확률을 곱한 값
- 독립 사건이므로 곱셈 사용

**예시**: 
- 데이터 1: 이미지가 고양이일 확률 $P(y_1 | x_1, \theta) = 0.9$
- 데이터 2: 이미지가 개일 확률 $P(y_2 | x_2, \theta) = 0.8$
- 두 데이터가 동시에 나올 확률 = $0.9 \times 0.8 = 0.72$

분류 문제에서 실제 레이블은 one-hot 벡터: $y_i = [0, ..., 1, ..., 0]$ (정답 클래스만 1)

모델이 예측한 확률 분포: $\hat{p}_i = [\hat{p}_{i,1}, ..., \hat{p}_{i,K}]$ (소프트맥스 출력, 합은 1)

#### Step 2: 다항 분포의 확률

실제 레이블 $y_i$가 나올 확률:

$$P(y_i | x_i, \theta) = \prod_{k=1}^K (\hat{p}_{i,k})^{y_{i,k}}$$

**의미**: one-hot 벡터에서 $y_{i,k} = 1$인 정답 클래스 $k^*$만 남고 나머지는 $y_{i,k} = 0$입니다:
- 정답 클래스 $k^*$: $(\hat{p}_{i,k^*})^{1} = \hat{p}_{i,k^*}$
- 나머지 클래스 $k \neq k^*$: $(\hat{p}_{i,k})^{0} = 1$ (어떤 수의 0제곱은 1)

따라서:

$$P(y_i | x_i, \theta) = \hat{p}_{i,k^*}$$

여기서 $k^*$는 실제 정답 클래스예요.

#### Step 3: 로그 우도 계산

모든 데이터에 대한 로그 우도:

$$\log L(\theta) = \sum_{i=1}^n \log P(y_i | x_i, \theta) = \sum_{i=1}^n \log \hat{p}_{i,k_i^*}$$

여기서 $k_i^*$는 $i$번째 데이터의 정답 클래스예요.

one-hot 인코딩을 사용하면:

$$\log L(\theta) = \sum_{i=1}^n \sum_{k=1}^K y_{i,k} \log \hat{p}_{i,k}$$

**이유**: 
- 정답 클래스 $k^*$: $y_{i,k^*} = 1$이므로 $1 \cdot \log(\hat{p}_{i,k^*}) = \log(\hat{p}_{i,k^*})$
- 나머지 클래스 $k \neq k^*$: $y_{i,k} = 0$이므로 $0 \cdot \log(\hat{p}_{i,k}) = 0$
- 따라서 합은 정답 클래스의 로그 확률만 남습니다: $\sum_{k=1}^K y_{i,k} \log \hat{p}_{i,k} = \log \hat{p}_{i,k^*}$

**참고**: 수학적으로 $0 \cdot \log(0) = 0$으로 정의합니다 (극한값).

#### Step 4: 손실 함수로 변환

**MLE 목표**: $\hat{\theta} = \arg\max_\theta \log L(\theta)$

최대화 문제를 최소화 문제로 변환:

$$\arg\max_\theta \sum_{i=1}^n \sum_{k=1}^K y_{i,k} \log \hat{p}_{i,k} = \arg\min_\theta \left( -\sum_{i=1}^n \sum_{k=1}^K y_{i,k} \log \hat{p}_{i,k} \right)$$

**결론**: CrossEntropy Loss가 등장합니다!

$$\text{CrossEntropy} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{i,k} \log \hat{p}_{i,k}$$

**핵심 통찰**: 
- 로그 우도를 최대화 = CrossEntropy를 최소화
- 정답 클래스의 예측 확률이 높을수록 로그 우도가 커지고 손실이 감소
- 이것이 정보 이론의 CrossEntropy와 동일한 형태!

## 2.4 손실 함수 (Loss Function)와 확률 분포의 연결

**손실 함수가 뭔가요?** 모델이 예측한 값과 실제 값의 차이를 측정하는 함수예요. 이 값이 작을수록 모델이 잘 예측하는 거죠.

**왜 중요한가요?** 
- 손실 함수를 최소화하는 것이 모델 학습의 목표예요
- 경사하강법(Gradient Descent)으로 이 손실을 줄여나가요
- 손실 함수는 단순한 수식이 아니라 확률론적 의미를 가지고 있어요

### 핵심 원칙

**손실 함수는 확률 분포의 가정에서 자연스럽게 유도됩니다!**

손실 함수는 그냥 만든 게 아니라, "데이터가 이런 확률 분포를 따른다"고 가정하고 그 가정에서 자연스럽게 나온 거예요.

### 1) 평균 제곱 오차 손실 (Mean Squared Error, MSE Loss) ↔ 정규 분포 (Normal Distribution)

**MSE Loss가 뭔가요?** 회귀 문제에서 가장 많이 쓰는 손실 함수예요. 예측값과 실제값의 차이를 제곱해서 평균내는 거죠.

#### Step 1: 모델 가정 및 오차 개념 명확화

**통계 모델**: $y = f(x) + \epsilon$, 여기서 $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**오차 $\epsilon$의 의미**:
- **통계 관점**: 관측 오차 (Observation Error) - 측정 과정에서 발생하는 노이즈
- **머신러닝 관점**: $y - f(x)$는 예측 오차 (Prediction Error) 또는 잔차 (Residual)
- **가정**: 이 오차가 정규분포를 따른다고 가정 → $\epsilon \sim \mathcal{N}(0, \sigma^2)$

**왜 정규분포 가정이 합리적인가?**
- 중심극한정리: 많은 독립적인 작은 오차들의 합이 정규분포에 근사
- 실제로 많은 자연 현상의 오차가 정규분포를 따름
- 하지만 항상 맞는 것은 아님 (이상치, 비대칭 분포 등)

**의미**: 오차 $\epsilon$가 **정규분포 (Normal Distribution)**를 따른다고 가정

#### Step 2: 우도 함수 유도

정규분포의 확률 밀도 함수 (Probability Density Function):

$$P(y | x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x))^2}{2\sigma^2}\right)$$

**의미**: 모델이 $f(x)$를 예측할 때, 실제 값 $y$가 나올 확률

#### Step 3: 로그 우도 (Log-Likelihood) 계산

$$\log P(y | x, \theta) = -\frac{(y - f(x))^2}{2\sigma^2} + \text{const}$$

**관찰**: $(y - f(x))^2$ 항이 나타남!

#### Step 4: 손실 함수 (Loss Function)로 변환

**최대우도 추정 (MLE) = MSE 최소화**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**결론**: MSE Loss는 **오차가 정규분포를 따른다**는 가정에서 자연스럽게 나옵니다!

### 2) 교차 엔트로피 손실 (CrossEntropy Loss) ↔ 다항 분포 (Multinomial Distribution) + 정보 이론

**CrossEntropy Loss가 뭔가요?** 분류 문제에서 가장 많이 쓰는 손실 함수예요. 모델이 예측한 확률 분포와 실제 레이블의 차이를 측정해요.

#### Step 1: 정보 이론 관점에서의 이해

**실제 분포 $P$**: one-hot 벡터 $y = [0, \ldots, 1, \ldots, 0]$ (정답 클래스만 1)

**예측 분포 $Q$**: 소프트맥스 출력 $\hat{p} = [\hat{p}_1, \ldots, \hat{p}_K]$ (모든 클래스의 확률, 합은 1)

**CrossEntropy**:
$$\text{CrossEntropy} = H(P, Q) = -\sum_{k=1}^K y_k \log \hat{p}_k$$

**핵심 관계식** (2.2.5 정보 이론 섹션 참조):
$$\text{CrossEntropy} = \text{Entropy}(P) + \text{KL Divergence}(P||Q)$$

**의미**:
- 실제 분포 $P$의 엔트로피는 고정 (one-hot이므로 엔트로피가 낮음)
- 따라서 CrossEntropy를 최소화 = KL Divergence 최소화 = 두 분포의 차이 최소화
- **정보량 관점**: 실제 분포와 예측 분포를 가깝게 만드는 것이 목표

#### Step 2: MLE 관점에서의 유도

**분류 문제**: 클래스 $k$ 중 하나 선택

모델 출력 (소프트맥스, Softmax):

$$\hat{p}_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

실제 레이블 (one-hot): $y = [0, \ldots, 1, \ldots, 0]$

**다항 분포 가정**: 레이블이 다항분포를 따름

로그 우도 (Log-Likelihood) - 상세 유도는 2.3 MLE 섹션 참조:

$$\log P(y | x, \theta) = \sum_{k=1}^K y_k \log \hat{p}_k$$

**최대화 = CrossEntropy 최소화**:

$$\text{CrossEntropy} = -\sum_{k=1}^K y_k \log \hat{p}_k$$

#### Step 3: 두 관점의 통합

**정보 이론 + MLE**: 
- MLE 관점: 다항 분포의 로그 우도를 최대화
- 정보 이론 관점: 실제 분포와 예측 분포의 KL Divergence를 최소화
- **결론**: 두 관점이 동일한 손실 함수(CrossEntropy)로 수렴!

**왜 CrossEntropy라는 이름인가?**
- 실제 분포 $P$(one-hot)의 관점에서 예측 분포 $Q$(softmax)의 엔트로피를 측정
- 단순히 $Q$의 엔트로피가 아니라, $P$의 가중치를 사용하여 $Q$를 평가
- 따라서 "교차(Cross)" 엔트로피

### 핵심 통찰

**손실 함수는 확률 분포의 가정에서 자연스럽게 유도됩니다!**

**이게 왜 중요한가요?** 
손실 함수는 그냥 만든 게 아니라, "데이터가 이 확률 분포를 따른다"고 가정하고 그 가정에서 나온 거예요.

- **MSE**: 오차가 정규분포 (Normal Distribution)를 따른다고 가정 → 제곱 오차가 자연스럽게 나타남
- **CrossEntropy**: 레이블이 다항분포 (Multinomial Distribution)를 따른다고 가정 → 로그 확률이 자연스럽게 나타남

**따라서**:
- 회귀 문제에 MSE를 쓰는 건, 오차가 정규분포라는 합리적 가정 때문이에요
- 분류 문제에 CrossEntropy를 쓰는 건, 레이블이 다항분포라는 합리적 가정 때문이에요

### 배치 정규화 (Batch Normalization)와 정규 분포

**배치 정규화가 뭔가요?** 신경망 층의 입력을 정규분포로 만들어주는 기법이에요.

#### 문제 상황: 내부 공변량 이동 (Internal Covariate Shift)

**내부 공변량 이동이란?** 학습 중에 각 층으로 들어가는 입력 데이터의 분포가 계속 변하는 현상이에요.

**원인**:
1. 이전 층의 가중치가 업데이트되면서 출력 분포가 변함
2. 다음 층은 계속 변화하는 분포의 입력을 받게 됨
3. 각 층이 자신의 입력 분포에 적응하려고 하면서 학습이 불안정해짐

**문제점**:
- 학습률을 작게 설정해야 안정적 학습 (느림)
- 초기화에 매우 민감
- 활성화 함수의 포화(saturation) 문제 (예: sigmoid, tanh)

#### 해결책: 배치 정규화 (Batch Normalization)

**표준화 (Standardization)** 과정:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

여기서:
- $\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$: 배치의 평균 (배치 크기 $m$)
- $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$: 배치의 분산
- $\epsilon$: 수치 안정성을 위한 작은 값 (예: $10^{-5}$)

**추가 파라미터** (선택적):
$$\text{BN}(x) = \gamma \hat{x} + \beta$$

- $\gamma$: 스케일 파라미터 (학습 가능)
- $\beta$: 시프트 파라미터 (학습 가능)

#### 통계적 의미

**표준화의 목적**: 
- 평균을 0으로, 분산을 1로 만들기 → **표준 정규분포 $\mathcal{N}(0, 1)$에 근사**

**왜 평균 0, 분산 1인가?**
- 정규분포 가정: 많은 자연 현상이 정규분포를 따름
- 표준화된 입력은 대부분의 활성화 함수(sigmoid, tanh, ReLU)의 좋은 동작 범위에 있음
- 각 층이 비슷한 분포의 입력을 받아서 학습이 안정적

**학습 중 통계량 추정**:
- **학습 시**: 배치의 평균과 분산을 사용 (온라인 추정)
- **추론 시**: 학습 중 계산한 이동 평균(Moving Average)을 사용
  - $\mu_{\text{running}} = \alpha \mu_{\text{running}} + (1-\alpha) \mu_B$
  - $\sigma^2_{\text{running}} = \alpha \sigma^2_{\text{running}} + (1-\alpha) \sigma_B^2$
- 이렇게 하면 배치 크기가 1이어도 안정적으로 추론 가능

#### 효과

**학습 안정성**:
- 입력 분포를 안정적으로 유지
- 학습률을 크게 설정 가능 (학습 속도 향상)
- 초기화에 덜 민감

**정규 분포와의 연결**:
배치 정규화는 데이터를 표준 정규분포 $\mathcal{N}(0, 1)$에 가깝게 만들어요. 이렇게 하면:
- 모든 층이 비슷한 분포의 입력을 받음
- 학습이 더 안정적이고 빠름
- 그래디언트 소실/폭발 문제 완화

---
# 3. PyTorch 구현 (How)
---

## 3.1 확률 분포 시각화

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 정규 분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1) 다양한 평균
x = np.linspace(-10, 10, 200)
for mu in [-2, 0, 2]:
    y = stats.norm.pdf(x, mu, 1)
    axes[0].plot(x, y, label=f'μ={mu}, σ²=1', linewidth=2)
axes[0].set_title('평균이 다른 정규분포')
axes[0].set_xlabel('x')
axes[0].set_ylabel('확률 밀도')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2) 다양한 분산
for sigma in [0.5, 1, 2]:
    y = stats.norm.pdf(x, 0, sigma)
    axes[1].plot(x, y, label=f'μ=0, σ²={sigma**2}', linewidth=2)
axes[1].set_title('분산이 다른 정규분포')
axes[1].set_xlabel('x')
axes[1].set_ylabel('확률 밀도')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3) 표준 정규분포와 68-95-99.7 규칙
x = np.linspace(-4, 4, 200)
y = stats.norm.pdf(x, 0, 1)
axes[2].plot(x, y, 'b-', linewidth=2, label='표준정규분포')
axes[2].fill_between(x, 0, y, where=(x >= -1) & (x <= 1), alpha=0.3, label='68% (±1σ)')
axes[2].fill_between(x, 0, y, where=(x >= -2) & (x <= 2), alpha=0.2, label='95% (±2σ)')
axes[2].set_title('68-95-99.7 규칙')
axes[2].set_xlabel('x')
axes[2].set_ylabel('확률 밀도')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("정규분포의 특징:")
print("- 평균(μ): 분포의 중심")
print("- 분산(σ²): 퍼진 정도")
print("- 68%의 데이터가 ±1σ 안에")
print("- 95%의 데이터가 ±2σ 안에")
```

## 3.2 MSE와 정규분포의 연결

```python
# MSE Loss와 정규분포
import torch
import torch.nn as nn

# 간단한 회귀 문제
torch.manual_seed(42)
x = torch.randn(100, 1)
y_true = 2 * x + 1 + 0.5 * torch.randn(100, 1)  # y = 2x + 1 + noise

# 모델 예측
model = nn.Linear(1, 1)
y_pred = model(x)

# MSE Loss 계산
mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)

print("MSE Loss와 정규분포의 연결:")
print(f"MSE Loss: {loss.item():.4f}")
print("\n해석:")
print("- MSE를 최소화 = 오차의 제곱합을 최소화")
print("- 오차가 정규분포 N(0, σ²)를 따른다고 가정")
print("- MLE 관점: 데이터의 로그 우도를 최대화")

# 오차 분포 시각화
with torch.no_grad():
    errors = (y_pred - y_true).numpy().flatten()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(x.numpy(), y_true.numpy(), alpha=0.5, label='실제 데이터')
plt.scatter(x.numpy(), y_pred.detach().numpy(), alpha=0.5, label='모델 예측')
plt.xlabel('x')
plt.ylabel('y')
plt.title('회귀 문제')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(errors, bins=20, density=True, alpha=0.7, label='실제 오차')
x_range = np.linspace(errors.min(), errors.max(), 100)
plt.plot(x_range, stats.norm.pdf(x_range, errors.mean(), errors.std()), 
         'r-', linewidth=2, label='정규분포 근사')
plt.xlabel('오차')
plt.ylabel('밀도')
plt.title('오차 분포 (정규분포에 근사)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 3.3 CrossEntropy와 다항분포의 연결

```python
# CrossEntropy Loss와 다항분포 - 단계별 이해
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("Step 1: 모델 출력 (로짓, Logit)")
print("="*60)

# 3-클래스 분류 문제
logits = torch.tensor([[2.0, 1.0, 0.1]])  # 모델의 원시 출력
true_label = torch.tensor([0])  # 실제 레이블 (클래스 0)

print(f"로짓 (Logit, 원시 출력): {logits.tolist()}")
print("→ 아직 확률이 아님! (합이 1이 아님)")

print("\n" + "="*60)
print("Step 2: 소프트맥스 (Softmax)로 확률 분포로 변환")
print("="*60)

# Softmax: 확률 분포로 변환
probs = F.softmax(logits, dim=1)
print("소프트맥스 (Softmax) 출력 (확률 분포):")
print(f"  P(클래스 0) = {probs[0, 0].item():.4f}")
print(f"  P(클래스 1) = {probs[0, 1].item():.4f}")
print(f"  P(클래스 2) = {probs[0, 2].item():.4f}")
print(f"  합계: {probs.sum().item():.4f} (확률의 합 = 1)")

print("\n" + "="*60)
print("Step 3: 다항분포 (Multinomial Distribution) 가정")
print("="*60)

print("가정: 실제 레이블이 다항분포를 따름")
print("→ 정답 클래스의 확률이 높을수록 좋은 모델")
print(f"→ 정답은 클래스 {true_label.item()}이므로 P(클래스 {true_label.item()})가 높아야 함")

print("\n" + "="*60)
print("Step 4: 우도 (Likelihood)와 로그 우도 (Log-Likelihood)")
print("="*60)

# 우도: 정답 클래스의 확률
likelihood = probs[0, true_label].item()
log_likelihood = torch.log(probs[0, true_label]).item()

print(f"우도 (Likelihood): P(클래스 {true_label.item()} | 모델) = {likelihood:.4f}")
print(f"로그 우도 (Log-Likelihood): log(P) = {log_likelihood:.4f}")

print("\n" + "="*60)
print("Step 5: CrossEntropy Loss = 음의 로그 우도 (Negative Log-Likelihood, NLL)")
print("="*60)

# CrossEntropy Loss
ce_loss = F.cross_entropy(logits, true_label)
manual_loss = -torch.log(probs[0, true_label])

print(f"CrossEntropy Loss: {ce_loss.item():.4f}")
print(f"수동 계산 (-log P): {manual_loss.item():.4f}")
print(f"→ 일치함: {torch.allclose(ce_loss, manual_loss)}")

print("\n해석:")
print("- 손실을 최소화 = 로그 우도 (Log-Likelihood)를 최대화")
print("- 정답 클래스의 확률이 높을수록 손실 감소")
print("- 이것이 최대우도 추정 (MLE)의 원리!")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1) 확률 분포
axes[0].bar(['클래스 0', '클래스 1', '클래스 2'], probs[0].detach().numpy(), 
            color=['red', 'gray', 'gray'], alpha=0.7)
axes[0].axhline(y=1/3, color='k', linestyle='--', alpha=0.3, label='균등 분포 (0.333)')
axes[0].set_ylabel('확률')
axes[0].set_title('소프트맥스 (Softmax) 출력 (확률 분포)')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].legend()

# 2) 확률에 따른 손실 변화
probs_range = np.linspace(0.01, 1, 100)
losses = -np.log(probs_range)
axes[1].plot(probs_range, losses, linewidth=2, color='blue')
axes[1].axvline(x=probs[0, 0].item(), color='r', linestyle='--', 
            label=f'현재={probs[0, 0].item():.3f}')
axes[1].scatter([probs[0, 0].item()], [-np.log(probs[0, 0].item())], 
                color='red', s=100, zorder=5)
axes[1].set_xlabel('정답 클래스의 확률')
axes[1].set_ylabel('손실 (-log p)')
axes[1].set_title('확률에 따른 CrossEntropy Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3) 로그 우도 (Log-Likelihood) 비교
x_pos = np.arange(3)
log_probs = torch.log(probs[0]).detach().numpy()
axes[2].bar(x_pos, log_probs, alpha=0.7, color=['red', 'gray', 'gray'])
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(['클래스 0', '클래스 1', '클래스 2'])
axes[2].set_ylabel('로그 확률 (log P)')
axes[2].set_title('로그 우도 (Log-Likelihood, 높을수록 좋음)')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n관찰:")
print("- 정답 확률이 1에 가까울수록 손실이 0에 가까움")
print("- 정답 확률이 0에 가까울수록 손실이 무한대로 증가")
print("- 로그 우도 (Log-Likelihood)를 최대화하면 정답 확률이 최대화됨")
```

## 3.4 실제 학습 루프 예제

**목적**: 손실 함수가 실제로 어떻게 최적화에 사용되는지 보여드립니다.

이제까지는 손실을 계산만 했는데, 실제 학습에서는 어떻게 쓰일까요?

```python
# 실제 학습 루프: MSE Loss를 사용한 선형 회귀
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성: y = 2x + 1 + noise
torch.manual_seed(42)
n_samples = 100
x = torch.randn(n_samples, 1)
y_true = 2 * x + 1 + 0.3 * torch.randn(n_samples, 1)

# 모델 정의: 선형 회귀
model = nn.Linear(1, 1)
criterion = nn.MSELoss()  # MSE Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 확률적 경사하강법

# 학습 기록
losses = []
epochs = 100

print("="*60)
print("학습 시작")
print("="*60)

# 학습 루프
for epoch in range(epochs):
    # Forward pass: 예측값 계산
    y_pred = model(x)
    
    # Loss 계산: MSE Loss
    loss = criterion(y_pred, y_true)
    
    # Backward pass: 그래디언트 계산
    optimizer.zero_grad()  # 이전 그래디언트 초기화
    loss.backward()         # 역전파: 그래디언트 계산
    
    # 파라미터 업데이트: 경사하강법
    optimizer.step()        # 가중치 업데이트: θ ← θ - lr * ∇θ
    
    # 기록
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 학습된 모델 파라미터 확인
with torch.no_grad():
    weight = model.weight.item()
    bias = model.bias.item()
    print(f"\n학습된 모델: y = {weight:.3f}x + {bias:.3f}")
    print(f"실제 모델: y = 2.000x + 1.000")
    print(f"→ 거의 정확하게 학습됨!")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1) 학습 곡선
axes[0].plot(losses, linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('학습 곡선: 손실이 점점 감소함')
axes[0].grid(True, alpha=0.3)

# 2) 데이터와 학습된 모델
with torch.no_grad():
    y_pred_final = model(x)
    
axes[1].scatter(x.numpy(), y_true.numpy(), alpha=0.5, label='실제 데이터')
axes[1].plot(x.numpy(), y_pred_final.numpy(), 'r-', linewidth=2, label=f'학습된 모델: y={weight:.3f}x+{bias:.3f}')
axes[1].plot(x.numpy(), (2*x + 1).numpy(), 'g--', linewidth=2, label='실제 모델: y=2x+1')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('데이터와 학습된 모델')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n핵심 과정:")
print("1. Forward: 모델 예측 → loss 계산")
print("2. Backward: loss.backward() → 그래디언트 계산")
print("3. Update: optimizer.step() → 파라미터 업데이트")
print("4. 반복: 위 과정을 여러 epoch 동안 반복")
print("\n→ 손실 함수를 최소화하는 파라미터를 찾는 것이 목표!")
```

### 배치 처리 패턴

**왜 배치를 사용하나요?** 전체 데이터를 한 번에 처리하면 메모리 부족이 발생할 수 있어요. 작은 배치로 나눠서 처리합니다.

```python
# 배치 처리 예제
import torch
from torch.utils.data import Dataset, DataLoader

# 커스텀 데이터셋
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 데이터 생성
torch.manual_seed(42)
n_samples = 1000
x = torch.randn(n_samples, 1)
y_true = 2 * x + 1 + 0.3 * torch.randn(n_samples, 1)

# Dataset과 DataLoader 생성
dataset = SimpleDataset(x, y_true)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("="*60)
print(f"전체 데이터: {len(dataset)}개")
print(f"배치 크기: {batch_size}")
print(f"배치 수: {len(dataloader)}개")
print("="*60)

# 배치 단위 학습
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    
    # 각 배치마다 학습
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        # Forward
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    avg_loss = epoch_loss / n_batches
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], 평균 Loss: {avg_loss:.4f}")

print("\n배치 처리의 장점:")
print("- 메모리 효율적: 작은 배치만 메모리에 로드")
print("- 빠른 학습: 배치 단위로 병렬 처리 가능")
print("- 일반화: 각 epoch마다 데이터 순서가 섞임 (shuffle=True)")
```

## 3.5 실제 데이터 예제: 가정 검증과 한계

**목적**: "오차가 정규분포를 따른다"는 가정이 실제로 맞는지 확인하고, 가정이 틀렸을 때 어떤 일이 일어나는지 봅시다.

```python
# 오차 분포 검증: 정규분포 가정이 맞는지 확인
import torch
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성: y = 2x + 1 + 정규분포 노이즈
torch.manual_seed(42)
n_samples = 500
x = torch.randn(n_samples, 1)
y_true = 2 * x + 1 + 0.3 * torch.randn(n_samples, 1)  # 정규분포 노이즈

# 모델 학습
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 빠른 학습 (간단한 예제)
for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습 후 오차 계산
with torch.no_grad():
    y_pred = model(x)
    errors = (y_true - y_pred).numpy().flatten()

# 정규분포 검증: Shapiro-Wilk 테스트 (작은 샘플에 적합)
# 정규분포를 따르면 p-value가 큼 (> 0.05)
if len(errors) <= 50:
    stat, p_value = stats.shapiro(errors)
    test_name = "Shapiro-Wilk"
else:
    # 큰 샘플: Kolmogorov-Smirnov 테스트
    stat, p_value = stats.kstest(errors, 'norm', args=(errors.mean(), errors.std()))
    test_name = "Kolmogorov-Smirnov"

print("="*60)
print("오차 분포 검증")
print("="*60)
print(f"오차의 평균: {errors.mean():.4f} (0에 가까워야 함)")
print(f"오차의 표준편차: {errors.std():.4f}")
print(f"\n정규성 검정 ({test_name}):")
print(f"  p-value: {p_value:.4f}")
if p_value > 0.05:
    print("  → 정규분포를 따른다는 가정을 기각하지 않음 (가정이 합리적)")
else:
    print("  → 정규분포를 따른다는 가정을 기각 (가정이 맞지 않을 수 있음)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1) 오차 히스토그램 + 정규분포 비교
axes[0].hist(errors, bins=30, density=True, alpha=0.7, label='실제 오차 분포', color='blue')
x_range = np.linspace(errors.min(), errors.max(), 100)
normal_fit = stats.norm.pdf(x_range, errors.mean(), errors.std())
axes[0].plot(x_range, normal_fit, 'r-', linewidth=2, label='정규분포 근사')
axes[0].set_xlabel('오차 (y_true - y_pred)')
axes[0].set_ylabel('밀도')
axes[0].set_title('오차 분포 vs 정규분포')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2) Q-Q 플롯 (정규성 검증)
from scipy.stats import probplot
probplot(errors, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (정규분포 검증)')
axes[1].grid(True, alpha=0.3)
# 점들이 직선에 가깝게 분포하면 정규분포를 따름

plt.tight_layout()
plt.show()

print("\n관찰:")
print("- 오차가 정규분포에 가깝게 분포 → MSE Loss 가정이 합리적")
print("- Q-Q Plot에서 점들이 직선에 가까움 → 정규분포 가정 타당")
```

### 가정이 틀렸을 때: 이상치와 비대칭 분포

**실제 데이터는 항상 정규분포를 따르지 않습니다!** 이상치(outlier)나 비대칭 분포가 있을 때 어떻게 될까요?

```python
# 이상치가 있는 경우
torch.manual_seed(42)
n_samples = 200
x = torch.randn(n_samples, 1)

# 정상 데이터: y = 2x + 1 + 작은 노이즈
y_normal = 2 * x + 1 + 0.2 * torch.randn(n_samples, 1)

# 이상치 추가 (10%의 데이터)
n_outliers = n_samples // 10
outlier_indices = torch.randperm(n_samples)[:n_outliers]
y_with_outliers = y_normal.clone()
y_with_outliers[outlier_indices] += 5 * torch.randn(n_outliers, 1)  # 큰 오차 추가

# 정상 데이터와 이상치 데이터로 각각 학습
models = {
    '정상 데이터': nn.Linear(1, 1),
    '이상치 있음': nn.Linear(1, 1)
}

for name, model in models.items():
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    y_data = y_normal if name == '정상 데이터' else y_with_outliers
    
    for epoch in range(100):
        y_pred = model(x)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 결과 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, model) in enumerate(models.items()):
    y_data = y_normal if name == '정상 데이터' else y_with_outliers
    
    with torch.no_grad():
        y_pred = model(x)
        errors = (y_data - y_pred).numpy().flatten()
        
        # 통계량
        mean_err = errors.mean()
        std_err = errors.std()
        
        # 시각화
        axes[idx].scatter(x.numpy(), y_data.numpy(), alpha=0.5, label='데이터', s=20)
        axes[idx].plot(x.numpy(), y_pred.numpy(), 'r-', linewidth=2, label='학습된 모델')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].set_title(f'{name}\n오차 평균: {mean_err:.3f}, 표준편차: {std_err:.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 오차 분포 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, model) in enumerate(models.items()):
    y_data = y_normal if name == '정상 데이터' else y_with_outliers
    
    with torch.no_grad():
        y_pred = model(x)
        errors = (y_data - y_pred).numpy().flatten()
    
    axes[idx].hist(errors, bins=30, density=True, alpha=0.7, label='실제 오차')
    x_range = np.linspace(errors.min(), errors.max(), 100)
    normal_fit = stats.norm.pdf(x_range, errors.mean(), errors.std())
    axes[idx].plot(x_range, normal_fit, 'r-', linewidth=2, label='정규분포 근사')
    axes[idx].set_xlabel('오차')
    axes[idx].set_ylabel('밀도')
    axes[idx].set_title(f'{name}의 오차 분포')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*60)
print("관찰: 이상치의 영향")
print("="*60)
print("1. 이상치가 있으면:")
print("   - 오차 분포가 비대칭이 됨 (정규분포 가정 위배)")
print("   - 모델이 이상치에 끌려서 성능이 떨어짐")
print("   - MSE Loss는 이상치에 매우 민감 (제곱 때문에)")

print("\n2. 해결 방법:")
print("   - Robust Loss 사용 (예: Huber Loss, L1 Loss)")
print("   - 이상치 제거 또는 처리")
print("   - 다른 확률 분포 가정 (예: t-분포)")

print("\n3. 교훈:")
print("   - 가정이 틀리면 모델 성능이 떨어질 수 있음")
print("   - 실제 데이터를 분석하여 가정을 검증해야 함")
print("   - 가정이 맞지 않으면 다른 손실 함수 고려")
```

---
# 5. 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 확률의 기본과 수학적 엄밀성
- **확률의 공리**: 콜모고로프의 3가지 공리 (비음성, 정규화, 가산성)
- **확률변수**: 실험 결과를 숫자로 표현 (예: 주사위 결과 $X = 1, 2, ..., 6$, 동전 앞면 $X = 1$)
- **확률변수의 값 vs 확률**: $X = 1$ (값)과 $P(X = 1) = \frac{1}{6}$ (확률)은 다름
- **기댓값과 분산**: $E[X]$, $\text{Var}(X)$의 정의와 의미
- **독립성**: MLE의 독립 가정의 근거
- **PDF vs PMF**: 확률 밀도 함수는 확률이 아닌 밀도

### 2. 정보 이론 (Information Theory)
- **엔트로피 (Entropy)**: $H(X) = -\sum p(x) \log p(x)$ - 불확실성의 척도
- **KL Divergence**: $D_{KL}(P||Q)$ - 두 분포의 차이 측정
- **CrossEntropy**: $H(P, Q) = H(P) + D_{KL}(P||Q)$ - 실제 분포와 예측 분포의 차이
- **핵심**: CrossEntropy Loss는 정보량 관점에서 분포 간 거리를 측정

### 3. 확률 분포 (Probability Distribution)
- **이산 분포**: 이항 분포, 다항 분포 (조합론적 의미 포함)
- **연속 분포**: 정규 분포 $\mathcal{N}(\mu, \sigma^2)$
- **정규 분포의 특징**: 종 모양, 68-95-99.7 규칙

### 4. 최대우도 추정 (Maximum Likelihood Estimation, MLE)
- **아이디어**: 데이터를 가장 잘 설명하는 파라미터 찾기
- **완전한 유도**: MSE와 CrossEntropy의 단계별 유도 과정
- **로그 우도**: $\log L(\theta | \mathbf{x}) = \sum \log P(x_i | \theta)$
- **손실 함수**: $\text{Loss} = -\log L$ (음의 로그 우도, NLL)

### 5. 손실 함수와 확률 분포의 연결
- **MSE ↔ 정규 분포**: 오차가 정규분포를 따른다는 가정에서 유도
- **CrossEntropy ↔ 다항 분포 + 정보 이론**: 레이블이 다항분포를 따르며, 정보론적으로는 분포 간 거리
- **핵심**: 손실 함수는 확률 가정에서 자연스럽게 유도됨

### 6. 베이즈 정리와 실전 활용
- **베이지안 최적화**: 하이퍼파라미터 튜닝에서 사전/사후 확률 업데이트
- **MLE vs 베이지안**: 점 추정 vs 분포 추정, 불확실성 표현

### 7. 배치 정규화 (Batch Normalization)
- **내부 공변량 이동 (ICS)**: 학습 중 입력 분포 변화 문제
- **표준화**: 평균 0, 분산 1로 만들어 표준 정규분포에 근사
- **통계량 추정**: 학습 시 배치 통계, 추론 시 이동 평균

### 8. 실전 구현
- **실제 학습 루프**: `loss.backward()`, `optimizer.step()` 포함
- **배치 처리**: DataLoader를 통한 효율적 학습
- **가정 검증**: 오차 분포의 정규성 검정
- **한계 이해**: 이상치와 비대칭 분포에서의 문제점

## 다음 단원 미리보기

**다음 단원 (01)**: 텐서 (Tensor) 기초

- 수학 기초를 다졌으니, 이제 PyTorch의 핵심 데이터 구조인 텐서를 배워봅시다
- 텐서는 벡터와 행렬의 일반화입니다
- 모든 연산의 기본 단위입니다

수학적 기초가 탄탄해졌으니, 이제 본격적으로 PyTorch를 다룰 준비가 되었습니다!
