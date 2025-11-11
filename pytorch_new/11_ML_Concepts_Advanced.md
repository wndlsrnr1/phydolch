# 11. 머신러닝 개념 심화

## 목표·범위·평가 기준

**목표**: 각 개념의 수학적 원리와 PyTorch 구현을 연결하여 "왜"를 이해한다.

**출력물**: 개념별 비교표 + 실험 결과(손실 함수 비교, 정규화 효과) + 각 개념의 "왜" 설명 가능

**합격선**: 각 개념의 수학적 정의, PyTorch 구현, 사용 시기를 설명할 수 있음

---

## Step 0: 환경 설정 및 기호 정리

### 이번 단계 목표
- 재현 가능한 실험 환경 구축
- 문서 전체에서 사용할 기호와 변수 정의

### 입력
- 없음 (환경 설정)

### 출력
- 고정된 환경 설정
- 기호 정의표

### 평가 기준
- 동일 환경에서 동일 결과 재현 가능

### 환경 설정

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LinearRegression, HuberRegressor
import seaborn as sns
import os

# ============================================
# 환경 설정 (재현성 보장)
# ============================================

# PyTorch 버전 확인
print(f"PyTorch 버전: {torch.__version__}")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 난수 시드 고정
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# 결과 저장 디렉토리
os.makedirs('results', exist_ok=True)
```

### 기호 정리 (전체 문서 공통)

**역할 표 (손실 함수)**:

| 역할 | 수학 기호 | PyTorch 변수 | 설명 |
|------|-----------|--------------|------|
| 실제 값 | $y$, $y_i$ | `y_true`, `target` | 정답 라벨 또는 값 |
| 예측 값 | $\hat{y}$, $\hat{y}_i$ | `y_pred`, `output` | 모델 예측 |
| 오차 | $e_i = y_i - \hat{y}_i$ | `error` | 실제 - 예측 |
| 손실 함수 | $L(y, \hat{y})$ | `loss`, `criterion` | 전체 손실 |
| 샘플 수 | $n$ | `batch_size`, `N` | 배치 크기 |
| 클래스 수 | $C$ | `num_classes` | 분류 클래스 개수 |

**역할 표 (정규화)**:

| 역할 | 수학 기호 | PyTorch 변수 | 설명 |
|------|-----------|--------------|------|
| 가중치 | $w_i$, $\mathbf{w}$ | `weight`, `params` | 모델 파라미터 |
| 정규화 계수 | $\lambda$, $\lambda_1$, $\lambda_2$ | `weight_decay`, `l1_lambda` | 정규화 강도 |
| 데이터 손실 | $L_{\text{data}}$ | `data_loss` | 원본 손실 |
| 정규화 항 | $R(\mathbf{w})$ | `reg_loss` | 정규화 손실 |
| 총 손실 | $L_{\text{total}}$ | `total_loss` | 데이터 손실 + 정규화 |

**역할 표 (평가 지표)**:

| 역할 | 수학 기호 | PyTorch 변수 | 설명 |
|------|-----------|--------------|------|
| True Positive | $TP$ | `tp` | 양성으로 올바르게 예측 |
| True Negative | $TN$ | `tn` | 음성으로 올바르게 예측 |
| False Positive | $FP$ | `fp` | 양성으로 잘못 예측 |
| False Negative | $FN$ | `fn` | 음성으로 잘못 예측 |
| 정확도 | $\text{Accuracy}$ | `accuracy` | 전체 정확도 |
| 정밀도 | $\text{Precision}$ | `precision` | 양성 예측 중 실제 양성 비율 |
| 재현율 | $\text{Recall}$ | `recall` | 실제 양성 중 올바르게 예측 비율 |

---

## 1. 손실 함수의 심화

### Step 1: 손실 함수란 무엇인가?

### 이번 단계 목표
- 손실 함수의 역할과 필요성 이해
- 회귀 vs 분류 문제별 손실 함수 선택 기준 파악

### 입력
- 없음 (개념 설명)

### 출력
- 손실 함수의 역할 이해
- 문제 유형별 적합한 손실 함수 선택 기준

### 평가 기준
- "왜 손실 함수가 필요한가?"를 설명할 수 있음

### 왜 손실 함수가 필요한가?

**문제 상황**:
- 모델의 예측이 얼마나 틀렸는지 측정해야 함
- 틀린 정도를 수치로 표현하여 모델을 개선할 수 있어야 함
- 최적화 알고리즘이 방향을 알 수 있어야 함 (미분 가능)

**해결책**: 손실 함수 (Loss Function)
- 예측과 정답의 차이를 수치로 표현
- 작을수록 좋은 예측 (최소화 목표)
- 미분 가능하여 Gradient Descent 사용 가능

**문제 유형별 선택**:
- **회귀 문제**: MSE, MAE, Huber Loss
- **분류 문제**: CrossEntropy, Focal Loss

---

### Step 2: 회귀 문제 손실 함수

### 이번 단계 목표
- MSE, MAE, Huber Loss의 수학적 정의와 특징 이해
- 각 손실 함수의 장단점 및 사용 시기 파악

### 입력
- 실제 값 $y$, 예측 값 $\hat{y}$

### 출력
- 손실 값 $L(y, \hat{y})$

### 평가 기준
- 각 손실 함수의 수식과 특징을 설명할 수 있음
- 이상치 상황에서 어떤 손실 함수를 선택할지 판단 가능

#### Step 2-1: MSE (Mean Squared Error)

**수학적 정의**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**통계적 의미**:
- 오차가 정규분포 $\mathcal{N}(0, \sigma^2)$를 따른다고 가정
- 최대우도추정(MLE) 관점에서 정규분포 가정 하의 최적 손실

**특징**:
- ✅ 큰 오차에 민감 (제곱 효과)
- ✅ 미분 가능, 최적화 용이
- ❌ 이상치(outlier)의 영향 큼

**사용 시기**:
- 오차가 정규분포를 따를 때
- 이상치가 적을 때
- 큰 오차를 강하게 페널티 주고 싶을 때

#### Step 2-2: MAE (Mean Absolute Error)

**수학적 정의**:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**특징**:
- ✅ 이상치에 덜 민감 (절댓값)
- ✅ 모든 오차를 동등하게 취급
- ❌ 0에서 미분 불가능 (최적화 어려움)

**사용 시기**:
- 이상치가 많을 때
- 모든 오차를 동등하게 다루고 싶을 때
- 로버스트한 예측이 필요할 때

#### Step 2-3: Huber Loss

**수학적 정의**:

$$L_\delta(y, \hat{y}) = \begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

**의미**: MSE와 MAE의 장점 결합
- 작은 오차 ($|e| \leq \delta$): MSE처럼 동작 (미분 가능)
- 큰 오차 ($|e| > \delta$): MAE처럼 동작 (이상치 강건)

**특징**:
- ✅ 이상치에 강건하면서 미분 가능
- ✅ $\delta$ 파라미터로 민감도 조절 가능

**사용 시기**:
- 이상치가 있지만 미분 가능성이 필요할 때
- MSE와 MAE의 절충안이 필요할 때

### Step 3: 회귀 손실 함수 비교 실험

### 이번 단계 목표
- MSE, MAE, Huber Loss를 실제 데이터로 비교
- 이상치가 있을 때 각 손실 함수의 동작 관찰

### 입력
- 이상치가 포함된 회귀 데이터

### 출력
- 손실 함수 비교 그래프
- 이상치 상황에서의 피팅 결과

### 평가 기준
- 각 손실 함수의 차이를 시각적으로 확인
- 이상치에 대한 반응 차이를 설명 가능

```python
# ============================================
# 손실 함수 비교 실험
# ============================================

# 실험 설정
np.random.seed(SEED)

# 손실 함수 시각화
errors = np.linspace(-3, 3, 100)

# MSE
mse = errors**2

# MAE
mae = np.abs(errors)

# Huber Loss (delta=1)
delta = 1.0
huber = np.where(np.abs(errors) <= delta,
                 0.5 * errors**2,
                 delta * np.abs(errors) - 0.5 * delta**2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(errors, mse, label='MSE', linewidth=2)
plt.plot(errors, mae, label='MAE', linewidth=2)
plt.plot(errors, huber, label='Huber', linewidth=2)
plt.xlabel('Error (y - ŷ)')
plt.ylabel('Loss')
plt.title('손실 함수 비교')
plt.legend()
plt.grid(True, alpha=0.3)

# 이상치가 있는 데이터 생성
x = np.linspace(0, 10, 50)
y_true = 2*x + 1 + np.random.randn(50) * 2
y_true[45:] += 20  # 이상치 추가 (마지막 5개)

# MSE로 학습
lr_mse = LinearRegression()
lr_mse.fit(x.reshape(-1, 1), y_true)
y_pred_mse = lr_mse.predict(x.reshape(-1, 1))

# Huber로 학습
lr_huber = HuberRegressor()
lr_huber.fit(x.reshape(-1, 1), y_true)
y_pred_huber = lr_huber.predict(x.reshape(-1, 1))

plt.subplot(1, 3, 2)
plt.scatter(x, y_true, alpha=0.5, label='데이터 (이상치 포함)', s=30)
plt.plot(x, y_pred_mse, 'r-', linewidth=2, label='MSE')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MSE (이상치에 민감)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(x, y_true, alpha=0.5, label='데이터 (이상치 포함)', s=30)
plt.plot(x, y_pred_huber, 'g-', linewidth=2, label='Huber')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Huber (이상치에 강건)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/loss_functions_comparison.png', dpi=150)
plt.show()

print("="*60)
print("관찰 사항")
print("="*60)
print("1. MSE: 이상치에 의해 기울기가 크게 영향받음")
print("2. Huber: 이상치의 영향을 줄여 더 나은 피팅")
print("3. 작은 오차에서는 MSE와 유사, 큰 오차에서는 MAE와 유사")
print("="*60)
```

### 실험 기록표 (손실 함수 비교)

| 손실 함수 | 작은 오차 반응 | 큰 오차 반응 | 이상치 강건성 | 미분 가능성 |
|-----------|---------------|-------------|-------------|------------|
| MSE | 제곱 (민감) | 제곱 (매우 민감) | 낮음 | ✅ |
| MAE | 절댓값 (보통) | 절댓값 (보통) | 높음 | ❌ (0에서) |
| Huber | 제곱 (민감) | 절댓값 (보통) | 중간 | ✅ |

---

### Step 4: 분류 문제 손실 함수

### 이번 단계 목표
- CrossEntropy Loss의 수학적 정의와 의미 이해
- Focal Loss의 필요성과 사용 시기 파악

### 입력
- 실제 클래스 $y \in \{0, 1, ..., C-1\}$
- 예측 확률 $\hat{y}_i = P(\text{class}=i)$

### 출력
- 손실 값 $L(y, \hat{y})$

### 평가 기준
- CrossEntropy의 수식과 의미를 설명할 수 있음
- 불균형 데이터셋에서 Focal Loss의 필요성을 설명 가능

#### Step 4-1: CrossEntropy Loss

**수학적 정의**:

$$\text{CE} = -\sum_{i=1}^C y_i \log(\hat{y}_i)$$

여기서:
- $y_i \in \{0, 1\}$: 실제 클래스 원-핫 인코딩
- $\hat{y}_i \in [0, 1]$: 예측 확률 (softmax 출력)
- $C$: 클래스 수

**Binary CrossEntropy** (이진 분류):

$$\text{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**통계적 의미**:
- 최대우도추정(MLE) 관점에서 다항 분포 가정 하의 최적 손실
- 정보 이론 관점에서 실제 분포와 예측 분포의 교차 엔트로피

**특징**:
- ✅ 분류 문제의 표준 손실
- ✅ 확률 해석 가능
- ✅ 미분 가능, 최적화 용이

**사용 시기**:
- 다중 분류 문제
- 클래스가 균형잡혀 있을 때

#### Step 4-2: Focal Loss

**수학적 정의**:

$$\text{FL} = -\alpha (1-\hat{y})^\gamma \log(\hat{y})$$

여기서:
- $\alpha$: 클래스 가중치 (불균형 조절)
- $\gamma$: 포커싱 파라미터 (쉬운 예제 down-weighting)

**의미**:
- 쉬운 예제($\hat{y} \approx 1$)는 $(1-\hat{y})^\gamma \approx 0$으로 가중치 감소
- 어려운 예제($\hat{y} \approx 0$)는 $(1-\hat{y})^\gamma \approx 1$로 가중치 유지

**특징**:
- ✅ 불균형 데이터셋에 효과적
- ✅ 어려운 예제에 집중
- ✅ 객체 탐지에서 널리 사용

**사용 시기**:
- 클래스 불균형이 심할 때
- 어려운 예제 학습이 중요할 때

---

## 2. 정규화 기법

### Step 1: 정규화란 무엇인가?

### 이번 단계 목표
- 정규화의 목적과 필요성 이해
- 과적합 방지 메커니즘 파악

### 입력
- 없음 (개념 설명)

### 출력
- 정규화의 역할 이해
- 과적합 방지 원리

### 평가 기준
- "왜 정규화가 필요한가?"를 설명할 수 있음

### 왜 정규화가 필요한가?

**문제 상황**: 과적합 (Overfitting)
- 모델이 학습 데이터에만 과도하게 맞춤
- 학습 데이터: 낮은 손실, 높은 정확도
- 검증 데이터: 높은 손실, 낮은 정확도
- 일반화 성능 저하

**해결책**: 정규화 (Regularization)
- 모델 복잡도를 제한
- 가중치 크기 제약
- 학습 데이터에 과도하게 맞추는 것 방지

**정규화 기법**:
1. **L1/L2 정규화**: 가중치 크기 제약
2. **Dropout**: 뉴런 랜덤 비활성화
3. **Batch Normalization**: 통계적 정규화
4. **Early Stopping**: 검증 손실 기준 조기 종료

---

### Step 2: L1과 L2 정규화

### 이번 단계 목표
- L1, L2 정규화의 수학적 정의와 기하학적 해석 이해
- 각 정규화의 효과 차이 파악

### 입력
- 데이터 손실 $L_{\text{data}}$
- 가중치 $\mathbf{w}$

### 출력
- 총 손실 $L_{\text{total}} = L_{\text{data}} + \lambda R(\mathbf{w})$

### 평가 기준
- L1과 L2 정규화의 수식과 효과를 설명할 수 있음
- 특징 선택 vs 가중치 수축의 차이를 이해

#### Step 2-1: L2 정규화 (Ridge, Weight Decay)

**수학적 정의**:

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda \sum_{i} w_i^2$$

**기하학적 해석**:
- 가중치를 원점으로 수축 (shrinkage)
- 모든 가중치를 골고루 작게 만듦
- 부드러운 모델 (smooth)

**특징**:
- ✅ 미분 가능 (최적화 용이)
- ✅ 가중치를 0에 가깝게 만듦 (완전히 0은 아님)
- ✅ PyTorch: `weight_decay` 파라미터

**사용 시기**:
- 모든 특징이 유용할 때
- 가중치를 작게 유지하고 싶을 때

#### Step 2-2: L1 정규화 (Lasso)

**수학적 정의**:

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda \sum_{i} |w_i|$$

**기하학적 해석**:
- 일부 가중치를 정확히 0으로 만듦
- 특징 선택 효과 (feature selection)
- 희소한 모델 (sparse)

**특징**:
- ✅ 불필요한 특징 제거
- ❌ 0에서 미분 불가능 (최적화 어려움)
- ✅ 해석 가능성 향상

**사용 시기**:
- 불필요한 특징이 많을 때
- 특징 선택이 필요할 때
- 모델 해석이 중요할 때

#### Step 2-3: Elastic Net

**수학적 정의**:

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

**의미**: L1과 L2의 결합
- L1: 특징 선택
- L2: 가중치 수축

**사용 시기**:
- L1과 L2의 장점을 모두 원할 때

### Step 3: 정규화 효과 실험

### 이번 단계 목표
- L2 정규화의 효과를 실제 모델로 관찰
- 가중치 크기 변화와 분포 비교

### 입력
- 간단한 선형 모델
- 동일 데이터셋

### 출력
- 정규화 전/후 가중치 크기 변화 그래프
- 가중치 분포 히스토그램

### 평가 기준
- 정규화가 가중치를 작게 만드는 것을 확인
- 정규화 전/후 차이를 설명 가능

```python
# ============================================
# 정규화 효과 실험
# ============================================

# 실험 설정
torch.manual_seed(SEED)

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# 데이터 생성
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# 정규화 없음
model_none = SimpleModel()
optimizer_none = torch.optim.SGD(model_none.parameters(), lr=0.01)

# L2 정규화 (weight_decay)
model_l2 = SimpleModel()
optimizer_l2 = torch.optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.1)

# 학습
criterion = nn.MSELoss()
epochs = 100

weights_none = []
weights_l2 = []

for epoch in range(epochs):
    # 정규화 없음
    optimizer_none.zero_grad()
    loss_none = criterion(model_none(X), y)
    loss_none.backward()
    optimizer_none.step()
    weights_none.append(model_none.fc.weight.detach().clone().norm().item())
    
    # L2 정규화
    optimizer_l2.zero_grad()
    loss_l2 = criterion(model_l2(X), y)
    loss_l2.backward()
    optimizer_l2.step()
    weights_l2.append(model_l2.fc.weight.detach().clone().norm().item())

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(weights_none, label='정규화 없음', linewidth=2)
plt.plot(weights_l2, label='L2 정규화', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Weight Norm (||w||)')
plt.title('가중치 크기 변화')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
w_none = model_none.fc.weight.detach().numpy().flatten()
w_l2 = model_l2.fc.weight.detach().numpy().flatten()
plt.hist(w_none, bins=20, alpha=0.5, label='정규화 없음', color='blue')
plt.hist(w_l2, bins=20, alpha=0.5, label='L2 정규화', color='red')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('가중치 분포')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/regularization_effect.png', dpi=150)
plt.show()

print("="*60)
print("관찰 사항")
print("="*60)
print(f"정규화 없음 - Weight Norm: {weights_none[-1]:.4f}")
print(f"L2 정규화 - Weight Norm: {weights_l2[-1]:.4f}")
print("\nL2 정규화는 가중치를 더 작고 균일하게 유지합니다.")
print("="*60)
```

### 실험 기록표 (정규화 비교)

| 정규화 방법 | 가중치 효과 | 특징 선택 | 미분 가능성 | PyTorch 구현 |
|------------|------------|----------|------------|-------------|
| 없음 | 제한 없음 | 없음 | ✅ | - |
| L2 (Ridge) | 수축 (0에 가까움) | 없음 | ✅ | `weight_decay` |
| L1 (Lasso) | 희소화 (정확히 0) | 있음 | ❌ (0에서) | 직접 구현 |
| Elastic Net | 수축 + 희소화 | 있음 | ❌ (0에서) | 직접 구현 |

---

### Step 4: Dropout

### 이번 단계 목표
- Dropout의 원리와 앙상블 관점 이해
- 학습/평가 모드 차이 파악

### 입력
- 뉴런 출력 $x$
- Dropout 확률 $p$

### 출력
- Dropout 적용 출력 $y$

### 평가 기준
- Dropout이 과적합을 방지하는 원리를 설명할 수 있음
- 학습/평가 모드에서의 동작 차이를 이해

**수학적 정의**:

$$y = \text{dropout}(x, p) = \begin{cases} 
0 & \text{확률 } p \\
\frac{x}{1-p} & \text{확률 } 1-p
\end{cases}$$

**앙상블 관점**:
- 매 iteration마다 다른 서브네트워크 학습
- 여러 모델의 앙상블 효과
- 뉴런 간 co-adaptation 방지

**학습/평가 모드**:
- **학습 모드**: 랜덤하게 뉴런 비활성화, 출력 스케일링 ($\frac{x}{1-p}$)
- **평가 모드**: 모든 뉴런 활성화, 스케일링 없음

**PyTorch 구현**:

```python
# Dropout 레이어 정의
dropout = nn.Dropout(p=0.5)

# 학습 모드
model.train()
x = torch.randn(10, 20)
x_dropped = dropout(x)  # 일부 뉴런 0, 나머지는 2배 스케일링

# 평가 모드
model.eval()
x_dropped = dropout(x)  # 모든 뉴런 활성화, 스케일링 없음
```

---

### Step 5: Batch Normalization

### 이번 단계 목표
- Batch Normalization의 통계적 정규화 원리 이해
- 학습 안정화 메커니즘 파악

### 입력
- 배치 데이터 $x$
- 학습 파라미터 $\gamma$, $\beta$

### 출력
- 정규화된 출력 $y$

### 평가 기준
- Batch Normalization이 학습을 안정화하는 원리를 설명할 수 있음

**수학적 정의**:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

여기서:
- $\mu_B$: 배치 평균
- $\sigma_B^2$: 배치 분산
- $\epsilon$: 수치 안정성 상수
- $\gamma$, $\beta$: 학습 가능한 파라미터

**효과**:
- 내부 공변량 이동(Internal Covariate Shift) 감소
- 학습 속도 향상
- 정규화 효과
- 더 큰 learning rate 사용 가능

**PyTorch 구현**:

```python
# Batch Normalization 레이어
bn = nn.BatchNorm2d(num_features=32)  # 2D (이미지)
# 또는
bn = nn.BatchNorm1d(num_features=128)  # 1D (벡터)

# 학습 모드: 배치 통계 사용
model.train()
x = torch.randn(32, 32, 28, 28)  # (B, C, H, W)
x_norm = bn(x)

# 평가 모드: 이동 평균 통계 사용
model.eval()
x_norm = bn(x)
```

---

### Step 6: Early Stopping

### 이번 단계 목표
- Early Stopping의 원리와 구현 방법 이해
- 과적합 방지 메커니즘 파악

### 입력
- 검증 손실 히스토리
- Patience 파라미터

### 출력
- 학습 중단 시점 결정

### 평가 기준
- Early Stopping이 과적합을 방지하는 원리를 설명할 수 있음

**아이디어**: 검증 손실이 증가하면 학습 중단

**원리**:
- 과적합 전에 학습 중단
- 가장 간단한 정규화
- 계산 비용 절감

**구현 예시**:

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    # 학습 및 검증
    train_loss = train_epoch(...)
    val_loss = validate(...)
    
    # 최고 성능 업데이트
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 최고 모델 저장
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    # Early Stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

---

## 3. 평가 지표

### Step 1: 평가 지표란 무엇인가?

### 이번 단계 목표
- 평가 지표의 역할과 필요성 이해
- Accuracy만으로 부족한 이유 파악

### 입력
- 없음 (개념 설명)

### 출력
- 평가 지표의 역할 이해
- 문제 유형별 적합한 지표 선택 기준

### 평가 기준
- "왜 Accuracy만으로 부족한가?"를 설명할 수 있음

### 왜 Accuracy만으로 부족한가?

**문제 상황**:
- 불균형 데이터셋에서 Accuracy는 오해의 소지
- 예: 100개 중 95개가 클래스 A, 5개가 클래스 B
- 모든 것을 A로 예측하면 Accuracy = 95%
- 하지만 클래스 B를 전혀 예측하지 못함

**해결책**: 다양한 평가 지표 사용
- **분류**: Precision, Recall, F1, AUC
- **회귀**: MAE, RMSE, R²
- **진단**: Confusion Matrix

---

### Step 2: 분류 문제 평가 지표

### 이번 단계 목표
- Confusion Matrix 기반 지표 이해
- Precision, Recall, F1의 의미와 계산 방법 파악

### 입력
- 실제 라벨 $y \in \{0, 1\}$ (이진 분류)
- 예측 라벨 $\hat{y} \in \{0, 1\}$

### 출력
- TP, TN, FP, FN
- Accuracy, Precision, Recall, F1

### 평가 기준
- 각 지표의 수식과 의미를 설명할 수 있음
- 불균형 데이터에서 어떤 지표를 봐야 하는지 판단 가능

#### Step 2-1: Confusion Matrix

**정의**:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | TP (True Positive)  | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative)  |

**의미**:
- **TP**: 양성으로 올바르게 예측
- **TN**: 음성으로 올바르게 예측
- **FP**: 양성으로 잘못 예측 (Type I Error)
- **FN**: 음성으로 잘못 예측 (Type II Error)

#### Step 2-2: 주요 지표

**Accuracy (정확도)**:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- 전체 중 올바르게 예측한 비율
- 균형잡힌 데이터에서 유용

**Precision (정밀도)**:

$$\text{Precision} = \frac{TP}{TP + FP}$$

- "양성으로 예측한 것 중 실제 양성 비율"
- False Positive를 줄이고 싶을 때 중요

**Recall (재현율, Sensitivity)**:

$$\text{Recall} = \frac{TP}{TP + FN}$$

- "실제 양성 중 올바르게 예측한 비율"
- False Negative를 줄이고 싶을 때 중요

**F1 Score**:

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- Precision과 Recall의 조화평균
- 불균형 데이터에서 유용

#### Step 2-3: ROC Curve와 AUC

**ROC Curve**: TPR vs FPR 곡선
- **TPR (True Positive Rate)** = Recall = $\frac{TP}{TP + FN}$
- **FPR (False Positive Rate)** = $\frac{FP}{FP + TN}$

**AUC (Area Under Curve)**: ROC 곡선 아래 면적
- 1.0: 완벽한 분류기
- 0.5: 랜덤 분류기
- 0.5 ~ 1.0: 분류 성능

### Step 3: 평가 지표 계산 실험

### 이번 단계 목표
- 실제 예측 결과로 평가 지표 계산
- Confusion Matrix, ROC Curve 시각화

### 입력
- 실제 라벨, 예측 라벨, 예측 확률

### 출력
- 평가 지표 값
- 시각화 그래프

### 평가 기준
- 각 지표를 계산하고 해석할 수 있음

```python
# ============================================
# 평가 지표 계산 실험
# ============================================

# 예측 결과 시뮬레이션
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
y_scores = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.3, 0.6, 0.85, 0.15])

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# 지표 계산
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("="*60)
print("평가 지표")
print("="*60)
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("="*60)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)

plt.subplot(1, 3, 3)
plt.plot(recall_vals, precision_vals, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/evaluation_metrics.png', dpi=150)
plt.show()

print(f"\nROC AUC: {roc_auc:.3f}")
```

---

### Step 4: 회귀 문제 평가 지표

### 이번 단계 목표
- MAE, RMSE, R²의 수학적 정의와 의미 이해
- 각 지표의 해석 방법 파악

### 입력
- 실제 값 $y$, 예측 값 $\hat{y}$

### 출력
- MAE, RMSE, R² 값

### 평가 기준
- 각 지표의 수식과 의미를 설명할 수 있음

**MAE (Mean Absolute Error)**:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

- 평균 절댓값 오차
- 단위가 원본과 동일

**RMSE (Root Mean Squared Error)**:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

- 평균 제곱 오차의 제곱근
- 큰 오차에 민감

**R² (결정계수)**:

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

- 1.0: 완벽한 예측
- 0.0: 평균만큼 예측
- 음수: 평균보다 나쁨

---

## 4. 편향-분산 트레이드오프

### Step 1: 편향-분산이란 무엇인가?

### 이번 단계 목표
- 편향과 분산의 개념 이해
- 총 오차 분해 공식 이해

### 입력
- 없음 (개념 설명)

### 출력
- 편향과 분산의 정의
- 총 오차 분해 공식

### 평가 기준
- 편향과 분산의 의미를 설명할 수 있음
- 총 오차 분해 공식을 이해

### 총 오차 분해

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Bias (편향)**:
- 정의: 모델의 예측 평균과 실제 값의 차이
- 의미: 모델이 얼마나 단순한가?
- 높은 편향: 언더피팅 (underfitting)

**Variance (분산)**:
- 정의: 다른 학습 데이터에 대한 예측의 변동성
- 의미: 모델이 얼마나 복잡한가?
- 높은 분산: 오버피팅 (overfitting)

**Irreducible Error**:
- 정의: 데이터 자체의 노이즈로 인한 오차
- 의미: 모델로 줄일 수 없는 오차

---

### Step 2: 모델 복잡도와의 관계

### 이번 단계 목표
- 모델 복잡도에 따른 편향-분산 변화 이해
- 최적 복잡도 선택 기준 파악

### 입력
- 모델 복잡도 (파라미터 수, 레이어 수 등)

### 출력
- 편향, 분산, 총 오차

### 평가 기준
- 모델 복잡도와 편향-분산의 관계를 설명할 수 있음

**관계**:

```
모델 복잡도: 낮음 ────────────────> 높음
편향:        높음 ────────────────> 낮음
분산:        낮음 ────────────────> 높음
총 오차:     높음 → 최소 → 높음
```

**최적점**: 편향과 분산의 균형
- 너무 단순: 높은 편향 (언더피팅)
- 너무 복잡: 높은 분산 (오버피팅)
- 적절한 복잡도: 편향과 분산의 균형

---

### Step 3: 오버피팅 vs 언더피팅

### 이번 단계 목표
- 오버피팅과 언더피팅의 증상과 해결책 이해
- 학습/검증 곡선으로 진단 방법 파악

### 입력
- 학습/검증 손실 곡선

### 출력
- 오버피팅/언더피팅 진단
- 해결책 제시

### 평가 기준
- 학습/검증 곡선을 보고 오버피팅/언더피팅을 진단할 수 있음

#### 오버피팅 (Overfitting)

**증상**:
- 학습 손실: 낮음
- 검증 손실: 높음
- 학습 데이터에만 과도하게 맞춤

**해결책**:
- 더 많은 데이터
- 정규화 (L1, L2, Dropout)
- 모델 단순화
- Early Stopping

#### 언더피팅 (Underfitting)

**증상**:
- 학습 손실: 높음
- 검증 손실: 높음
- 데이터의 패턴을 학습하지 못함

**해결책**:
- 모델 복잡도 증가
- 더 많은 특징
- 정규화 감소
- 더 오래 학습

---

### Step 4: 일반화 (Generalization)

### 이번 단계 목표
- 일반화의 의미와 중요성 이해
- 일반화 능력 향상 방법 파악

### 입력
- 없음 (개념 설명)

### 출력
- 일반화 능력 향상 방법 리스트

### 평가 기준
- 일반화의 의미를 설명할 수 있음
- 일반화 능력 향상 방법을 제시할 수 있음

**목표**: 학습 데이터가 아닌 **새로운 데이터**에서 좋은 성능

**일반화 능력 향상 방법**:
1. 충분한 학습 데이터
2. 적절한 모델 복잡도
3. 정규화 기법
4. 교차 검증 (Cross-Validation)
5. 앙상블 (Ensemble)

---

## 5. 의도된 실패 실험

### Step 1: 정규화 없이 학습 (과적합 관찰)

### 이번 단계 목표
- 정규화 없이 학습하여 과적합 현상 관찰
- 정규화의 필요성을 직접 경험

### 입력
- 정규화 없는 모델
- 학습/검증 데이터

### 출력
- 과적합 학습 곡선
- 정규화 포함 모델과 비교

### 평가 기준
- 과적합 증상을 관찰하고 설명할 수 있음

**실험 설계**:
- **고정 변수**: 데이터, 모델 구조, 학습률, 에폭
- **조작 변수**: 정규화 유무 (Dropout, Weight Decay)

**예상 결과**:
- 정규화 없음: 학습 손실 ↓, 검증 손실 ↑ (과적합)
- 정규화 있음: 학습 손실 ≈ 검증 손실 (균형)

---

## 6. 핵심 요약

### 이번 단원에서 배운 내용

#### 1. 손실 함수의 선택
- **MSE**: 정규분포 가정, 이상치에 민감
- **MAE**: 이상치에 강건, 미분 불가능
- **Huber**: MSE와 MAE의 장점 결합
- **CrossEntropy**: 분류 문제의 표준
- **Focal Loss**: 불균형 데이터셋

#### 2. 정규화 기법
- **L1**: 희소성, 특징 선택
- **L2**: 가중치 수축, 부드러운 모델
- **Dropout**: 앙상블 효과, co-adaptation 방지
- **BatchNorm**: 통계적 정규화, 학습 안정화
- **Early Stopping**: 가장 간단한 정규화

#### 3. 평가 지표
- **분류**: Accuracy, Precision, Recall, F1, AUC
- **회귀**: MAE, RMSE, R²
- **Confusion Matrix**: 오분류 패턴 분석
- **ROC Curve**: 임계값에 따른 성능

#### 4. 편향-분산 트레이드오프
- **편향**: 모델의 단순성, 언더피팅
- **분산**: 모델의 복잡성, 오버피팅
- **최적점**: 편향과 분산의 균형
- **일반화**: 새로운 데이터에서의 성능

#### 5. 실전 팁
- 이상치가 많으면 Huber Loss 사용
- 과적합이면 정규화 강화
- 불균형 데이터면 F1, AUC 확인
- 학습/검증 곡선으로 진단

---

## 추가 도전 과제

### 과제 1: 손실 함수 비교 실험 (평가 가능)

**목표**: 동일 데이터에서 MSE, MAE, Huber Loss로 학습하여 성능 비교

**산출물**:
- 실험 기록표 (고정: 데이터/모델, 조작: 손실 함수)
- 학습 곡선 비교 그래프
- 이상치 상황에서의 성능 비교 리포트

**제한 조건**:
- 동일 모델 구조 사용
- 동일 시드 사용
- 각 실험 50 에폭

**채점 기준**:
- 실험 기록표 완성: 50점
- 그래프 및 분석: 50점

### 과제 2: 정규화 효과 비교 (실험 기록)

**목표**: L1, L2, Dropout의 효과를 비교

**산출물**:
- 실험 기록표 (고정: 모델/데이터, 조작: 정규화 방법)
- 가중치 분포 비교 그래프
- 검증 성능 비교표

### 과제 3: 의도된 실패 실험 (안티패턴 학습)

**목표**: 정규화 없이 학습하여 과적합 관찰

**산출물**:
- 정규화 없이 학습한 모델의 학습 곡선
- 정규화 포함 모델과 비교 그래프
- 과적합 증상 설명 (1문단)

**실험 설계**:
- 모델: MLP (은닉층 3개)
- 에폭: 30
- 정규화 없음 vs Dropout(0.5) + Weight Decay(1e-4)

**채점 기준**:
- 과적합 관찰 및 설명: 만점
- 그래프만 제시: 50점

