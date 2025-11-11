# 11. 머신러닝 개념 심화

이번 단원에서는 **머신러닝의 핵심 개념**을 깊이 있게 배워보겠습니다.

## 학습 목표
- 손실 함수의 통계적 의미
- 정규화 기법의 원리
- 평가 지표의 이해와 활용
- 편향-분산 트레이드오프
- 오버피팅과 언더피팅

## 왜 이 단원이 필요한가?

머신러닝 모델을 제대로 이해하고 개선하려면 **왜 그렇게 작동하는지** 알아야 합니다.

- **손실 함수**: 왜 MSE와 CrossEntropy를 사용하는가?
- **정규화**: 왜 L1, L2, Dropout이 과적합을 방지하는가?
- **평가 지표**: Accuracy만으로는 왜 부족한가?
- **편향-분산**: 모델 복잡도를 어떻게 조절하는가?

이 단원을 이해하면 모델의 성능을 체계적으로 개선할 수 있습니다.

---
# 1. 손실 함수의 심화
---

## 1.1 손실 함수의 통계적 의미

### MSE (Mean Squared Error)

**가정**: 오차가 정규분포 $\mathcal{N}(0, \sigma^2)$를 따름

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**특징**:
- 큰 오차에 민감 (제곱)
- 이상치(outlier)의 영향 큼
- 미분 가능, 최적화 용이

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**특징**:
- 이상치에 덜 민감
- 모든 오차를 동등하게 취급
- 0에서 미분 불가능

### Huber Loss

MSE와 MAE의 장점을 결합:

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

**특징**:
- 작은 오차: MSE처럼 동작
- 큰 오차: MAE처럼 동작
- 이상치에 강건하면서 미분 가능

### CrossEntropy Loss

**분류 문제의 표준**:

$$\text{CE} = -\sum_{i=1}^C y_i \log(\hat{y}_i)$$

**Binary CrossEntropy**:

$$\text{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

### Focal Loss

**불균형 데이터셋을 위한 손실**:

$$\text{FL} = -\alpha (1-\hat{y})^\gamma \log(\hat{y})$$

**특징**:
- 쉬운 예제의 가중치 감소
- 어려운 예제에 집중
- 객체 탐지에서 효과적

---
# 2. 정규화 기법
---

## 2.1 L1과 L2 정규화

### L2 정규화 (Ridge)

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda \sum_{i} w_i^2$$

**기하학적 해석**:
- 가중치를 원점으로 수축
- 모든 가중치를 골고루 작게 만듦
- 부드러운 모델 (smooth)

### L1 정규화 (Lasso)

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda \sum_{i} |w_i|$$

**기하학적 해석**:
- 일부 가중치를 정확히 0으로
- 특징 선택 효과 (feature selection)
- 희소한 모델 (sparse)

### Elastic Net

L1과 L2의 결합:

$$\text{Loss}_{\text{total}} = \text{Loss}_{\text{data}} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

## 2.2 Dropout

**아이디어**: 학습 시 랜덤하게 뉴런 비활성화

**앙상블 관점**:
- 매 iteration마다 다른 서브네트워크 학습
- 여러 모델의 앙상블 효과
- 뉴런 간 co-adaptation 방지

**수식**:

$$y = \text{dropout}(x, p) = \begin{cases} 0 & \text{확률 } p \\ \frac{x}{1-p} & \text{확률 } 1-p \end{cases}$$

## 2.3 Batch Normalization

**통계적 정규화**:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

**효과**:
- 내부 공변량 이동 감소
- 학습 속도 향상
- 정규화 효과
- 더 큰 learning rate 사용 가능

## 2.4 Early Stopping

**아이디어**: 검증 손실이 증가하면 학습 중단

**원리**:
- 과적합 전에 학습 중단
- 가장 간단한 정규화
- 계산 비용 절감

---
# 3. 평가 지표
---

## 3.1 분류 문제 지표

### Confusion Matrix

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | TP (True Positive)  | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative)  |

### 주요 지표

**Accuracy (정확도)**:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision (정밀도)**:

$$\text{Precision} = \frac{TP}{TP + FP}$$

"양성으로 예측한 것 중 실제 양성 비율"

**Recall (재현율, Sensitivity)**:

$$\text{Recall} = \frac{TP}{TP + FN}$$

"실제 양성 중 올바르게 예측한 비율"

**F1 Score**:

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Precision과 Recall의 조화평균

### ROC Curve와 AUC

**ROC Curve**: TPR vs FPR 곡선

- TPR (True Positive Rate) = Recall
- FPR (False Positive Rate) = $\frac{FP}{FP + TN}$

**AUC (Area Under Curve)**: ROC 곡선 아래 면적

- 1.0: 완벽한 분류기
- 0.5: 랜덤 분류기

## 3.2 회귀 문제 지표

**MAE (Mean Absolute Error)**:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**RMSE (Root Mean Squared Error)**:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**R² (결정계수)**:

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

- 1.0: 완벽한 예측
- 0.0: 평균만큼 예측

---
# 4. 편향-분산 트레이드오프
---

## 4.1 개념 이해

**총 오차 분해**:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Bias (편향)

- **정의**: 모델의 예측 평균과 실제 값의 차이
- **의미**: 모델이 얼마나 단순한가?
- **높은 편향**: 언더피팅 (underfitting)

### Variance (분산)

- **정의**: 다른 학습 데이터에 대한 예측의 변동성
- **의미**: 모델이 얼마나 복잡한가?
- **높은 분산**: 오버피팅 (overfitting)

## 4.2 모델 복잡도와의 관계

```
모델 복잡도 낮음 ────────────────> 높음
편향:    높음 ────────────────> 낮음
분산:    낮음 ────────────────> 높음
```

**최적점**: 편향과 분산의 균형

## 4.3 오버피팅 vs 언더피팅

### 오버피팅 (Overfitting)

**증상**:
- 학습 손실: 낮음
- 검증 손실: 높음
- 학습 데이터에만 과도하게 맞춤

**해결책**:
- 더 많은 데이터
- 정규화 (L1, L2, Dropout)
- 모델 단순화
- Early Stopping

### 언더피팅 (Underfitting)

**증상**:
- 학습 손실: 높음
- 검증 손실: 높음
- 데이터의 패턴을 학습하지 못함

**해결책**:
- 모델 복잡도 증가
- 더 많은 특징
- 정규화 감소
- 더 오래 학습

## 4.4 일반화 (Generalization)

**목표**: 학습 데이터가 아닌 **새로운 데이터**에서 좋은 성능

**일반화 능력 향상 방법**:
1. 충분한 학습 데이터
2. 적절한 모델 복잡도
3. 정규화 기법
4. 교차 검증 (Cross-Validation)
5. 앙상블 (Ensemble)

---
# 5. PyTorch 구현
---

## 5.1 다양한 손실 함수 비교

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 다양한 손실 함수 비교
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
plt.xlabel('Error')
plt.ylabel('Loss')
plt.title('손실 함수 비교')
plt.legend()
plt.grid(True, alpha=0.3)

# 이상치가 있는 데이터
np.random.seed(42)
x = np.linspace(0, 10, 50)
y_true = 2*x + 1 + np.random.randn(50) * 2
y_true[45:] += 20  # 이상치 추가

# MSE로 학습
from sklearn.linear_model import LinearRegression, HuberRegressor

lr_mse = LinearRegression()
lr_mse.fit(x.reshape(-1, 1), y_true)
y_pred_mse = lr_mse.predict(x.reshape(-1, 1))

# Huber로 학습
lr_huber = HuberRegressor()
lr_huber.fit(x.reshape(-1, 1), y_true)
y_pred_huber = lr_huber.predict(x.reshape(-1, 1))

plt.subplot(1, 3, 2)
plt.scatter(x, y_true, alpha=0.5, label='데이터 (이상치 포함)')
plt.plot(x, y_pred_mse, 'r-', linewidth=2, label='MSE')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MSE (이상치에 민감)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(x, y_true, alpha=0.5, label='데이터 (이상치 포함)')
plt.plot(x, y_pred_huber, 'g-', linewidth=2, label='Huber')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Huber (이상치에 강건)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("관찰:")
print("- MSE: 이상치에 의해 기울기가 크게 영향받음")
print("- Huber: 이상치의 영향을 줄여 더 나은 피팅")
```

## 5.2 정규화 효과 시각화

```python
import torch
import torch.nn as nn

# L1 vs L2 정규화 비교
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# 데이터 생성
torch.manual_seed(42)
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# 정규화 없음
model_none = SimpleModel()
optimizer_none = torch.optim.SGD(model_none.parameters(), lr=0.01)

# L2 정규화
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
plt.ylabel('Weight Norm')
plt.title('가중치 크기 변화')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
w_none = model_none.fc.weight.detach().numpy().flatten()
w_l2 = model_l2.fc.weight.detach().numpy().flatten()
plt.hist(w_none, bins=20, alpha=0.5, label='정규화 없음')
plt.hist(w_l2, bins=20, alpha=0.5, label='L2 정규화')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('가중치 분포')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"정규화 없음 - Weight Norm: {weights_none[-1]:.4f}")
print(f"L2 정규화 - Weight Norm: {weights_l2[-1]:.4f}")
print("\nL2 정규화는 가중치를 더 작고 균일하게 유지합니다.")
```

## 5.3 평가 지표 계산

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# 예측 결과 시뮬레이션
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
y_scores = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.3, 0.6, 0.85, 0.15])

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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

print("평가 지표:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

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
from sklearn.metrics import precision_recall_curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)

plt.subplot(1, 3, 3)
plt.plot(recall_vals, precision_vals, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nROC AUC: {roc_auc:.3f}")
```

---
# 6. 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 손실 함수의 선택
- **MSE**: 정규분포 가정, 이상치에 민감
- **MAE**: 이상치에 강건, 미분 불가능
- **Huber**: MSE와 MAE의 장점 결합
- **CrossEntropy**: 분류 문제의 표준
- **Focal Loss**: 불균형 데이터셋

### 2. 정규화 기법
- **L1**: 희소성, 특징 선택
- **L2**: 가중치 수축, 부드러운 모델
- **Dropout**: 앙상블 효과, co-adaptation 방지
- **BatchNorm**: 통계적 정규화, 학습 안정화
- **Early Stopping**: 가장 간단한 정규화

### 3. 평가 지표
- **분류**: Accuracy, Precision, Recall, F1, AUC
- **회귀**: MAE, RMSE, R²
- **Confusion Matrix**: 오분류 패턴 분석
- **ROC Curve**: 임계값에 따른 성능

### 4. 편향-분산 트레이드오프
- **편향**: 모델의 단순성, 언더피팅
- **분산**: 모델의 복잡성, 오버피팅
- **최적점**: 편향과 분산의 균형
- **일반화**: 새로운 데이터에서의 성능

### 5. 실전 팁
- 이상치가 많으면 Huber Loss 사용
- 과적합이면 정규화 강화
- 불균형 데이터면 F1, AUC 확인
- 학습/검증 곡선으로 진단

다음 단원에서는 **디버깅 전략**을 배워보겠습니다!
