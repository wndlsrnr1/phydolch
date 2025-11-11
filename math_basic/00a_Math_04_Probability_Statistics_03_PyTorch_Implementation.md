# 00b. 확률과 통계 기초 (Probability & Statistics Foundations) - Part 3: PyTorch 구현

**이 파일은 Part 3: PyTorch 구현입니다.**

---
# 3. PyTorch 구현 (How)
---

## 3.1 확률 분포 시각화

- 2장에서 다룬 정규분포의 평균/분산 역할을 실제 곡선으로 확인합니다.
- 68-95-99.7 법칙이 무엇을 의미하는지 눈으로 확인해요.
- 이후 MSE와 CrossEntropy를 이해할 때 사용할 "정규분포 모양" 감각을 갖춥니다.

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

**그래프에서 볼 것**
- (왼쪽) 평균이 달라지면 그래프 전체가 좌우로 이동합니다. 이론에서 말한 "분포의 중심"이 시각적으로 확인돼요.
- (가운데) 분산을 키우면 종 모양이 납작해지고, 줄이면 뾰족해집니다. 분산이 클수록 불확실성이 크다는 말이죠.
- (오른쪽) 68-95-99.7 법칙이 면적으로 표시돼서, 표준정규분포에서 어느 범위까지 데이터를 기대할 수 있는지 한눈에 볼 수 있습니다.

## 3.2 MSE와 정규분포의 연결

- 2d 단원에서 본 "정규분포 가정 → MSE" 흐름을 코드로 재현합니다.
- 인공 데이터를 만들고, MSE Loss가 실제로 얼마나 작아지는지 확인해요.
- 오차 히스토그램을 통해 "정규분포를 따른다"는 가정이 어떻게 드러나는지도 살펴봅니다.

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

**시각화 해석 가이드**
- 왼쪽 산점도에서 붉은 직선(모델 예측)이 노이즈가 섞인 데이터와 거의 겹치면, MLE가 찾은 파라미터가 실제 분포를 잘 설명한다는 뜻이에요.
- 오른쪽 히스토그램은 오차 분포를 보여줍니다. 파란 막대가 정규곡선(빨간 선)과 비슷하면 "오차가 정규분포"라는 가정이 코드에서도 확인됩니다.
- 만약 히스토그램이 비대칭이거나 꼬리가 길어지면, MSE 대신 다른 손실을 고민해야 한다는 신호예요.

## 3.3 CrossEntropy와 다항분포의 연결

- 소프트맥스 확률이 어떻게 만들어지고, 정답 클래스의 로그 확률이 손실과 직결되는지 확인합니다.
- 다항(카테고리) 분포 가정이 코드에서 어떻게 나타나는지 단계별로 따라갑니다.
- 시각화로 손실 곡선과 로그 우도를 함께 보며 이론과 감각을 연결해요.

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

**그래프 읽는 법**
- 첫 번째 막대 그래프: 소프트맥스가 만들어 준 확률 분포를 보여줘요. 정답 클래스 막대가 높을수록 교차 엔트로피가 작아집니다.
- 두 번째 곡선: $-\log p$가 확률에 따라 어떻게 변하는지. 빨간 점이 현재 손실 위치예요.
- 세 번째 그래프: 각 클래스의 로그 확률. 로그 우도를 최대화하면 정답 막대가 덜 음수(=더 큰 값)가 되는 걸 볼 수 있죠.

## 3.4 실제 학습 루프 예제

**목적**: 손실 함수가 실제로 어떻게 최적화에 사용되는지 보여드립니다.

이제까지는 손실을 계산만 했는데, 실제 학습에서는 어떻게 쓰일까요?

- NLL(=MSE)을 최소화하기 위해
  1. **Forward**: 우도를 계산해 손실을 얻고
  2. **Backward**: 손실의 기울기를 구해
  3. **Update**: 파라미터를 한 걸음 이동한다는 이론을 그대로 구현합니다.
- 각 단계가 2.3절의 MLE 공식과 어떻게 연결되는지 코드를 통해 확인해요.

**그래프 체크 포인트**
- 학습 곡선이 꾸준히 내려가면 경사하강법이 Loss(=NLL)를 잘 줄이고 있다는 뜻이에요.
- 오른쪽 그래프에서 붉은 선(학습된 모델)과 초록 점선(실제 분포)을 비교해 보세요. 두 선이 겹칠수록 파라미터가 참값에 가깝습니다.

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

**그래프 해석**
- 히스토그램이 정규곡선과 잘 맞으면, MSE의 정규분포 가정이 현실 데이터에서도 합리적이라는 뜻이에요.
- Q-Q Plot에서 점들이 대각선 위에 있으면 정규분포, 아래/위로 휘면 다른 분포를 의심해야 합니다.

### 배치 처리 패턴

**왜 배치를 사용하나요?** 전체 데이터를 한 번에 처리하면 메모리 부족이 발생할 수 있어요. 작은 배치로 나눠서 처리합니다.

- 통계적으로는 "미니배치가 데이터 분포의 근사 샘플"이 되기 때문에, 우도를 안정적으로 근사할 수 있어요.
- 매 배치에서 계산한 기울기는 전체 데이터 기울기의 추정치 → 확률적 경사하강법(SGD)의 핵심 아이디어죠.

**시각화에서 포인트 잡기**
- 왼쪽/오른쪽 비교 그림에서 이상치가 있을 때 회귀선이 어떻게 뒤틀리는지 봅니다.
- 오차 히스토그램이 한쪽으로 길게 늘어지면 정규분포 가정이 깨진 것이고, 제곱 오차가 급격히 커지는 이유를 직관적으로 파악할 수 있어요.

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

- 2d에서 배운 가정을 직접 검정(statistical test)으로 확인합니다.
- 가정이 틀릴 때 어떤 문제가 생기는지, 그리고 어떤 대안을 고려해야 하는지도 실험합니다.

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

- 이 실험을 통해 "가정 → 손실 함수"의 연결 고리를 다시 점검하고, 다른 분포/손실을 선택해야 하는 순간을 체감합니다.

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
