# 08. 기타 레이어들

이번 단원에서는 신경망의 추가 구성요소들을 배워보겠습니다.

## 학습 목표
- Pooling Layer (풀링 레이어)
- Non-linear Activations (비선형 활성화 함수)
- 정규화 및 규제 레이어 (Dropout, BatchNorm)
- 모델 정의 (nn.Module)

## 이 단원을 배우기 전에

**이전 단원 (07) 복습: Conv2d의 파라미터(stride, padding 등)를 이해하셨나요?**

## 이 단원 다음에는

**다음 단원 (09): 레이어를 배웠으니, 실제 프로젝트에서 필요한 데이터 라벨링을 배워봅시다.**

## 왜 이 단원이 필요한가?

신경망은 다양한 레이어들의 조합으로 이루어집니다. 각 레이어가 하는 역할을 알아야 모델을 설계할 수 있습니다.

- **Pooling**: 다운샘플링, 과적합 방지
- **Activation**: 비선형성 추가 (필수!)
- **Dropout, BatchNorm**: 학습 안정화
- **nn.Module**: 모델 구조화

이 레이어들을 조합해 다양한 아키텍처를 만들 수 있습니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 수학 기호

#### Pooling 관련
- $H_{in}$, $W_{in}$: 입력 특징맵의 높이, 너비
- $K$: 풀링 커널 크기 (예: $K = 2$이면 2×2 풀링)
- $S$: 풀링 스트라이드 (예: $S = 2$이면 2픽셀씩 이동)
- $H_{out}$, $W_{out}$: 출력 특징맵의 높이, 너비
- $x_{i,j}$: 입력 특징맵의 $(i, j)$ 위치의 값
- $y_{i,j}$: 출력 특징맵의 $(i, j)$ 위치의 값

#### Activation 관련
- $x$: 활성화 함수의 입력
- $\sigma(x)$: 활성화 함수의 출력
- $\text{ReLU}(x) = \max(0, x)$: ReLU 함수
- $\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$: Sigmoid 함수
- $\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$: Softmax 함수

#### Dropout 관련
- $p$: Dropout 확률 (예: $p = 0.5$이면 50% 비활성화)
- $m$: 마스크 (mask), 각 뉴런의 활성화 여부를 나타내는 이진 벡터
- $r \sim \text{Bernoulli}(1-p)$: 베르누이 분포에서 샘플링한 랜덤 변수

#### BatchNorm 관련
- $\mu_B$: 배치 평균 (batch mean)
- $\sigma_B^2$: 배치 분산 (batch variance)
- $\epsilon$: 수치 안정성을 위한 작은 상수 (예: $10^{-5}$)
- $\gamma$: 학습 가능한 스케일 파라미터 (scale parameter)
- $\beta$: 학습 가능한 시프트 파라미터 (shift parameter)
- $\hat{x}$: 정규화된 입력 (normalized input)

### PyTorch 코드에서의 변수

- `input`: 입력 텐서
- `output`: 출력 텐서
- `pool`: Pooling 레이어 객체 (예: `nn.MaxPool2d`)
- `activation`: 활성화 함수 (예: `nn.ReLU`, `F.relu`)
- `dropout`: Dropout 레이어 객체 (예: `nn.Dropout`)
- `batchnorm`: BatchNorm 레이어 객체 (예: `nn.BatchNorm2d`)
- `model`: `nn.Module`을 상속받은 모델 클래스

### 용어 정리

- **풀링(Pooling)**: 특징맵 크기를 줄이는 연산 (다운샘플링)
- **활성화 함수(Activation Function)**: 비선형성을 추가하는 함수
- **드롭아웃(Dropout)**: 학습 시 랜덤하게 일부 뉴런을 비활성화하여 과적합 방지
- **배치 정규화(Batch Normalization)**: 배치 내 통계량으로 입력을 정규화하여 학습 안정화

---

## Step 1: Pooling Layer (풀링 레이어)

### Step 1.1: 왜 Pooling이 필요한가?

#### 문제 상황

**Conv Layer를 거친 후의 문제**:
1. **특징맵 크기 증가**: 여러 Conv Layer를 거치면 특징맵이 커질 수 있음
2. **파라미터 수 증가**: 다음 레이어의 파라미터가 입력 크기에 비례하여 증가
3. **계산량 증가**: 큰 특징맵은 계산 비용이 큼
4. **과적합 위험**: 큰 모델은 학습 데이터에 과적합되기 쉬움

**예시**:
- Conv Layer 출력: $(B, 64, 224, 224)$
- 다음 Linear Layer: $64 \times 224 \times 224 = 3,211,264$ 입력
- Linear Layer 파라미터: $3,211,264 \times 1000 = 3,211,264,000$ (약 32억 개!)

#### 해결책: Pooling

**Pooling의 목적**:
1. **다운샘플링**: 특징맵 크기를 줄여 파라미터 수와 계산량 감소
2. **과적합 방지**: 모델 크기 감소로 과적합 위험 감소
3. **변환 불변성**: 작은 위치 변화에 강건함 (MaxPool의 경우)

**효과**:
- 특징맵 크기를 절반으로 줄이면 파라미터 수가 1/4로 감소
- 계산량도 크게 감소

### Step 1.2: 수학적 정의

#### MaxPool

**MaxPool2d**는 영역 내 최댓값을 선택합니다:

$$y_{i,j} = \max_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}$$

여기서:
- $\mathcal{R}_{i,j}$: $(i, j)$ 위치의 풀링 영역 (예: 2×2 영역)
- $x_{u,v}$: 입력 특징맵의 $(u, v)$ 위치의 값
- $y_{i,j}$: 출력 특징맵의 $(i, j)$ 위치의 값

**의미**: 풀링 영역 내에서 가장 큰 값을 선택합니다.

#### AvgPool

**AvgPool2d**는 영역 내 평균을 계산합니다:

$$y_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}$$

여기서:
- $|\mathcal{R}_{i,j}|$: 풀링 영역의 크기 (예: 2×2 = 4)

**의미**: 풀링 영역 내 값들의 평균을 계산합니다.

#### 출력 크기 계산

**Pooling의 출력 크기 공식**:

$$H_{out} = \left\lfloor \frac{H_{in} - K}{S} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} - K}{S} \right\rfloor + 1$$

여기서:
- $K$: 풀링 커널 크기
- $S$: 풀링 스트라이드

**예시**: 입력 28×28, kernel=2, stride=2
$$H_{out} = \left\lfloor \frac{28 - 2}{2} \right\rfloor + 1 = 14$$

### Step 1.3: PyTorch 구현

#### MaxPool2d 기본 사용법

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 입력 텐서 생성: 배치 1, 채널 16, 28×28
input_tensor = torch.randn(1, 16, 28, 28)
print(f"입력 shape: {input_tensor.shape}")

# 방법 1: nn.MaxPool2d 사용
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
output1 = maxpool(input_tensor)
print(f"MaxPool2d 출력: {output1.shape}")  # (1, 16, 14, 14)

# 방법 2: F.max_pool2d 사용 (함수형)
output2 = F.max_pool2d(input_tensor, kernel_size=2, stride=2)
print(f"F.max_pool2d 출력: {output2.shape}")  # (1, 16, 14, 14)

# 결과 일치 확인
print(f"결과 일치: {torch.allclose(output1, output2)}")  # True
```

#### AvgPool2d 사용법

```python
# AvgPool2d 사용
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg = avgpool(input_tensor)
print(f"AvgPool2d 출력: {output_avg.shape}")  # (1, 16, 14, 14)

# MaxPool과 AvgPool 비교
print(f"MaxPool 출력 범위: [{output1.min():.2f}, {output1.max():.2f}]")
print(f"AvgPool 출력 범위: [{output_avg.min():.2f}, {output_avg.max():.2f}]")
```

#### MaxPool vs AvgPool 비교 예시

```python
# 구체적 예시로 비교
# 작은 입력 텐서 생성
small_input = torch.tensor([[[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
]]])  # (1, 1, 4, 4)

print("입력:")
print(small_input[0, 0])

# MaxPool: 최댓값 선택
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
max_output = maxpool(small_input)
print("\nMaxPool 출력 (2×2 영역의 최댓값):")
print(max_output[0, 0])
# 출력:
# [[ 6.  8.]
#  [14. 16.]]

# AvgPool: 평균 계산
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_output = avgpool(small_input)
print("\nAvgPool 출력 (2×2 영역의 평균):")
print(avg_output[0, 0])
# 출력:
# [[ 3.5  5.5]
#  [11.5 13.5]]
```

#### MaxPool의 특징

**MaxPool Layer는 weight가 없습니다**:

```python
# MaxPool은 파라미터가 없음
maxpool = nn.MaxPool2d(2, 2)
params = list(maxpool.parameters())
print(f"파라미터 수: {len(params)}")  # 0

# 따라서 학습 가능한 파라미터가 없어 바로 numpy() 변환 가능
# (하지만 MaxPool은 가중치가 없으므로 의미 없음)
output = maxpool(input_tensor)
print(f"출력 shape: {output.shape}")
```

**의미**: MaxPool은 학습 가능한 파라미터가 없는 연산입니다. 단순히 최댓값을 선택하는 연산일 뿐입니다.

### Step 1.4: 구체적 예시

#### 출력 크기 계산 예시

```python
import torch
import torch.nn as nn

# 입력: 배치 1, 채널 16, 28×28
input_tensor = torch.randn(1, 16, 28, 28)
print(f"입력 shape: {input_tensor.shape}")

# MaxPool: kernel=2, stride=2
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
output = maxpool(input_tensor)
print(f"출력 shape: {output.shape}")  # (1, 16, 14, 14)

# 수동 계산 검증
H_in = 28
K = 2
S = 2
H_out = ((H_in - K) // S) + 1
print(f"수동 계산: {H_out}")  # 14
print(f"PyTorch 출력: {output.shape[2]}")  # 14
print(f"일치 여부: {H_out == output.shape[2]}")  # True
```

---

## Step 2: 비선형 활성화 함수 (Non-linear Activations)

### Step 2.1: 왜 비선형 활성화 함수가 필요한가?

#### 문제 상황: 선형 레이어만으로는 부족

**선형 레이어만으로 구성된 신경망**:

$$y = W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$$

여기서 $W' = W_2 W_1$, $b' = W_2 b_1 + b_2$입니다.

**문제**: 여러 선형 레이어를 쌓아도 **하나의 선형 레이어와 동일**합니다!

**증명**:
- 선형 변환의 합성은 여전히 선형 변환입니다
- $T_1(x) = W_1 x + b_1$, $T_2(x) = W_2 x + b_2$일 때
- $T_2(T_1(x)) = W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$

**의미**: 선형 레이어만으로는 **복잡한 비선형 패턴을 학습할 수 없습니다**.

#### 해결책: 비선형 활성화 함수

**비선형 활성화 함수를 추가**하면:

$$y = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)$$

여기서 $\sigma$는 비선형 활성화 함수입니다.

**효과**:
- 여러 레이어를 쌓아도 **비선형 변환**이 가능합니다
- 복잡한 패턴을 학습할 수 있습니다
- 신경망의 표현력이 크게 증가합니다

**비유**:
- 선형 레이어만: 직선으로만 표현 가능
- 비선형 활성화 함수 추가: 곡선, 복잡한 형태 표현 가능

### Step 2.2: ReLU (Rectified Linear Unit)

#### 수학적 정의

**ReLU 함수**:

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**그래프**:
- $x > 0$: $y = x$ (직선)
- $x \leq 0$: $y = 0$ (수평선)

**도함수**:

$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

#### 왜 ReLU를 사용하는가?

**장점**:
1. **계산 효율성**: $\max(0, x)$ 연산이 매우 빠름
2. **기울기 소실 완화**: $x > 0$일 때 기울기가 1로 유지되어 역전파가 잘 됨
3. **희소성**: 음수 입력을 0으로 만들어 희소 표현 생성

**단점**:
1. **죽은 ReLU 문제**: 입력이 항상 음수이면 기울기가 0이 되어 학습이 안 됨
2. **비대칭성**: 음수 영역에서 정보 손실

#### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 입력 텐서 생성 (음수와 양수 포함)
input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"입력: {input_tensor}")

# 방법 1: F.relu 사용 (함수형)
output1 = F.relu(input_tensor)
print(f"F.relu 출력: {output1}")  # tensor([0., 0., 0., 1., 2.])

# 방법 2: nn.ReLU 사용 (레이어)
relu = nn.ReLU()
output2 = relu(input_tensor)
print(f"nn.ReLU 출력: {output2}")  # tensor([0., 0., 0., 1., 2.])

# 결과 일치 확인
print(f"결과 일치: {torch.allclose(output1, output2)}")  # True
```

#### ReLU 동작 예시

```python
# 2D 텐서에 ReLU 적용
input_2d = torch.randn(3, 4)
print("입력 (일부 음수 포함):")
print(input_2d)

# ReLU 적용
output_relu = F.relu(input_2d)
print("\nReLU 출력 (음수는 0으로):")
print(output_relu)

# 음수 개수 확인
print(f"\n입력에서 음수 개수: {(input_2d < 0).sum().item()}")
print(f"출력에서 0의 개수: {(output_relu == 0).sum().item()}")
```

### Step 2.3: Sigmoid

#### 수학적 정의

**Sigmoid 함수**:

$$\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$$

**특징**:
- 출력 범위: $(0, 1)$
- 단조 증가 함수 (monotonic increasing)
- $x = 0$일 때 $\text{Sigmoid}(0) = 0.5$

**도함수**:

$$\frac{d}{dx}\text{Sigmoid}(x) = \text{Sigmoid}(x)(1 - \text{Sigmoid}(x))$$

**의미**: Sigmoid의 도함수는 최대값이 0.25 ($x = 0$일 때)입니다.

#### 왜 Sigmoid를 사용하는가?

**장점**:
1. **확률 해석**: 출력이 0~1 범위이므로 확률로 해석 가능
2. **이진 분류**: 출력을 확률로 해석하여 이진 분류에 사용

**단점**:
1. **기울기 소실**: 입력이 크거나 작을 때 기울기가 거의 0에 가까워짐
2. **비대칭성**: 출력이 0에 가까워지기 쉬움
3. **계산 비용**: 지수 함수 계산이 비용이 큼

#### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 입력 텐서 생성
input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"입력: {input_tensor}")

# 방법 1: F.sigmoid 사용 (함수형)
output1 = F.sigmoid(input_tensor)
print(f"F.sigmoid 출력: {output1}")
# 출력: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

# 방법 2: nn.Sigmoid 사용 (레이어)
sigmoid = nn.Sigmoid()
output2 = sigmoid(input_tensor)
print(f"nn.Sigmoid 출력: {output2}")

# 결과 일치 확인
print(f"결과 일치: {torch.allclose(output1, output2)}")  # True
```

### Step 2.4: Softmax

#### 수학적 정의

**Softmax 함수**:

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

여기서:
- $x_i$: $i$번째 클래스의 로짓(logit) 값
- $K$: 클래스 개수
- 분모: 모든 클래스의 지수 합

**특징**:
- 출력 범위: $(0, 1)$
- 모든 출력의 합이 1: $\sum_{i=1}^{K} \text{Softmax}(x_i) = 1$
- 확률 분포로 해석 가능

**의미**: 각 클래스에 대한 확률을 계산합니다.

#### 왜 Softmax를 사용하는가?

**용도**:
1. **다중 분류**: 여러 클래스 중 하나를 선택하는 문제
2. **확률 계산**: 각 클래스에 대한 확률을 계산
3. **손실 함수와의 연결**: Cross-Entropy Loss와 함께 사용

**예시**: 10개 클래스 분류 문제
- 입력: 10개 클래스에 대한 점수 (로짓)
- 출력: 10개 클래스에 대한 확률 (합이 1)

#### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 입력: 배치 4, 클래스 10 (로짓)
logits = torch.randn(4, 10)
print(f"입력 (logits) shape: {logits.shape}")
print(f"입력 값 범위: [{logits.min():.2f}, {logits.max():.2f}]")

# 방법 1: F.softmax 사용 (함수형)
# dim=1: 각 샘플(행)에 대해 softmax 적용
probs1 = F.softmax(logits, dim=1)
print(f"\nF.softmax 출력 shape: {probs1.shape}")
print(f"각 샘플의 합: {probs1.sum(dim=1)}")  # 모두 1.0에 가까워야 함

# 방법 2: nn.Softmax 사용 (레이어)
softmax = nn.Softmax(dim=1)
probs2 = softmax(logits)
print(f"\nnn.Softmax 출력 shape: {probs2.shape}")
print(f"각 샘플의 합: {probs2.sum(dim=1)}")  # 모두 1.0에 가까워야 함

# 결과 일치 확인
print(f"\n결과 일치: {torch.allclose(probs1, probs2)}")  # True

# 실제 예측 (가장 높은 확률의 클래스)
predictions = probs1.argmax(dim=1)
print(f"\n예측된 클래스: {predictions}")
```

#### Softmax 동작 예시

```python
# 구체적 예시
logits_example = torch.tensor([[1.0, 2.0, 3.0]])  # 3개 클래스
print(f"입력 로짓: {logits_example}")

# Softmax 적용
probs_example = F.softmax(logits_example, dim=1)
print(f"출력 확률: {probs_example}")
print(f"확률 합: {probs_example.sum().item()}")  # 1.0

# 수동 계산 검증
# exp(1) = 2.718, exp(2) = 7.389, exp(3) = 20.086
# 합 = 2.718 + 7.389 + 20.086 = 30.193
# Softmax(1.0) = 2.718 / 30.193 ≈ 0.090
# Softmax(2.0) = 7.389 / 30.193 ≈ 0.245
# Softmax(3.0) = 20.086 / 30.193 ≈ 0.665
print(f"\n수동 계산:")
print(f"  exp 값: {torch.exp(logits_example)}")
print(f"  합: {torch.exp(logits_example).sum().item():.3f}")
print(f"  Softmax: {torch.exp(logits_example) / torch.exp(logits_example).sum()}")
```

### Step 2.5: 활성화 함수 비교

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 입력 범위 생성
x = torch.linspace(-5, 5, 100)

# 각 활성화 함수 적용
y_relu = F.relu(x)
y_sigmoid = torch.sigmoid(x)
y_tanh = torch.tanh(x)

# 시각화 (옵션)
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.plot(x.numpy(), y_relu.numpy())
# plt.title('ReLU')
# plt.grid(True)
# plt.subplot(1, 3, 2)
# plt.plot(x.numpy(), y_sigmoid.numpy())
# plt.title('Sigmoid')
# plt.grid(True)
# plt.subplot(1, 3, 3)
# plt.plot(x.numpy(), y_tanh.numpy())
# plt.title('Tanh')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

print("활성화 함수 출력 범위:")
print(f"ReLU: [{y_relu.min():.2f}, {y_relu.max():.2f}]")
print(f"Sigmoid: [{y_sigmoid.min():.2f}, {y_sigmoid.max():.2f}]")
print(f"Tanh: [{y_tanh.min():.2f}, {y_tanh.max():.2f}]")
```

---

## Step 3: 정규화 및 규제 레이어

### Step 3.1: Dropout

#### Step 3.1.1: 왜 Dropout이 필요한가?

**문제 상황: 과적합 (Overfitting)**

**과적합이란?**:
- 모델이 학습 데이터에 너무 맞춰져서 새로운 데이터에 대해 성능이 낮은 현상
- 학습 데이터의 노이즈까지 학습하여 일반화 성능이 떨어짐

**과적합의 원인**:
1. **모델이 너무 복잡**: 파라미터가 너무 많아 학습 데이터를 외움
2. **데이터가 부족**: 학습 데이터가 적어 일반화하기 어려움
3. **뉴런 간 의존성**: 특정 뉴런들이 서로 강하게 의존하여 함께 작동

**해결책: Dropout**

**Dropout의 아이디어**:
- 학습 시 랜덤하게 일부 뉴런을 비활성화 (0으로 설정)
- 각 뉴런이 다른 뉴런에 의존하지 않고 독립적으로 학습
- 앙상블 효과: 여러 작은 네트워크의 평균

#### Step 3.1.2: 수학적 정의

**Dropout 연산**:

학습 시 (training mode):
$$y = \frac{1}{1-p} \cdot m \odot x$$

여기서:
- $m \sim \text{Bernoulli}(1-p)$: 마스크 (각 뉴런을 비활성화할지 결정)
- $p$: Dropout 확률 (예: $p = 0.5$이면 50% 비활성화)
- $\odot$: 요소별 곱 (element-wise multiplication)
- $\frac{1}{1-p}$: 스케일링 팩터 (평균을 유지하기 위해)

평가 시 (eval mode):
$$y = x$$

**의미**:
- 학습 시: 랜덤하게 뉴런을 비활성화하되, 평균을 유지하기 위해 $\frac{1}{1-p}$로 스케일링
- 평가 시: 모든 뉴런을 활성화 (평균 예측)

#### Step 3.1.3: PyTorch 구현

```python
import torch
import torch.nn as nn

# 입력 텐서 생성
input_tensor = torch.randn(10, 20)
print(f"입력 shape: {input_tensor.shape}")

# Dropout 레이어 생성 (확률 0.5)
dropout = nn.Dropout(p=0.5)

# 학습 모드 (training mode)
dropout.train()
output_train = dropout(input_tensor)
print(f"\nDropout 출력 (학습 모드):")
print(f"  입력 평균: {input_tensor.mean():.4f}")
print(f"  출력 평균: {output_train.mean():.4f}")
print(f"  0의 개수: {(output_train == 0).sum().item()} / {output_train.numel()}")

# 평가 모드 (eval mode)
dropout.eval()
output_eval = dropout(input_tensor)
print(f"\nDropout 출력 (평가 모드):")
print(f"  0의 개수: {(output_eval == 0).sum().item()} / {output_eval.numel()}")
print(f"모든 값 활성화: {torch.allclose(output_eval, input_tensor)}")  # True
```

#### Step 3.1.4: Dropout 효과 시각화

```python
# Dropout 효과 비교
input_example = torch.ones(1, 10)
print(f"입력: {input_example}")

dropout = nn.Dropout(p=0.5)

# 학습 모드에서 여러 번 실행
dropout.train()
outputs = []
for i in range(5):
    output = dropout(input_example)
    outputs.append(output)
    print(f"실행 {i+1}: {output}")

# 평가 모드
dropout.eval()
output_eval = dropout(input_example)
print(f"\n평가 모드: {output_eval}")
```

### Step 3.2: Batch Normalization

#### Step 3.2.1: 왜 BatchNorm이 필요한가?

**문제 상황: 내부 공변량 이동 (Internal Covariate Shift)**

**내부 공변량 이동이란?**:
- 각 레이어의 입력 분포가 학습 중에 계속 변경되는 현상
- 이전 레이어의 가중치가 업데이트되면 다음 레이어의 입력 분포가 변경됨
- 학습이 불안정해지고 학습 속도가 느려짐

**구체적 문제**:
1. **학습 불안정**: 입력 분포가 계속 변경되어 학습이 어려움
2. **학습률 제한**: 작은 학습률을 사용해야 함 (큰 학습률 사용 시 발산)
3. **초기화 의존성**: 가중치 초기화에 매우 민감

**해결책: Batch Normalization**

**BatchNorm의 아이디어**:
- 각 배치의 통계량(평균, 분산)으로 입력을 정규화
- 학습 안정화 및 학습 속도 향상
- 내부 파라미터($\gamma$, $\beta$)로 스케일/시프트 조정

#### Step 3.2.2: 수학적 정의

**BatchNorm 연산**:

1. **배치 평균 계산**:
$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

2. **배치 분산 계산**:
$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

3. **정규화**:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

4. **스케일 및 시프트**:
$$y_i = \gamma \hat{x}_i + \beta$$

여기서:
- $m$: 배치 크기
- $\epsilon$: 수치 안정성을 위한 작은 상수 (예: $10^{-5}$)
- $\gamma$: 학습 가능한 스케일 파라미터 (초기값: 1)
- $\beta$: 학습 가능한 시프트 파라미터 (초기값: 0)

**의미**:
- 정규화: 입력을 평균 0, 분산 1로 변환
- 스케일/시프트: $\gamma$와 $\beta$로 원하는 분포로 변환

#### Step 3.2.3: PyTorch 구현

```python
import torch
import torch.nn as nn

# BatchNorm1d: Linear 레이어와 함께 사용
x_1d = torch.randn(32, 128)  # 배치 32, 특징 128
print(f"입력 shape: {x_1d.shape}")
print(f"입력 평균: {x_1d.mean():.4f}, 표준편차: {x_1d.std():.4f}")

bn1d = nn.BatchNorm1d(num_features=128)
output_1d = bn1d(x_1d)
print(f"\nBatchNorm1d 출력:")
print(f"  출력 shape: {output_1d.shape}")
print(f"  출력 평균: {output_1d.mean():.6f}, 표준편차: {output_1d.std():.6f}")

# BatchNorm2d: Conv 레이어와 함께 사용 (이미지)
x_2d = torch.randn(8, 16, 32, 32)  # 배치 8, 채널 16, 32×32
print(f"\n입력 shape: {x_2d.shape}")

bn2d = nn.BatchNorm2d(num_features=16)
output_2d = bn2d(x_2d)
print(f"BatchNorm2d 출력:")
print(f"  출력 shape: {output_2d.shape}")

# BatchNorm의 학습 파라미터
print(f"\n학습 파라미터:")
print(f"  가중치 (gamma): {bn2d.weight.shape}")  # (16,)
print(f"  편향 (beta): {bn2d.bias.shape}")  # (16,)
print(f"  gamma 초기값: {bn2d.weight.data[:5]}")  # 모두 1에 가까움
print(f"  beta 초기값: {bn2d.bias.data[:5]}")  # 모두 0에 가까움
```

#### Step 3.2.4: BatchNorm 동작 확인

```python
# BatchNorm이 정규화를 수행하는지 확인
# 큰 값으로 구성된 입력 생성
x_large = torch.ones(32, 128) * 100.0  # 평균 100
print(f"입력 평균: {x_large.mean():.2f}, 표준편차: {x_large.std():.2f}")

bn = nn.BatchNorm1d(128)
bn.train()  # 학습 모드
output = bn(x_large)
print(f"출력 평균: {output.mean():.6f}, 표준편차: {output.std():.6f}")
# 출력 평균은 0에 가까워지고, 표준편차는 1에 가까워짐
```

---

## Step 4: 모델 정의 (nn.Module)

### Step 4.1: 왜 nn.Module이 필요한가?

**문제 상황: 모델 구조화 필요**

**신경망 모델의 구성 요소**:
- 여러 레이어 (Linear, Conv, Pooling, Activation 등)
- 각 레이어의 파라미터 (가중치, 편향)
- Forward pass 로직 (데이터가 레이어를 거치는 순서)
- Backward pass (자동 미분을 통한 기울기 계산)

**문제**: 레이어와 로직을 어떻게 구조화할까?

**해결책: nn.Module**

**nn.Module의 역할**:
- 모든 레이어를 하나의 클래스로 묶어 구조화
- 파라미터 관리 (가중치, 편향)
- Forward pass 정의
- 자동 미분 지원

### Step 4.2: nn.Module 기본 구조

#### 수학적 정의

**모델 $M$은 다음과 같이 정의됩니다**:

$$M(x) = f_n(f_{n-1}(\cdots f_2(f_1(x))\cdots))$$

여기서:
- $f_1, f_2, \ldots, f_n$: 각 레이어 (Linear, Conv, Activation 등)
- $x$: 입력 데이터
- $M(x)$: 모델의 출력

**의미**: 여러 레이어를 순차적으로 적용하여 최종 출력을 계산합니다.

#### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    간단한 CNN 모델
    
    구조:
    - Conv2d → ReLU → MaxPool2d
    - Conv2d → ReLU → MaxPool2d
    - Flatten
    - Linear → ReLU
    - Linear (출력)
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Conv 레이어들
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Pooling 레이어
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Linear 레이어들
        # Conv → Pool → Conv → Pool 후 크기: 32×32 → 16×16 → 8×8
        # 32 채널 × 8 × 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)  # (B, 3, 32, 32) → (B, 16, 32, 32)
        x = F.relu(x)
        x = self.pool(x)   # (B, 16, 32, 32) → (B, 16, 16, 16)
        
        # Conv Block 2
        x = self.conv2(x)  # (B, 16, 16, 16) → (B, 32, 16, 16)
        x = F.relu(x)
        x = self.pool(x)   # (B, 32, 16, 16) → (B, 32, 8, 8)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 32, 8, 8) → (B, 2048)
        
        # Linear layers
        x = self.fc1(x)    # (B, 2048) → (B, 128)
        x = F.relu(x)
        x = self.fc2(x)    # (B, 128) → (B, num_classes)
        
        return x

# 모델 생성
model = SimpleCNN(num_classes=10)
print(f"모델: {model}")

# 입력 생성 및 Forward pass
input_tensor = torch.randn(4, 3, 32, 32)  # 배치 4, 채널 3, 32×32
print(f"\n입력 shape: {input_tensor.shape}")

output = model(input_tensor)
print(f"출력 shape: {output.shape}")  # (4, 10)

# 파라미터 수 확인
total_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터 수: {total_params:,}")
```

### Step 4.3: 모델 구조 확인

```python
# 모델의 각 레이어 확인
print("모델 구조:")
for name, module in model.named_children():
    print(f"  {name}: {module}")

# 파라미터 확인
print("\n파라미터:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

### Step 4.4: 레이어 조합 예시 (Conv → ReLU → MaxPool)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv → ReLU → MaxPool 블록 만들기
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

# 사용 예시
block = ConvBlock(3, 16)
input_tensor = torch.randn(1, 3, 28, 28)
print(f"입력 shape: {input_tensor.shape}")

output = block(input_tensor)
print(f"출력 shape: {output.shape}")  # (1, 16, 14, 14)
```

---

## 연습 문제

### 문제 1: Pooling 비교
MaxPool과 AvgPool의 출력 차이를 설명하고 예시로 보여주세요.

```python
import torch
import torch.nn as nn

# 입력 생성
input_tensor = torch.randn(1, 1, 4, 4)
print("입력:")
print(input_tensor[0, 0])

# MaxPool
maxpool = nn.MaxPool2d(2, 2)
output_max = maxpool(input_tensor)
print("\nMaxPool 출력:")
print(output_max[0, 0])

# AvgPool
avgpool = nn.AvgPool2d(2, 2)
output_avg = avgpool(input_tensor)
print("\nAvgPool 출력:")
print(output_avg[0, 0])

# 차이 설명
print("\n차이:")
print(f"MaxPool: 영역 내 최댓값 선택")
print(f"AvgPool: 영역 내 평균 계산")
```

### 문제 2: ReLU 동작
음수 값을 가진 텐서에 ReLU를 적용하고 변화를 관찰하세요.

```python
import torch
import torch.nn.functional as F

# 음수와 양수를 포함한 입력
input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"입력: {input_tensor}")

# ReLU 적용
output = F.relu(input_tensor)
print(f"출력: {output}")

# 변화 관찰
print(f"\n음수는 0으로, 양수는 그대로 유지됨")
print(f"음수 개수: {(input_tensor < 0).sum().item()}")
print(f"출력에서 0의 개수: {(output == 0).sum().item()}")
```

### 문제 3: Sigmoid vs Softmax
Sigmoid와 Softmax의 차이를 설명하고 각각 어떤 경우에 사용하나요?

```python
import torch
import torch.nn.functional as F

# Sigmoid: 각 요소에 독립적으로 적용
input_sigmoid = torch.tensor([-1.0, 0.0, 1.0])
output_sigmoid = torch.sigmoid(input_sigmoid)
print(f"Sigmoid 입력: {input_sigmoid}")
print(f"Sigmoid 출력: {output_sigmoid}")
print(f"Sigmoid 합: {output_sigmoid.sum().item()}")  # 1이 아닐 수 있음

# Softmax: 모든 요소의 합이 1이 되도록 정규화
input_softmax = torch.tensor([[-1.0, 0.0, 1.0]])
output_softmax = F.softmax(input_softmax, dim=1)
print(f"\nSoftmax 입력: {input_softmax}")
print(f"Softmax 출력: {output_softmax}")
print(f"Softmax 합: {output_softmax.sum().item()}")  # 1.0

print("\n차이:")
print("Sigmoid: 이진 분류, 각 요소가 독립적인 확률")
print("Softmax: 다중 분류, 모든 요소의 합이 1인 확률 분포")
```

### 문제 4: 비선형성 필요성
ReLU 없이 Linear만으로 쌓은 모델의 한계를 설명하세요.

```python
import torch
import torch.nn as nn

# Linear만으로 구성된 모델
class LinearOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)  # ReLU 없음!
        return x

# ReLU를 포함한 모델
class NonlinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)  # ReLU 추가!
        x = self.linear2(x)
        return x

# 테스트
input_tensor = torch.randn(1, 10)

model_linear = LinearOnlyModel()
model_nonlinear = NonlinearModel()

output_linear = model_linear(input_tensor)
output_nonlinear = model_nonlinear(input_tensor)

print("Linear만:")
print(f"  출력 범위: [{output_linear.min():.2f}, {output_linear.max():.2f}]")

print("\nReLU 포함:")
print(f"  출력 범위: [{output_nonlinear.min():.2f}, {output_nonlinear.max():.2f}]")

print("\n한계:")
print("Linear만으로는 여러 레이어를 쌓아도 하나의 Linear와 동일함")
print("복잡한 비선형 패턴을 학습할 수 없음")
```

### 문제 5: Layer 조합
Conv → ReLU → MaxPool 블록을 만들고 출력을 확인하세요.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv → ReLU → MaxPool 블록
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

# 사용
block = ConvBlock(3, 16)
input_tensor = torch.randn(1, 3, 28, 28)
print(f"입력 shape: {input_tensor.shape}")

output = block(input_tensor)
print(f"출력 shape: {output.shape}")  # (1, 16, 14, 14)
```

---

## 핵심 요약

### 1. Pooling Layer
- **MaxPool**: 지역 최댓값 선택 (가장 많이 사용)
- **AvgPool**: 지역 평균 계산
- **역할**: 다운샘플링 + 과적합 방지
- **파라미터**: 없음 (학습 가능한 파라미터 없음)
- **출력 크기**: $H_{out} = \lfloor (H_{in} - K) / S \rfloor + 1$

### 2. 비선형 활성화 함수
- **ReLU**: $\max(0, x)$ - 가장 많이 사용, 계산 효율적
- **Sigmoid**: $\frac{1}{1+e^{-x}}$ - 이진 분류, 확률 해석
- **Softmax**: $\frac{e^{x_i}}{\sum_j e^{x_j}}$ - 다중 분류, 확률 분포
- **비선형성의 필요성**: 선형 레이어만으로는 복잡한 패턴 학습 불가

### 3. 정규화 및 규제 레이어
- **Dropout**: 과적합 방지, 학습 시 랜덤 비활성화
- **BatchNorm**: 학습 안정화, 평균/분산 정규화
- **학습/평가 모드**: Dropout과 BatchNorm은 모드에 따라 다르게 동작

### 4. 모델 정의 (nn.Module)
- **구조**: `__init__()`에서 레이어 정의, `forward()`에서 연산 정의
- **레이어 조합**: Conv → ReLU → MaxPool 블록이 기본 단위
- **파라미터 관리**: `model.parameters()`로 모든 파라미터 접근

**축하합니다!** PyTorch의 기본 구조에 대한 모든 단원을 완료했습니다!

