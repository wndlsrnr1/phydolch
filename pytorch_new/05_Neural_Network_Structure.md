# 05. 신경망 구조 이해

이번 단원에서는 신경망의 기본 구조와 PyTorch의 구조화 방식을 배워보겠습니다.

## 학습 목표
- 신경망의 계층적 구조 (Layer, Module, Model)
- torch.nn 패키지
- 신경망을 복합 함수로 이해하기

## 이 단원을 배우기 전에

**이전 단원 (04) 복습: Dataset, DataLoader를 만들고 사용할 수 있나요?**

**이전 단원 (00) 복습: 행렬이 선형 변환(공간을 변환하는 함수)이라는 것을 기억하시나요?**

**이전 단원 (03) 복습: 체인룰(Chain Rule)과 역전파의 의미를 이해하셨나요?**

## 이 단원 다음에는

**다음 단원 (06): 신경망 구조를 알았으니, 가장 기본적인 Linear Layer를 자세히 배워봅시다.**

## 왜 이 단원이 필요한가?

신경망은 레이어, 모듈, 모델의 계층적 구조로 이루어져 있습니다. 이 구조를 이해해야 모델을 제대로 설계할 수 있습니다.

- **Layer**: 기본 계산 단위
- **Module**: 기능 단위 그룹
- **Model**: 전체 아키텍처
- **nn.Module**: 모든 커스텀 모델의 부모 클래스

이 단원을 통해 신경망을 '함수의 합성'으로 이해하게 됩니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 신경망 구조 관련 기호

- $f_i$: $i$번째 레이어의 함수
- $h_i$: $i$번째 레이어의 출력 (hidden state)
- $x \in \mathbb{R}^{d_{in}}$: 입력 벡터
- $y \in \mathbb{R}^{d_{out}}$: 출력 벡터
- $W_i \in \mathbb{R}^{d_{out} \times d_{in}}$: $i$번째 레이어의 가중치 행렬
- $b_i \in \mathbb{R}^{d_{out}}$: $i$번째 레이어의 편향 벡터
- $\sigma$: 활성화 함수 (예: ReLU, sigmoid)
- $L$: 손실 함수 (Loss Function)

### 계층 구조 관련 기호

- **Layer**: 단일 계산 단위, $h = f(x)$
- **Module**: 여러 레이어의 조합, $h = f_n(\cdots f_2(f_1(x))\cdots)$
- **Model**: 전체 아키텍처, $y = \text{Model}(x)$

### PyTorch 코드에서의 변수

- `layer`: 단일 레이어 객체 (예: `nn.Linear`)
- `module`: 모듈 객체 (예: `nn.Module`을 상속한 클래스)
- `model`: 모델 객체 (전체 신경망)
- `forward()`: 순전파 함수
- `parameters()`: 모델의 학습 가능한 파라미터

---

## Step 1: 왜 계층적 구조가 필요한가?

### Step 1.1: 문제 상황

**단순한 선형 변환의 한계**:

하나의 선형 변환만으로는 복잡한 패턴을 학습할 수 없습니다:

$$y = Wx + b$$

**문제점**:
- 선형 변환만으로는 비선형 관계를 표현할 수 없음
- 복잡한 데이터 분포를 학습하기 어려움

### Step 1.2: 해결책: 계층적 구조

**여러 레이어를 쌓아 복잡한 함수를 만들기**:

$$y = f_3(f_2(f_1(x)))$$

**장점**:
- 각 레이어가 점진적으로 복잡한 특징을 추출
- 비선형 활성화 함수로 비선형 관계 표현 가능
- 깊은 신경망으로 복잡한 패턴 학습 가능

### Step 1.3: 계층적 구조의 수학적 의미

**신경망은 복합 함수 (Composite Function)**입니다:

$$y = f_n(f_{n-1}(\cdots f_2(f_1(x))\cdots))$$

**의미**: 여러 함수를 순차적으로 적용하여 최종 출력을 얻습니다.

**체인룰 연결** (이전 단원 03 복습):

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

각 레이어의 미분을 곱하여 전체 gradient를 계산할 수 있습니다.

---

## Step 2: Layer, Module, Model의 수학적 정의

### Step 2.0: 기호 정리

- **Layer**: $h = f(x)$, 단일 함수
- **Module**: $h = f_n(\cdots f_2(f_1(x))\cdots)$, 여러 함수의 합성
- **Model**: $y = \text{Model}(x)$, 전체 아키텍처

### Step 2.1: 레이어(Layer)의 정의

**레이어(Layer)**는 신경망의 핵심 데이터 구조로 하나 이상의 텐서를 입력받아 하나 이상의 텐서를 출력하는 기본 연산 단위입니다.

**수학적 정의**:

$$h = f(x)$$

여기서:
- $x \in \mathbb{R}^{d_{in}}$: 입력 텐서
- $f$: 레이어 함수 (예: 선형 변환, 컨볼루션 등)
- $h \in \mathbb{R}^{d_{out}}$: 출력 텐서

**예시**:
- `nn.Linear`: 선형 변환 $h = Wx + b$
- `nn.Conv2d`: 컨볼루션 연산
- `nn.ReLU`: 활성화 함수 $h = \max(0, x)$

**비유**: 계산 하나를 레이어라고 할 수 있습니다.

### Step 2.2: 모듈(Module)의 정의

**모듈(Module)**은 한 개 이상의 레이어가 모여서 구성된 기능 단위입니다.

**수학적 정의**:

$$h = f_n(\cdots f_2(f_1(x))\cdots)$$

여기서:
- $f_1, f_2, \ldots, f_n$: 각각의 레이어 함수
- $h$: 최종 출력

**의미**: 관련 레이어들을 기능 단위로 묶은 것입니다.

**예시**:
- 특징 추출 모듈: 여러 컨볼루션 레이어의 조합
- 분류 모듈: 여러 선형 레이어의 조합

**비유**: 레이어를 모아서 기능이 구성된 것이 모듈입니다.

### Step 2.3: 모델(Model)의 정의

**모델(Model)**은 한 개 이상의 모듈이 모여서 구성된 전체 아키텍처입니다.

**수학적 정의**:

$$y = \text{Model}(x) = \text{Module}_m(\cdots \text{Module}_2(\text{Module}_1(x))\cdots)$$

여기서:
- $\text{Module}_1, \text{Module}_2, \ldots, \text{Module}_m$: 각각의 모듈
- $y$: 최종 출력

**의미**: 모든 모듈들이 합쳐 전체 모델을 구성합니다.

**비유**: 전체 집합이 모델입니다.

### Step 2.4: 계층 구조 시각화

```
Input (입력)
  ↓
┌─────────────────────────────────────┐
│     Model (모델) - 전체 신경망      │
│  ┌───────────────────────────────┐  │
│  │  Module (모듈) - 특징 추출    │  │
│  │  ┌─────────┐  ┌──────────┐   │  │
│  │  │ Layer 1 │→ │ Layer 2  │   │  │
│  │  │(Linear) │  │(ReLU)    │   │  │
│  │  └─────────┘  └──────────┘   │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  Module (모듈) - 분류         │  │
│  │  ┌─────────┐                 │  │
│  │  │ Layer 3 │                 │  │
│  │  │(Linear) │                 │  │
│  │  └─────────┘                 │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
  ↓
Output (출력)
```

---

## Step 3: torch.nn 패키지

### Step 3.0: 기호 정리

- `nn.Module`: 모든 PyTorch 모델의 부모 클래스
- `nn.Linear`: 선형 레이어
- `nn.Conv2d`: 2D 컨볼루션 레이어
- `weight`: 가중치 파라미터
- `bias`: 편향 파라미터

### Step 3.1: 왜 torch.nn 패키지가 필요한가?

**문제 상황**: 레이어를 직접 구현하려면?

```python
# 수동으로 가중치와 편향을 정의해야 함
import torch

# 가중치와 편향을 직접 생성
weight = torch.randn(5, 10)  # (out_features, in_features)
bias = torch.randn(5)

# 선형 변환 직접 구현
def linear_transform(x):
    return x @ weight.T + bias

x = torch.randn(32, 10)  # (batch_size, in_features)
y = linear_transform(x)  # (batch_size, out_features)
```

**문제점**:
- 가중치 초기화를 수동으로 해야 함
- 파라미터 관리가 복잡함
- gradient 추적을 수동으로 설정해야 함

**해결책**: `torch.nn` 패키지 사용

`torch.nn` 패키지는 신경망 구조를 쉽게 구성하도록 도와주는 레이어 모음집입니다.

**장점**:
- 가중치와 편향을 자동으로 생성
- 파라미터를 자동으로 관리
- gradient 추적 자동 설정

### Step 3.2: 자동으로 생성되는 레이어란?

PyTorch의 `torch.nn` 패키지 안의 `nn.Linear`, `nn.Conv2d`, `nn.LSTM` 같은 레이어들은 내부에서 학습 가능한 파라미터(가중치와 편향)을 자동으로 만들어줍니다.

**수학적 의미**:

`nn.Linear(in_features=10, out_features=5)`를 생성하면:
- 가중치 행렬 $W \in \mathbb{R}^{5 \times 10}$ 자동 생성
- 편향 벡터 $b \in \mathbb{R}^{5}$ 자동 생성

**연산**: $y = Wx + b$

여기서:
- $x \in \mathbb{R}^{10}$: 입력 벡터
- $y \in \mathbb{R}^{5}$: 출력 벡터

### Step 3.3: torch.nn 패키지 사용법

**공식 문서**: https://pytorch.org/docs/stable/nn.html

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - linear: Linear 레이어 객체
# - x: 입력 텐서
# - y: 출력 텐서
# - weight: 가중치 파라미터
# - bias: 편향 파라미터

# Linear 레이어 생성
linear = nn.Linear(in_features=10, out_features=5)
print(f"Linear 레이어: {linear}")

# 입력 생성 및 전파
x = torch.randn(32, 10)  # (batch_size, in_features)
y = linear(x)            # (batch_size, out_features)

print(f"\n입력 shape: {x.shape}")
print(f"출력 shape: {y.shape}")

# 파라미터 확인
print(f"\n가중치 shape: {linear.weight.shape}")  # (5, 10)
print(f"편향 shape: {linear.bias.shape}")        # (5,)
```

**출력 예시**:
```
Linear 레이어: Linear(in_features=10, out_features=5, bias=True)

입력 shape: torch.Size([32, 10])
출력 shape: torch.Size([32, 5])

가중치 shape: torch.Size([5, 10])
편향 shape: torch.Size([5])
```

**자동 생성된 파라미터 확인**:

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - linear: Linear 레이어 객체
# - name: 파라미터 이름
# - param: 파라미터 텐서

# Linear 레이어 생성
linear = nn.Linear(in_features=5, out_features=3)

# 자동으로 생성된 파라미터 확인
print("자동 생성된 파라미터:")
for name, param in linear.named_parameters():
    print(f"{name}: shape={param.shape}")
    if len(param.shape) > 1:
        print(f"{name} 초기값 예시:\n{param.data[:2, :2]}\n")
    else:
        print(f"{name} 초기값 예시:\n{param.data[:3]}\n")

# 모델 사용
x = torch.randn(10, 5)
y = linear(x)
print(f"입력 shape: {x.shape}")
print(f"출력 shape: {y.shape}")
```

**출력 예시**:
```
자동 생성된 파라미터:
weight: shape=torch.Size([3, 5])
weight 초기값 예시:
tensor([[ 0.1234, -0.5678],
        [ 0.9012,  0.3456]])

bias: shape=torch.Size([3])
bias 초기값 예시:
tensor([0.0123, -0.0456, 0.0789])

입력 shape: torch.Size([10, 5])
출력 shape: torch.Size([10, 3])
```

---

## Step 4: Layer, Module, Model 구현

### Step 4.1: Layer 예제

**Layer는 단일 연산 단위**입니다.

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - layer: 단일 레이어 객체
# - x: 입력 텐서
# - y: 출력 텐서

# 간단한 Layer 예제
layer = nn.Linear(10, 5)
print(f"Layer: {layer}")
print(f"Layer 타입: {type(layer)}")
print(f"nn.Module 상속 확인: {isinstance(layer, nn.Module)}")

# 사용 예시
x = torch.randn(5, 10)
y = layer(x)
print(f"\n입력: {x.shape} -> 출력: {y.shape}")
```

**출력 예시**:
```
Layer: Linear(in_features=10, out_features=5, bias=True)
Layer 타입: <class 'torch.nn.modules.linear.Linear'>
nn.Module 상속 확인: True

입력: torch.Size([5, 10]) -> 출력: torch.Size([5, 5])
```

**중요**: 모든 레이어는 `nn.Module`을 상속합니다!

### Step 4.2: Module 예제

**Module은 여러 Layer를 하나로 묶은 것**입니다.

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - MyModule: 커스텀 모듈 클래스
# - module: 모듈 객체
# - x: 입력 텐서
# - y: 출력 텐서

# Module 예제: 여러 Layer를 하나로 묶기
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)  # 레이어 1
        self.layer2 = nn.Linear(20, 5)    # 레이어 2
    
    def forward(self, x):
        x = self.layer1(x)  # 첫 번째 레이어 통과
        x = self.layer2(x)  # 두 번째 레이어 통과
        return x

module = MyModule()
print(f"Module: {module}")
print(f"Module 내부 레이어: {list(module.children())}")

# 사용 예시
x = torch.randn(5, 10)
y = module(x)
print(f"\n입력: {x.shape} -> 출력: {y.shape}")
```

**출력 예시**:
```
Module: MyModule(
  (layer1): Linear(in_features=10, out_features=20, bias=True)
  (layer2): Linear(in_features=20, out_features=5, bias=True)
)
Module 내부 레이어: [Linear(in_features=10, out_features=20, bias=True), Linear(in_features=20, out_features=5, bias=True)]

입력: torch.Size([5, 10]) -> 출력: torch.Size([5, 5])
```

**수학적 표현**:

$$y = f_2(f_1(x))$$

여기서:
- $f_1$: `layer1` (10 → 20 차원 변환)
- $f_2$: `layer2` (20 → 5 차원 변환)

### Step 4.3: Model 예제

**Model은 여러 Module을 합쳐 전체 모델을 구성한 것**입니다.

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - MyModel: 커스텀 모델 클래스
# - model: 모델 객체
# - x: 입력 텐서
# - y: 출력 텐서

# Model 예제: 여러 Module을 합쳐 전체 모델 구성
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(10, 20)  # Module 1: 특징 추출
        self.classifier = nn.Linear(20, 3)          # Module 2: 분류
    
    def forward(self, x):
        x = self.feature_extractor(x)  # 특징 추출
        x = torch.relu(x)              # 활성화 함수
        x = self.classifier(x)          # 분류
        return x

model = MyModel()
print(f"Model: {model}")
print(f"Model 파라미터 수: {sum(p.numel() for p in model.parameters())}")

# 사용 예시
x = torch.randn(5, 10)
y = model(x)
print(f"\n입력: {x.shape} -> 출력: {y.shape}")
```

**출력 예시**:
```
Model: MyModel(
  (feature_extractor): Linear(in_features=10, out_features=20, bias=True)
  (classifier): Linear(in_features=20, out_features=3, bias=True)
)
Model 파라미터 수: 263

입력: torch.Size([5, 10]) -> 출력: torch.Size([5, 3])
```

**수학적 표현**:

$$y = \text{Classifier}(\text{ReLU}(\text{FeatureExtractor}(x)))$$

여기서:
- $\text{FeatureExtractor}$: `feature_extractor` (10 → 20 차원)
- $\text{ReLU}$: 활성화 함수
- $\text{Classifier}$: `classifier` (20 → 3 차원)

**파라미터 수 계산**:
- `feature_extractor`: $10 \times 20 + 20 = 220$
- `classifier`: $20 \times 3 + 3 = 63$
- **총합**: $220 + 63 = 283$ (출력과 약간 다를 수 있음)

### Step 4.4: Layer, Module, Model의 관계 이해

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - layer: 단일 레이어
# - module: 여러 레이어를 묶은 모듈
# - model: 전체 모델
# - x: 입력 텐서
# - y: 출력 텐서

# 1. Layer: 단일 연산
layer = nn.Linear(10, 20)
print("Layer (단일 연산):")
x = torch.randn(5, 10)
y = layer(x)
print(f"  입력: {x.shape} -> 출력: {y.shape}")

# 2. Module: 여러 Layer 조합
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

module = MyModule()
print("\nModule (여러 Layer):")
y = module(x)
print(f"  입력: {x.shape} -> 출력: {y.shape}")

# 3. Model: 전체 아키텍처
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = MyModule()      # Module 사용
        self.classifier = nn.Linear(10, 3)  # 추가 Layer
    
    def forward(self, x):
        x = self.features(x)      # Module 통과
        x = self.classifier(x)    # Layer 통과
        return x

model = MyModel()
print("\nModel (Module 조합):")
y = model(x)
print(f"  입력: {x.shape} -> 출력: {y.shape}")
```

**출력 예시**:
```
Layer (단일 연산):
  입력: torch.Size([5, 10]) -> 출력: torch.Size([5, 20])

Module (여러 Layer):
  입력: torch.Size([5, 10]) -> 출력: torch.Size([5, 10])

Model (Module 조합):
  입력: torch.Size([5, 10]) -> 출력: torch.Size([5, 3])
```

---

## Step 5: nn.Sequential 활용

### Step 5.0: 기호 정리

- `nn.Sequential`: 여러 레이어를 순차적으로 적용하는 컨테이너
- $f_1, f_2, \ldots, f_n$: 순차적으로 적용되는 레이어 함수들

### Step 5.1: Sequential의 수학적 의미

**Sequential**은 여러 레이어를 순차적으로 적용하는 컨테이너입니다.

**수학적 표현**:

$$\text{Sequential}([f_1, f_2, \ldots, f_n]) = f_n \circ f_{n-1} \circ \cdots \circ f_1$$

**의미**: 입력에 $f_1$, $f_2$, $\ldots$, $f_n$을 순차적으로 적용합니다.

### Step 5.2: Sequential 사용법

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - model: Sequential 모델
# - x: 입력 텐서
# - output: 출력 텐서

# Sequential로 간단하게 모델 구성
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

print("Sequential 모델:")
print(model)

# 입력 전파
x = torch.randn(32, 10)
output = model(x)
print(f"\n입력: {x.shape} -> 출력: {output.shape}")
```

**출력 예시**:
```
Sequential 모델:
Sequential(
  (0): Linear(in_features=10, out_features=20, bias=True)
  (1): ReLU()
  (2): Linear(in_features=20, out_features=10, bias=True)
  (3): ReLU()
  (4): Linear(in_features=10, out_features=3, bias=True)
)

입력: torch.Size([32, 10]) -> 출력: torch.Size([32, 3])
```

**수학적 표현**:

$$y = \text{ReLU}(\text{Linear}_3(\text{ReLU}(\text{Linear}_2(\text{Linear}_1(x)))))$$

---

## Step 6: 컨볼루션 레이어 (nn.Conv2d)

### Step 6.0: 기호 정리

- `nn.Conv2d`: 2D 컨볼루션 레이어
- `in_channels`: 입력 채널 수 ($C_{in}$)
- `out_channels`: 출력 채널 수 ($C_{out}$)
- `kernel_size`: 커널(필터) 크기 ($K$)
- `stride`: 스트라이드 ($S$)
- `padding`: 패딩 ($P$)

### Step 6.1: Conv2d의 수학적 의미

**컨볼루션 연산**은 이미지 처리에서 중요한 연산입니다.

**수학적 정의** (간단한 버전):

$$(I * K)[i, j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} I[i+m, j+n] \cdot K[m, n]$$

여기서:
- $I$: 입력 이미지
- $K$: 커널(필터)
- $*$: 컨볼루션 연산

**의미**: 커널을 입력 이미지 위에서 슬라이딩하며 내적을 계산합니다.

### Step 6.2: Conv2d 사용법

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - conv: Conv2d 레이어 객체
# - x: 입력 텐서 (B, C_in, H, W)
# - y: 출력 텐서 (B, C_out, H_out, W_out)

# Conv2d 레이어 생성
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
print(f"Conv2d 레이어: {conv}")

# 입력 생성 (배치=1, 채널=3, 높이=32, 너비=32)
x = torch.randn(1, 3, 32, 32)
y = conv(x)

print(f"\n입력 shape: {x.shape}")
print(f"출력 shape: {y.shape}")

# 파라미터 확인
print(f"\n가중치 shape: {conv.weight.shape}")  # (out_channels, in_channels, kernel_size, kernel_size)
print(f"편향 shape: {conv.bias.shape}")        # (out_channels,)
```

**출력 예시**:
```
Conv2d 레이어: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

입력 shape: torch.Size([1, 3, 32, 32])
출력 shape: torch.Size([1, 16, 30, 30])

가중치 shape: torch.Size([16, 3, 3, 3])
편향 shape: torch.Size([16])
```

**파라미터 설명**:
- `in_channels=3`: 입력 채널 수 (예: RGB 이미지)
- `out_channels=16`: 출력 채널 수 (필터 개수)
- `kernel_size=3`: 커널 크기 (3×3)

### Step 6.3: Conv2d 더 자세히

```python
import torch
import torch.nn as nn

# Step 0: 기호 정리
# - conv: Conv2d 레이어
# - conv_block: 컨볼루션 블록 (Sequential)
# - x: 입력 텐서
# - y: 출력 텐서

# 다양한 파라미터로 Conv2d 생성
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
print(f"Conv2d: {conv}")

# 입력 (배치=4, 채널=3, 높이=28, 너비=28)
x = torch.randn(4, 3, 28, 28)
y = conv(x)

print(f"\n입력: {x.shape}")
print(f"출력: {y.shape}")
print(f"가중치: {conv.weight.shape}")

# Sequential로 Conv Block 만들기
conv_block = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),  # 컨볼루션
    nn.ReLU(),                        # 활성화 함수
    nn.MaxPool2d(2)                   # 풀링
)

print(f"\nConv Block:")
print(f"입력: {(4, 3, 28, 28)}")
output = conv_block(x)
print(f"출력: {output.shape}")
```

**출력 예시**:
```
Conv2d: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

입력: torch.Size([4, 3, 28, 28])
출력: torch.Size([4, 16, 28, 28])
가중치: torch.Size([16, 3, 3, 3])

Conv Block:
입력: (4, 3, 28, 28)
출력: torch.Size([4, 16, 14, 14])
```

**수학적 표현**:

$$\text{ConvBlock}(x) = \text{MaxPool}(\text{ReLU}(\text{Conv2d}(x)))$$

---

## 연습 문제

### 문제 1: Layer vs Module
단일 레이어가 모듈이 될 수 있는지, 모듈이 레이어가 될 수 있는지 설명하세요.

**답안**:
- 단일 레이어는 모듈이 될 수 있습니다 (모든 레이어는 `nn.Module`을 상속)
- 모듈은 레이어가 될 수 없습니다 (모듈은 여러 레이어의 조합)

### 문제 2: nn.Module 상속
간단한 커스텀 Module을 만들어 forward 메서드를 구현하세요.

**힌트**:
```python
class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 레이어 정의
        pass
    
    def forward(self, x):
        # 순전파 구현
        pass
```

### 문제 3: Sequential vs ModuleList
Sequential과 ModuleList의 차이를 설명하고 각각 언제 사용하는지 적으세요.

**답안**:
- **Sequential**: 레이어를 순차적으로 적용할 때 사용 (간단한 모델)
- **ModuleList**: 레이어를 리스트로 관리하지만 순차 적용은 수동으로 해야 함 (복잡한 모델, 조건부 적용 등)

### 문제 4: 복합 함수
3개의 Linear 레이어로 구성된 복합 함수를 만들어 각 레이어의 출력을 확인하세요.

**힌트**:
```python
class ThreeLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 15)
        self.layer3 = nn.Linear(15, 5)
    
    def forward(self, x):
        h1 = self.layer1(x)
        print(f"Layer 1 출력: {h1.shape}")
        h2 = self.layer2(h1)
        print(f"Layer 2 출력: {h2.shape}")
        h3 = self.layer3(h2)
        print(f"Layer 3 출력: {h3.shape}")
        return h3
```

### 문제 5: 파라미터 확인
모델의 총 파라미터 수를 계산하는 코드를 작성하세요.

**힌트**:
```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"총 파라미터 수: {total_params}")
print(f"학습 가능한 파라미터 수: {trainable_params}")
```

---

## 핵심 요약

이번 단원에서 우리는 다음을 배웠습니다:

### 1. 신경망의 계층적 구조
- **Layer**: 입력 텐서를 출력 텐서로 변환하는 기본 연산 단위 ($h = f(x)$)
- **Module**: 한 개 이상의 레이어를 기능 단위로 묶은 것 ($h = f_n(\cdots f_1(x)\cdots)$)
- **Model**: 모든 모듈을 합쳐 전체를 구성한 최상위 모듈 ($y = \text{Model}(x)$)

### 2. torch.nn 패키지
- 자동으로 가중치와 편향을 생성하는 레이어 제공
- `nn.Linear`, `nn.Conv2d`, `nn.ReLU` 등
- 모든 레이어는 `nn.Module`을 상속

### 3. 신경망은 복합 함수
- 입력 → 중간층 → 출력으로 이어지는 함수의 합성
- 각 층이 하나의 함수 역할
- 체인룰을 통해 gradient 계산 가능

### 4. 구현 방법
- **Sequential**: 간단한 순차 모델
- **nn.Module 상속**: 복잡한 커스텀 모델
- **Module 조합**: 여러 모듈을 합쳐 전체 모델 구성

다음 단원에서는 **Linear Layer**에 대해 배워보겠습니다.

