# 05. 신경망 구조 이해

이번 단원에서는 신경망의 기본 구조와 PyTorch의 구조화 방식을 배워보겠습니다.

## 학습 목표
- 신경망의 계층적 구조 (Layer, Module, Model)
- torch.nn 패키지
- 신경망을 복합 함수로 이해하기

## 이 단원을 배우기 전에

**이전 단원 (04) 복습: Dataset, DataLoader를 만들고 사용할 수 있나요?**

## 이 단원 다음에는

**다음 단원 (06): 신경망 구조를 알았으니, 가장 기본적인 Linear Layer를 자세히 배워봅시다.**

## 왜 이 단원이 필요한가?

신경망은 레이어, 모듈, 모델의 계층적 구조로 이루어져 있습니다. 이 구조를 이해해야 모델을 제대로 설계할 수 있습니다.

- **Layer**: 기본 계산 단위
- **Module**: 기능 단위 그룹
- **Model**: 전체 아키텍처
- **nn.Module**: 모든 커스텀 모델의 부모 클래스

이 단원을 통해 신경망을 '함수의 합성'으로 이해하게 됩니다.

# 신경망 구성

## 계층 구조: Layer → Module → Model

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

### 레이어(layer)

- 신경망의 핵심 데이터 구조로 하나 이상의 텐서를 입력받아 하나 이상의 텐서를 출력
- 단일 계산 단위
- 입력 텐서를 받아 출력 텐서를 만드는 기본 연산단위(nn.Linear, nn.Conv2d)

### 모듈(module)

- 한개 이상의 Layer가 모여서 구성
- 관련 레이어들을 기능 단위로 묶은 것
- 하나 이상의 레이어와 하위 모듈을 캡슐화한 Pytorch 기본 구성요소
- 모든 사용자 정의 네트워크는 nn.Module을 상속한다.

### 모델(model)

- 한개 이상의 모듈이 모여서 구성
- 모든 모듈들이 합쳐 전체 모델을 구성
- 문제를 풀기 위해 최종적으로 조립한 최상위 모듈이며, 내부에 여러 모듈과 레이어를 포함할 수 있음.

**비유**: 계산 하나를 레이어라 할수 있고 레이어를 모아서 기능이 구성된것이 모듈 그리고 전체 집합이 모델

## torch.nn 패키지
https://pytorch.org/docs/stable/nn.html

주로 가중치(weights), 편향(bias)값들이 내부에서 자동으로 생성되는 레이어들을 사용할 때 사용 (weight 값들을 직접 선언 안함.)


> 자동으로 생성되는 레이어란?

PyTorch의 torch.nn 패키지는 신경망 구조를 쉽게 구성하도록 도와주는 레이어 모음집이다.

그 안의 nn.Linear, nn.Conv2d, nn.LSTM 같은 레이어들은 내부에서 학습 가능한 파라미터(가중치와 편향)을 자동으로 만들어준다. 

[Bias_Weights_in_Linear.md] 파일 참고

```python
import torch
import torch.nn as nn

# 간단한 Layer 예제
linear = nn.Linear(10, 5)
print(f"Layer: {linear}")
print(f"Layer 타입: {type(linear)}")
print(f"nn.Module 상속 확인: {isinstance(linear, nn.Module)}")
```

```python
# Module 예제: 여러 Layer를 하나로 묶기
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)  # 레이어 1
        self.layer2 = nn.Linear(20, 5)   # 레이어 2
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

module = MyModule()
print(f"Module: {module}")
print(f"Module 내부 레이어: {list(module.children())}")
```

```python
# Model 예제: 여러 Module을 합쳐 전체 모델 구성
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(10, 20)  # Module 1
        self.classifier = nn.Linear(20, 3)          # Module 2
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.relu(x)
        x = self.classifier(x)
        return x

model = MyModel()
print(f"Model: {model}")
print(f"Model 파라미터 수: {sum(p.numel() for p in model.parameters())}")
```

```python
# nn 모듈의 자동 가중치 생성 확인
import torch.nn as nn

# Linear 레이어 생성
linear = nn.Linear(5, 3)

# 자동으로 생성된 파라미터 확인
print("자동 생성된 파라미터:")
for name, param in linear.named_parameters():
    print(f"{name}: shape={param.shape}")
    print(f"{name} 초기값 예시:\n{param.data[:2, :2] if len(param.shape) > 1 else param.data[:3]}\n")

# 모델 사용
x = torch.randn(10, 5)
y = linear(x)
print(f"입력 shape: {x.shape}")
print(f"출력 shape: {y.shape}")
```

## 신경망 구성

- 레이어(layer): 신경망의 핵심 데이터 구조로 하나 이상의 텐서를 입력받아 하나 이상의 텐서를 출력
- 모듈(module): 한 개 이상의 계층이 모여서 구성
- 모델(model): 한 개 이상의 모듈이 모여서 구성

### `torch.nn` 패키지

주로 가중치(weights), 편향(bias)값들이 내부에서 자동으로 생성되는 레이어들을 사용할 때 사용 (`weight`값들을 직접 선언 안함)

https://pytorch.org/docs/stable/nn.html

```python
# nn.Linear 예제
import torch
import torch.nn as nn

# Linear 레이어 생성
linear = nn.Linear(in_features=10, out_features=5)
print(f"Linear 레이어: {linear}")

# 입력 생성 및 전파
input_tensor = torch.randn(32, 10)
output = linear(input_tensor)

print(f"\n입력 shape: {input_tensor.shape}")
print(f"출력 shape: {output.shape}")

# 파라미터 확인
print(f"\n가중치 shape: {linear.weight.shape}")
print(f"편향 shape: {linear.bias.shape}")
```

`nn.Linear` 계층 예제

```python
# nn.Conv2d 예제
import torch
import torch.nn as nn

# Conv2d 레이어 생성
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
print(f"Conv2d 레이어: {conv}")

# 입력 생성 (배치=1, 채널=3, 높이=32, 너비=32)
input_tensor = torch.randn(1, 3, 32, 32)
output = conv(input_tensor)

print(f"\n입력 shape: {input_tensor.shape}")
print(f"출력 shape: {output.shape}")

# 파라미터 확인
print(f"\n가중치 shape: {conv.weight.shape}")
print(f"편향 shape: {conv.bias.shape}")
```

`nn.Conv2d` 계층 예시

```python
# Layer, Module, Model의 관계 이해
import torch
import torch.nn as nn

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
        self.features = MyModule()
        self.classifier = nn.Linear(10, 3)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MyModel()
print("\nModel (Module 조합):")
y = model(x)
print(f"  입력: {x.shape} -> 출력: {y.shape}")
```

```python
# nn.Sequential 활용
import torch.nn as nn

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

```python
# nn.Conv2d 더 자세히
import torch
import torch.nn as nn

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
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

print(f"\nConv Block:")
print(f"입력: {(4, 3, 28, 28)}")
output = conv_block(x)
print(f"출력: {output.shape}")
```

### 컨볼루션 레이어(Convolution Layers)

`nn.Conv2d` 예제

- `in_channels`: channel의 갯수
- `out_channels`: 출력 채널의 갯수
- `kernel_size`: 커널(필터) 사이즈

## 연습 문제

### 문제 1: Layer vs Module
단일 레이어가 모듈이 될 수 있는지, 모듈이 레이어가 될 수 있는지 설명하세요.

### 문제 2: nn.Module 상속
간단한 커스텀 Module을 만들어 forward 메서드를 구현하세요.

### 문제 3: Sequential vs ModuleList
Sequential과 ModuleList의 차이를 설명하고 각각 언제 사용하는지 적으세요.

### 문제 4: 복합 함수
3개의 Linear 레이어로 구성된 복합 함수를 만들어 각 레이어의 출력을 확인하세요.

### 문제 5: 파라미터 확인
모델의 총 파라미터 수를 계산하는 코드를 작성하세요.

## 핵심 요약

### 1. 신경망의 계층적 구조
- **Layer**: 입력 텐서를 출력 텐서로 변환하는 기본 연산 단위
- **Module**: 한 개 이상의 레이어를 기능 단위로 묶은 것
- **Model**: 모든 모듈을 합쳐 전체를 구성한 최상위 모듈

### 2. torch.nn 패키지
- 자동으로 가중치와 편향을 생성하는 레이어 제공
- nn.Linear, nn.Conv2d, nn.ReLU 등

### 3. 신경망은 복합 함수
- 입력 → 중간층 → 출력으로 이어지는 함수의 합성
- 각 층이 하나의 함수 역할

다음 단원에서는 **Linear Layer**에 대해 배워보겠습니다.
