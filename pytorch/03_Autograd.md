# 03. Autograd (자동 미분)

이번 단원에서는 PyTorch의 핵심 기능인 **자동 미분(Automatic Differentiation)**에 대해 배워보겠습니다.

## 학습 목표
- 자동 미분의 개념과 필요성
- Gradient (기울기)의 의미
- 계산 그래프와 역전파
- `requires_grad`, `backward()` 사용법

## 왜 이 단원이 필요한가?

자동 미분(Autograd)은 PyTorch의 가장 중요한 기능입니다. 이것 없이는 모델이 학습하지 못합니다.

- **Gradient 계산**: 수천 개의 파라미터에 대한 미분을 자동으로
- **역전파**: loss.backward()만으로 모든 레이어 업데이트
- **계산 그래프**: 연산을 추적하며 자동으로 체인룰 적용

이 단원을 이해하지 못하면 '코드가 어떻게 학습하는가'를 전혀 이해할 수 없습니다.

## 이 단원을 배우기 전에

**이전 단원 (02) 복습**: 텐서 연산, 행렬곱, reshape를 자유롭게 할 수 있나요?

**이전 단원 (00a) 복습**: 체인룰(Chain Rule)의 의미와 역전파의 수학적 원리를 이해하셨나요?

## 이 단원 다음에는

**다음 단원 (04)**: gradient 계산을 배웠으니, 실제 학습 데이터를 준비하는 방법을 배워봅시다.

## 1. 왜 자동 미분이 필요한가?

### 먼저 생각해봅시다: 신경망에서 얼마나 많은 파라미터가 있나?

간단한 신경망 예시:
- 입력: 784차원 (MNIST 이미지)
- 은닉층 1: 128개 뉴런
- 은닉층 2: 64개 뉴런
- 출력: 10개 클래스

**총 파라미터 수**: 784×128 + 128×64 + 64×10 = **100,480개 파라미터!**

### 문제 상황

**Q: 100,480개 파라미터 각각에 대해 gradient를 수동으로 계산해야 한다면?**

**A: 불가능합니다!**

- 각 파라미터별로 미분 공식 작성
- 100,480번 계산
- 실수하기 쉬움
- 체인룰을 수동 적용...

### 해결책: 자동 미분 (Autograd)

PyTorch가 **자동으로** gradient를 계산합니다:
```python
loss.backward()  # 한 줄로 끝!
```

### 기계학습의 핵심 목표

> **모델의 예측이 실제 값과 얼마나 다른지를 측정하고**  
> **그 차이를 줄이기 위해 내부 매개변수(가중치)를 조정하는 것.**

이를 위해 필요한 두 가지 핵심 개념:

### 1.1 스칼라 손실 (Scalar Loss)

- 모델이 출력한 결과값이 정답과 얼마나 떨어져 있는지를 나타내는 **단 하나의 숫자**
- 오차(error)를 수치화한 척도
- 다차원의 오류를 **한 점수로 압축**한 것

### 1.2 미분값 (Gradient)

- 손실을 줄이려면 각 변수(가중치, 입력 등)을 **어떤 방향으로 얼마나 바꿔야 하는가**를 나타내는 값
- 손실에 대한 **민감도(sensitivity)**
- **모델이 잘못된 이유가 어디에 있는가**를 알려주는 지도

**핵심**: 자동 미분은 체인룰을 자동으로 적용하여 모든 파라미터의 gradient를 한 번에 계산합니다!

## Autograd(자동미분)

- `torch.autograd` 패키지는 Tensor의 모든 연산에 대해 **자동 미분** 제공
- 이는 코드를 어떻게 작성하여 실행하느냐에 따라 역전파가 정의된다는 뜻
- `backprop`를 위해 미분값을 자동으로 계산

### 00a 단원과의 연결: 체인룰이 핵심!

**이전 단원(00a) 복습**: 체인룰은 합성 함수의 미분입니다.

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**신경망은 여러 층의 합성 함수**입니다:
$$L = f_n(f_{n-1}(...f_2(f_1(x))...))$$

체인룰을 반복 적용하면 모든 파라미터의 gradient를 계산할 수 있습니다. **PyTorch의 autograd가 이를 자동으로 수행합니다!**

`requires_grad` 속성을 `True`로 설정하면, 해당 텐서에서 이루어지는 모든 연산들을 추적하기 시작

기록을 추적하는 것을 중단하게 하려면, `.detach()`를 호출하여 연산기록으로부터 분리

```python
import torch

# requires_grad=True로 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x: {x}")
print(f"requires_grad: {x.requires_grad}")

# 연산 수행 (자동으로 추적됨)
y = x * 2
z = y.sum()

print(f"\ny = x * 2: {y}")
print(f"z = sum(y): {z}")
print(f"y.requires_grad: {y.requires_grad}")
print(f"z.requires_grad: {z.requires_grad}")
```

`requires_grad_(...)`는 기존 텐서의 `requires_grad` 값을 바꿔치기(`in-place`)하여 변경

`grad_fn`: 미분값을 계산한 함수에 대한 정보 저장 (어떤 함수에 대해서 backprop 했는지)

```python
# requires_grad_를 사용한 변경
w = torch.tensor([1.0, 2.0], requires_grad=False)
print(f"처음: requires_grad={w.requires_grad}")

w.requires_grad_(True)
print(f"변경 후: requires_grad={w.requires_grad}")

# grad_fn 확인
v = w * 3
print(f"\nv = w * 3")
print(f"v.grad_fn: {v.grad_fn}")  # MulBackward 등의 연산 정보
```

### 기울기(Gradient)

```python
# backward() 예제
import torch

a = torch.tensor([2.0], requires_grad=True)
b = a ** 2
c = b * 3

print(f"a: {a}")
print(f"b = a^2: {b}")
print(f"c = b*3: {c}")

# 역전파 실행
c.backward()

print(f"\nbackward() 실행 후:")
print(f"a.grad: {a.grad}")  # dc/da = 6a = 12
```

```python
# grad 속성 확인
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()  # x1^2 + x2^2

y.backward()

print(f"x: {x}")
print(f"y = sum(x^2): {y.item()}")
print(f"x.grad: {x.grad}")  # 각 원소에 대한 gradient
# dy/dx = [2x1, 2x2] = [2, 4]
```

```python
# grad_fn으로 계산 그래프 확인
import torch

x = torch.tensor([1.0], requires_grad=True)
a = x * 2
b = a ** 2
c = b + 3

print(f"x: {x}")
print(f"a = x*2: {a}")
print(f"b = a^2: {b}")
print(f"c = b+3: {c}")

print(f"\ngrad_fn 체인:")
print(f"c.grad_fn: {c.grad_fn}")
print(f"b.grad_fn: {b.grad_fn}")
print(f"a.grad_fn: {a.grad_fn}")
print(f"x.grad_fn: {x.grad_fn}")  # None (leaf node)
```

계산이 완료된 후, `.backward()`를 호출하면 자동으로 역전파 계산이 가능하고, `.grad` 속성에 누적됨

```python
# no_grad() 사용 예제
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

# gradient 계산
y = (x ** 2).sum()
y.backward()
print(f"gradient 계산 후: {x.grad}")

# grad 초기화
x.grad.zero_()

# no_grad 블록 내에서는 gradient 계산 안 됨
with torch.no_grad():
    z = (x ** 3).sum()
    # z.backward()  # 에러 발생!

print(f"\nno_grad 블록 밖에서:")
z = (x ** 3).sum()
z.backward()
print(f"normal gradient: {x.grad}")
```

`grad`: data가 거쳐온 layer에 대한 미분값 저장

```python
# detach() 예제
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y + 5

# z는 gradient 추적됨
print(f"z.requires_grad: {z.requires_grad}")

# detach로 추적 중단
z_detached = z.detach()
print(f"z_detached.requires_grad: {z_detached.requires_grad}")

# detach된 텐서는 같은 값이지만 gradient 계산에 영향 없음
print(f"z: {z}, z_detached: {z_detached}")

# detach된 텐서로 연산하면 gradient 없음
w = z_detached * 2
print(f"\nw = z_detached * 2")
print(f"w.requires_grad: {w.requires_grad}")
```

```python
# 실제 신경망에서의 활용 예제
import torch
import torch.nn as nn

# 간단한 모델
model = nn.Linear(3, 1)
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False)  # 입력 데이터는 gradient 불필요
target = torch.tensor([[5.0]])

# Forward pass
prediction = model(x)
loss = nn.functional.mse_loss(prediction, target)

print(f"입력 x: {x}")
print(f"예측: {prediction}")
print(f"손실: {loss}")

# Backward pass
loss.backward()

print(f"\n모델 가중치의 gradient:")
print(f"weight.grad: {model.weight.grad}")
print(f"bias.grad: {model.bias.grad}")
```

```python
# 평가 모드에서 no_grad 사용
import torch
import torch.nn as nn

model = nn.Linear(3, 1)

# 학습 시 (gradient 필요)
x_train = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
pred_train = model(x_train)
loss_train = pred_train.mean()
loss_train.backward()
print(f"학습 시 gradient 확인: {model.weight.grad is not None}")

# gradient 초기화
model.weight.grad.zero_()
model.bias.grad.zero_()

# 평가 시 (gradient 불필요) - 메모리 효율적
with torch.no_grad():
    x_eval = torch.tensor([[4.0, 5.0, 6.0]])
    pred_eval = model(x_eval)
    print(f"평가 시 prediction: {pred_eval}")
    print(f"평가 후 gradient 변경 확인: {model.weight.grad}")
```

`with torch.no_grad()`를 사용하여 기울기의 업데이트를 하지 않음

기록을 추적하는 것을 방지하기 위해 코드 블럭을 `with torch.no_grad()`로 감싸면 기울기 계산은 필요없지만, `requires_grad=True`로 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용

```python
# detach() 실제 활용: 중간 계산값 복사
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
z = y.sum()

# 중간값을 gradient 추적 없이 사용하고 싶을 때
y_detached = y.detach()

# detach된 값과 gradient 추적 값 비교
print(f"y (추적): {y}")
print(f"y_detached (비추적): {y_detached}")
print(f"값은 같음: {torch.equal(y, y_detached)}")
print(f"requires_grad 다름: y={y.requires_grad}, y_detached={y_detached.requires_grad}")
```

`detach()`: 내용물(content)은 같지만 `require_grad`가 다른 새로운 Tensor를 가져올 때

```python
# 다층 신경망에서의 gradient flow
import torch
import torch.nn as nn

# 간단한 2층 신경망
x = torch.randn(10, 5)

# 각 레이어별로 gradient 확인
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
output = model(x)
loss = output.mean()

# Backward pass
loss.backward()

print("각 레이어의 gradient:")
print(f"fc1.weight.grad shape: {model.fc1.weight.grad.shape}")
print(f"fc1.bias.grad shape: {model.fc1.bias.grad.shape}")
print(f"fc2.weight.grad shape: {model.fc2.weight.grad.shape}")
print(f"fc2.bias.grad shape: {model.fc2.bias.grad.shape}")
```

### 자동 미분 흐름 예제

- 계산 흐름 $a \rightarrow b  \rightarrow c  \rightarrow out $

## $\quad \frac{\partial out}{\partial a} = ?$
- `backward()`를 통해 $a \leftarrow b  \leftarrow c  \leftarrow out $을 계산하면 $\frac{\partial out}{\partial a}$값이 `a.grad`에 채워짐

```python
# 자동 미분 흐름 예제: a -> b -> c -> out
import torch

# a는 직접 생성 (leaf node)
a = torch.tensor([2.0], requires_grad=True)
print(f"a: {a}")

# b = a + 2
b = a + 2
print(f"b = a + 2: {b}")
print(f"b.grad_fn: {b.grad_fn}")

# c = b^2
c = b ** 2
print(f"c = b^2: {c}")
print(f"c.grad_fn: {c.grad_fn}")

# out = c의 합
out = c.sum()
print(f"out = sum(c): {out}")
print(f"out.grad_fn: {out.grad_fn}")

# 역전파: ∂out/∂a 계산
out.backward()
print(f"\nbackward() 실행 후:")
print(f"∂out/∂a = {a.grad}")  # 실제 값: 2*(a+2) = 2*4 = 8
```

```python
# 계산 그래프 추적하기
import torch

x = torch.tensor([3.0], requires_grad=True)

# 복잡한 계산 체인
y1 = x ** 2
y2 = y1 * 3
y3 = torch.sin(y2)
out = y3.sum()

print("계산 그래프:")
print(f"x (leaf): requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
print(f"y1: grad_fn={type(y1.grad_fn).__name__}")
print(f"y2: grad_fn={type(y2.grad_fn).__name__}")
print(f"y3: grad_fn={type(y3.grad_fn).__name__}")
print(f"out: grad_fn={type(out.grad_fn).__name__}")

# 역전파
out.backward()
print(f"\nx.grad: {x.grad}")
```

```python
# 중복 backward 방지 및 해결
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# 첫 번째 backward
y.backward()
print(f"첫 번째 backward 후: {x.grad}")

# 두 번째 backward 시도 (에러 발생 가능)
try:
    y.backward()
except RuntimeError as e:
    print(f"\n에러 발생: {e}")
    print("해결책: retain_graph=True 사용")
    
# retain_graph=True로 여러 번 backward 가능
x.grad.zero_()
y.backward(retain_graph=True)
print(f"첫 번째 backward (retain_graph): {x.grad}")
y.backward()  # 두 번째 backward도 성공
print(f"두 번째 backward 후 (누적): {x.grad}")
```

$b = a + 2$

```python
# $b = a + 2$ 계산
import torch

a = torch.tensor([2.0], requires_grad=True)
print(f"a: {a}")

# b = a + 2
b = a + 2
print(f"b = a + 2: {b}")
print(f"b의 연산: {b.grad_fn}")
```

$c = b^2$

```python
# $c = b^2$ 계산
import torch

a = torch.tensor([2.0], requires_grad=True)
b = a + 2

# c = b^2
c = b ** 2
print(f"c = b^2: {c}")
print(f"c의 연산: {c.grad_fn}")

# 전체 체인: a -> b -> c
print(f"\n전체 체인:")
print(f"a -> {a.grad_fn} -> b -> {b.grad_fn} -> c -> {c.grad_fn}")
```

```python
# backward()로 ∂out/∂a 계산
import torch

a = torch.tensor([2.0], requires_grad=True)
b = a + 2
c = b ** 2
out = c.sum()

print(f"계산 체인: a -> b -> c -> out")
print(f"a: {a}")
print(f"b = a + 2: {b}")
print(f"c = b^2: {c}")
print(f"out = sum(c): {out}")

# 역전파 실행
out.backward()

print(f"\n역전파 후:")
print(f"∂out/∂a = {a.grad}")

# 이론값 검증: ∂out/∂a = 2*(a+2) = 2*(2+2) = 8
print(f"이론값: 2*(a+2) = 2*{a.item()+2} = {2*(a.item()+2)}")
```

```python
# grad_fn None인 이유 확인
import torch

# 직접 생성된 텐서는 grad_fn이 None (leaf node)
a = torch.tensor([1.0], requires_grad=True)
print(f"a.grad_fn: {a.grad_fn}")  # None

# 연산을 통해 생성된 텐서는 grad_fn이 있음 (non-leaf node)
b = a * 2
print(f"b.grad_fn: {b.grad_fn}")  # MulBackward

# leaf node vs non-leaf node
print(f"\na.requires_grad: {a.requires_grad}")
print(f"a.is_leaf: {a.is_leaf}")  # True
print(f"b.is_leaf: {b.is_leaf}")  # False
```

a의 `grad_fn`이 None인 이유는 직접적으로 계산한 부분이 없었기 때문

```python
# 실제 모델 학습에서의 예제
import torch
import torch.nn as nn

# 간단한 선형 모델
model = nn.Linear(3, 1)
x = torch.randn(4, 3)
y = torch.randn(4, 1)

# Forward
pred = model(x)
loss = nn.functional.mse_loss(pred, y)

print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

# 각 파라미터의 gradient 확인
for name, param in model.named_parameters():
    print(f"{name} gradient shape: {param.grad.shape}")
    print(f"{name} gradient 값: {param.grad}")
```

```python
# gradient accumulation 예제
import torch
import torch.nn as nn

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 여러 배치 처리 (gradient 누적)
for i in range(3):
    x = torch.randn(2, 3)
    y = torch.randn(2, 1)
    
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    
    loss.backward()  # gradient 누적
    
    print(f"Batch {i+1} loss: {loss.item():.4f}")
    print(f"Gradient after batch {i+1}: {model.weight.grad[0, 0]:.6f}")

# optimizer step
optimizer.step()
optimizer.zero_grad()  # gradient 초기화
```

```python
# 여러 입력에 대한 gradient
import torch

# 벡터 입력
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 다중 출력
y = x ** 2
z = y.sum()

# backward
z.backward()

print(f"x: {x}")
print(f"y = x^2: {y}")
print(f"z = sum(y): {z}")
print(f"\n∂z/∂x: {x.grad}")  # [2, 4, 6]
```

```python
# 실제 활용: 신경망 전체 흐름
import torch
import torch.nn as nn

# 간단한 신경망
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 데이터
x = torch.randn(32, 5)
y = torch.randn(32, 1)

# Forward
pred = model(x)
loss = nn.functional.mse_loss(pred, y)

print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

# 모든 레이어의 gradient 확인
print("\n모든 레이어의 gradient:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: gradient 계산됨")
    else:
        print(f"{name}: gradient 없음")
```

## 데이터 준비

파이토치에서는 데이터 준비를 위해 `torch.utils.data`의 `Dataset`과 `DataLoader` 사용 가능

- `Dataset`에는 다양한 데이터셋이 존재 (MNIST, FashionMNIST, CIFAR10, ...)
  - Vision Dataset: https://pytorch.org/vision/stable/datasets.html
  - Text Dataset: https://pytorch.org/text/stable/datasets.html
  - Audio Dataset: https://pytorch.org/audio/stable/datasets.html
- `DataLoader`와 `Dataset`을 통해 `batch_size`, `train` 여부, `transform` 등을 인자로 넣어 데이터를 어떻게 load할 것인지 정해줄 수 있음

```python
# Dataset 예제 (실제 사용은 04단원에서)
import torch
from torch.utils.data import Dataset, DataLoader

# 간단한 커스텀 Dataset
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 데이터 생성
data = [torch.tensor(i, dtype=torch.float32) for i in range(10)]
dataset = SimpleDataset(data)
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

print("Dataloader 배치 출력:")
for i, batch in enumerate(dataloader):
    print(f"Batch {i}: {batch}")
```

## 4. 학습 가능한 매개변수와 평가 모드

### 4.1 `requires_grad=True`의 의미

PyTorch에서 `requires_grad=True`로 설정하면 "이 변수는 학습 대상이다"라고 명시하는 것입니다.

**학습 가능한 매개변수 (Learnable Parameters):**
- 신경망 내부에서 실제로 손실을 줄이기 위해 조정되는 값들
- 가중치(weight)와 편향(bias)
- 데이터(x, y) 자체가 아니라 **가중치나 편향**을 의미

이 매개변수들은 `requires_grad=True`로 설정되어 있어야:
- `loss.backward()` 실행 시 `param.grad`에 미분값이 기록됨
- `optimizer`가 값을 갱신할 수 있음

**한 줄 정리:** 학습 가능한 매개변수 = gradient가 계산되고 업데이트되는 대상

### 4.2 `torch.no_grad()` 활용

**평가(Evaluate):** 이미 학습된 모델의 성능을 확인하는 단계

평가 시에는 gradient 계산이 필요 없으므로 `torch.no_grad()`로 감싸면:
- 메모리 사용량 감소
- 계산 속도 향상

## 5. Layer별 Gradient 저장의 의미

### 5.1 신경망은 복합 함수

- 딥러닝 모델은 **복합 함수(composite function)**
- 각 Layer는 입력 x를 받아 새로운 표현으로 변환하는 하나의 함수
- 입력층: 원본 데이터를 첫 번째 수학 공간으로 투사
- 은닉층: 특징 추출 및 비선형 변환
- 출력층: 예측값 또는 확률로 변환

### 5.2 Gradient가 Layer별로 저장되는 이유

**Gradient는 "손실 L이 각 변수(가중치, 입력)에 얼마나 영향을 받는가"**를 나타내는 값입니다.

역전파 과정:
1. **Forward pass**: 입력에서 출력으로 순전파
2. **Backward pass**: 출력에서 입력으로 역전파하면서 각 Layer를 통과할 때마다 계산된 기울기 저장

> 각 Layer가 손실에 얼마나 기여했는지를 나타내는 기울기를 역전파 중에 저장합니다.

## 6. 계산 그래프 (Computational Graph)

PyTorch의 autograd 시스템은 내부적으로 **계산 그래프** 형태로 연산을 저장합니다.

### 6.1 왜 계산 그래프가 필요한가?

**00a 단원 연결**: 체인룰을 적용하려면 연산의 연결 구조가 필요합니다!

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial w_1}$$

위 공식에서 보듯, 각 레이어의 출력이 다음 레이어의 입력이 되는 **연결 관계**를 추적해야 합니다.  
**계산 그래프가 바로 이 연결 구조를 저장합니다!**

### 6.2 계산 그래프 구조

- **Node (노드)**: 하나의 연산 또는 중간 결과
  - 데이터(content)
  - 연산자(grad_fn)
- **Edge (엣지)**: 데이터가 전달되는 경로
- **Leaf Node**: 사용자가 직접 만든 입력 (grad_fn = None)
- **Non-leaf Node**: 연산을 통해 만들어진 결과 (grad_fn ≠ None)

### 6.3 예시

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
```

계산 그래프:
```
x (입력)
 ↓
LinearBackward (3→5 변환)
 ↓
ReLUBackward (활성화)
 ↓
LinearBackward (5→1 변환)
 ↓
output (출력)
```

**각 화살표가 체인룰의 한 단계입니다!**

Autograd는 각 연산의 합성 형태의 순서가 존재하고 그 연산의 원인이 존재하도록 그래프를 구성합니다.  
**backward()가 호출되면 이 그래프를 역순으로 순회하며 체인룰을 적용합니다!**

## 연습 문제

### 문제 1: requires_grad 이해
requires_grad=True와 False 텐서를 더하면, 결과 텐서의 requires_grad는? 실험해보세요.

### 문제 2: backward 호출
여러 스칼라 손실을 backward()로 역전파하면 어떤 일이 일어나는지 설명하세요.

### 문제 3: no_grad 활용
no_grad 블록 안팎에서 동일한 연산을 수행하고 메모리/시간 차이를 설명하세요.

### 문제 4: 계산 그래프
grad_fn 속성을 출력하여 계산 그래프를 추적해보세요.

### 문제 5: gradient flow
여러 레이어를 지나가는 gradient가 올바르게 전파되는지 확인하는 코드를 작성하세요.

## 핵심 요약

이번 단원에서 우리는 다음을 배웠습니다:

### 1. 자동 미분의 목적
- 모델이 예측과 실제의 차이를 줄이기 위해 가중치를 조정
- 손실(Loss)과 기울기(Gradient)라는 두 핵심 개념

### 2. Gradient의 의미
- 손실에 대한 **민감도**: 각 가중치를 얼마나, 어떤 방향으로 바꿔야 하는가
- "모델이 잘못된 이유가 어디에 있는가"를 알려주는 지도
- **역전파**: 출력에서 입력으로 gradient를 전파하는 연쇄적 미분 과정

### 3. PyTorch의 Autograd 시스템
- `requires_grad=True`: 이 변수가 학습 대상임을 명시
- `backward()`: 역전파 실행하여 gradient 계산
- `.grad`: 계산된 gradient 저장

### 4. 계산 그래프
- 각 연산을 노드로, 데이터 흐름을 엣지로 표현
- 자동으로 연쇄법칙(chain rule) 적용
- **Layer별로 gradient 저장**: 각 층의 기여도 추적

### 5. 학습 vs 평가
- **학습**: `requires_grad=True`로 gradient 계산
- **평가**: `torch.no_grad()`로 메모리/속도 최적화

다음 단원에서는 **데이터 준비**에 대해 배워보겠습니다.
