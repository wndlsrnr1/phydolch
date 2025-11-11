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

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 자동 미분 관련 기호

- $L$: 손실 함수 (Loss Function)
- $\theta$: 모델 파라미터 (가중치, 편향)
- $\frac{\partial L}{\partial \theta}$: 손실에 대한 파라미터의 기울기 (Gradient)
- $n$: 파라미터 개수 (예: 간단한 신경망에서 $n = 100,480$)
- $f_i$: $i$번째 레이어의 함수
- $h_i$: $i$번째 레이어의 출력 (hidden state)

### PyTorch 코드에서의 변수

- `requires_grad`: gradient 계산 여부를 나타내는 불리언 값
- `grad_fn`: gradient를 계산한 함수에 대한 정보
- `grad`: 계산된 gradient 값
- `backward()`: 역전파를 실행하는 함수
- `detach()`: gradient 추적을 중단하는 함수
- `no_grad()`: gradient 계산을 비활성화하는 컨텍스트 매니저

---

## Step 1: 왜 자동 미분이 필요한가?

### Step 1.1: 문제 상황

**신경망의 파라미터 개수**:

간단한 신경망 예시:
- 입력: 784차원 (MNIST 이미지)
- 은닉층 1: 128개 뉴런
- 은닉층 2: 64개 뉴런
- 출력: 10개 클래스

**총 파라미터 수**: 
- 첫 번째 레이어: $784 \times 128 = 100,352$ (가중치) + $128$ (편향) = $100,480$
- 두 번째 레이어: $128 \times 64 = 8,192$ (가중치) + $64$ (편향) = $8,256$
- 세 번째 레이어: $64 \times 10 = 640$ (가중치) + $10$ (편향) = $650$
- **총합**: $100,480 + 8,256 + 650 = 109,386$개 파라미터!

### Step 1.2: 수동 계산의 문제점

**Q: 109,386개 파라미터 각각에 대해 gradient를 수동으로 계산해야 한다면?**

**A: 불가능합니다!**

**문제점**:
1. 각 파라미터별로 미분 공식 작성 → 109,386번 계산
2. 실수하기 쉬움 (체인룰 적용 과정에서 실수)
3. 체인룰을 수동 적용 → 매우 복잡
4. 시간이 너무 오래 걸림

**예시**: 하나의 파라미터에 대한 gradient 계산
- 3층 신경망의 경우: $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial w_1}$
- 각 레이어마다 미분을 계산해야 함
- 109,386개 파라미터 각각에 대해 이 과정을 반복!

### Step 1.3: 해결책: 자동 미분 (Autograd)

PyTorch가 **자동으로** gradient를 계산합니다:

```python
loss.backward()  # 한 줄로 끝!
```

**장점**:
- 모든 파라미터의 gradient를 한 번에 계산
- 체인룰을 자동으로 적용
- 실수 없이 정확한 gradient 계산
- 빠른 계산 속도

### Step 1.4: 기계학습의 핵심 목표

> **모델의 예측이 실제 값과 얼마나 다른지를 측정하고**  
> **그 차이를 줄이기 위해 내부 매개변수(가중치)를 조정하는 것.**

이를 위해 필요한 두 가지 핵심 개념:

#### Step 1.4.1: 스칼라 손실 (Scalar Loss)

- 모델이 출력한 결과값이 정답과 얼마나 떨어져 있는지를 나타내는 **단 하나의 숫자**
- 오차(error)를 수치화한 척도
- 다차원의 오류를 **한 점수로 압축**한 것

**예시**: 
- 회귀 문제: $L = \frac{1}{2}(y - \hat{y})^2$ (평균 제곱 오차)
- 분류 문제: $L = -\log P(y|\hat{y})$ (교차 엔트로피)

#### Step 1.4.2: 미분값 (Gradient)

- 손실을 줄이려면 각 변수(가중치, 입력 등)을 **어떤 방향으로 얼마나 바꿔야 하는가**를 나타내는 값
- 손실에 대한 **민감도(sensitivity)**
- **모델이 잘못된 이유가 어디에 있는가**를 알려주는 지도

**예시**:
- $\frac{\partial L}{\partial w} > 0$: 가중치 $w$를 감소시켜야 손실이 줄어듦
- $\frac{\partial L}{\partial w} < 0$: 가중치 $w$를 증가시켜야 손실이 줄어듦
- $\frac{\partial L}{\partial w} = 0$: 현재 가중치가 최적 (로컬 최소값 또는 최대값)

**핵심**: 자동 미분은 체인룰을 자동으로 적용하여 모든 파라미터의 gradient를 한 번에 계산합니다!

---

## Step 2: 체인룰의 수학적 정의 (복습)

**이전 단원(00a) 복습**: 체인룰은 합성 함수의 미분입니다.

### Step 2.0: 기호 정리

- $y = f(g(x))$: 합성 함수
- $\frac{dy}{dx}$: $y$를 $x$에 대해 미분한 값
- $\frac{dy}{dg}$: $y$를 $g$에 대해 미분한 값
- $\frac{dg}{dx}$: $g$를 $x$에 대해 미분한 값

### Step 2.1: 체인룰의 정의

**체인룰 (Chain Rule)**:

합성 함수 $y = f(g(x))$의 미분:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**의미**: 
- 외부 함수 $f$의 미분 × 내부 함수 $g$의 미분
- 두 미분값을 곱하면 전체 미분값을 구할 수 있음

### Step 2.2: 체인룰의 확장

**여러 층의 합성 함수**:

$y = f_3(f_2(f_1(x)))$의 경우:

$$\frac{dy}{dx} = \frac{dy}{df_2} \cdot \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

**일반화**:

$y = f_n(f_{n-1}(...f_2(f_1(x))...))$의 경우:

$$\frac{dy}{dx} = \frac{dy}{df_n} \cdot \frac{df_n}{df_{n-1}} \cdots \frac{df_2}{df_1} \cdot \frac{df_1}{dx}$$

**의미**: 각 레이어의 미분을 차례로 곱하면 전체 미분값을 구할 수 있습니다.

---

## Step 3: 신경망에서의 체인룰

### Step 3.0: 기호 정리

- $x$: 입력 벡터
- $h_1 = f_1(x)$: 첫 번째 레이어의 출력
- $h_2 = f_2(h_1)$: 두 번째 레이어의 출력
- $L = f_3(h_2)$: 손실 함수
- $w_i$: $i$번째 레이어의 가중치

### Step 3.1: 신경망은 여러 층의 합성 함수

**신경망 구조**:

$$L = f_3(f_2(f_1(x)))$$

여기서:
- $f_1$: 첫 번째 레이어 (선형 변환 + 활성화 함수)
- $f_2$: 두 번째 레이어 (선형 변환 + 활성화 함수)
- $f_3$: 손실 함수

### Step 3.2: Gradient 계산

**첫 번째 레이어의 가중치 $w_1$에 대한 gradient**:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$$

**의미**:
- $\frac{\partial L}{\partial h_2}$: 손실이 두 번째 레이어 출력에 얼마나 민감한가?
- $\frac{\partial h_2}{\partial h_1}$: 두 번째 레이어 출력이 첫 번째 레이어 출력에 얼마나 민감한가?
- $\frac{\partial h_1}{\partial w_1}$: 첫 번째 레이어 출력이 가중치 $w_1$에 얼마나 민감한가?

**체인룰 적용**: 세 미분값을 곱하면 $w_1$에 대한 gradient를 구할 수 있습니다!

### Step 3.3: 역전파 (Backpropagation)

**역전파 과정**:
1. **Forward pass**: 입력에서 출력으로 순전파
   - $h_1 = f_1(x)$
   - $h_2 = f_2(h_1)$
   - $L = f_3(h_2)$

2. **Backward pass**: 출력에서 입력으로 역전파
   - $\frac{\partial L}{\partial h_2}$ 계산
   - $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1}$ 계산
   - $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$ 계산

**핵심**: 역전파는 체인룰을 역순으로 적용하여 각 파라미터의 gradient를 계산합니다!

---

## Step 4: PyTorch에서의 구현 (계산 그래프)

### Step 4.0: 기호 정리

- **Node (노드)**: 하나의 연산 또는 중간 결과
  - 데이터(content)
  - 연산자(grad_fn)
- **Edge (엣지)**: 데이터가 전달되는 경로
- **Leaf Node**: 사용자가 직접 만든 입력 (grad_fn = None)
- **Non-leaf Node**: 연산을 통해 만들어진 결과 (grad_fn ≠ None)

### Step 4.1: 왜 계산 그래프가 필요한가?

**00a 단원 연결**: 체인룰을 적용하려면 연산의 연결 구조가 필요합니다!

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial w_1}$$

위 공식에서 보듯, 각 레이어의 출력이 다음 레이어의 입력이 되는 **연결 관계**를 추적해야 합니다.  
**계산 그래프가 바로 이 연결 구조를 저장합니다!**

### Step 4.2: 계산 그래프 구조

**계산 그래프**는 다음과 같은 구조를 가집니다:

```
x (입력, leaf node)
 ↓
연산 1 (grad_fn: MulBackward)
 ↓
h1 (중간 결과, non-leaf node)
 ↓
연산 2 (grad_fn: PowBackward)
 ↓
h2 (중간 결과, non-leaf node)
 ↓
연산 3 (grad_fn: AddBackward)
 ↓
L (출력, non-leaf node)
```

**역전파 과정**:
1. $L$에서 시작하여 역순으로 그래프를 순회
2. 각 노드에서 gradient를 계산
3. 체인룰을 적용하여 다음 노드로 gradient 전파

### Step 4.3: PyTorch에서의 계산 그래프

PyTorch는 **계산 그래프(Computational Graph)**를 자동으로 추적합니다:

```python
import torch

# 입력 (leaf node)
x = torch.tensor([1.0], requires_grad=True)
print(f"x: {x}")
print(f"x.grad_fn: {x.grad_fn}")  # None (leaf node)

# 레이어 1: h1 = 2x
h1 = x * 2
print(f"\nh1 = x * 2: {h1}")
print(f"h1.grad_fn: {h1.grad_fn}")  # MulBackward

# 레이어 2: h2 = h1^2
h2 = h1 ** 2
print(f"\nh2 = h1^2: {h2}")
print(f"h2.grad_fn: {h2.grad_fn}")  # PowBackward

# 출력: L = 3h2
L = h2 * 3
print(f"\nL = 3h2: {L}")
print(f"L.grad_fn: {L.grad_fn}")  # MulBackward
```

**출력 예시**:
```
x: tensor([1.], requires_grad=True)
x.grad_fn: None

h1 = x * 2: tensor([2.], grad_fn=<MulBackward0>)
h1.grad_fn: <MulBackward0 object at 0x...>

h2 = h1^2: tensor([4.], grad_fn=<PowBackward0>)
h2.grad_fn: <PowBackward0 object at 0x...>

L = 3h2: tensor([12.], grad_fn=<MulBackward0>)
L.grad_fn: <MulBackward0 object at 0x...>
```

### Step 4.4: 역전파 (Backward Pass)

**역전파 실행**:

```python
import torch

# 계산 그래프 구성
x = torch.tensor([1.0], requires_grad=True)
h1 = x * 2  # h1 = 2x
h2 = h1 ** 2  # h2 = (2x)^2 = 4x^2
L = h2 * 3  # L = 3 * 4x^2 = 12x^2

print(f"계산 체인: x -> h1 -> h2 -> L")
print(f"x: {x.item()}")
print(f"h1 = 2x = {h1.item()}")
print(f"h2 = h1^2 = {h2.item()}")
print(f"L = 3h2 = {L.item()}")

# 역전파 실행
L.backward()

print(f"\n역전파 후:")
print(f"∂L/∂x = {x.grad.item()}")

# 이론값 검증: ∂L/∂x = d(12x^2)/dx = 24x = 24
print(f"이론값: 24x = 24×{x.item()} = {24 * x.item()}")
```

**출력 예시**:
```
계산 체인: x -> h1 -> h2 -> L
x: 1.0
h1 = 2x = 2.0
h2 = h1^2 = 4.0
L = 3h2 = 12.0

역전파 후:
∂L/∂x = 24.0
이론값: 24x = 24×1.0 = 24.0
```

**계산 과정**:
1. $\frac{\partial L}{\partial h_2} = 3$ (L = 3h2의 미분)
2. $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} = 3 \cdot 2h_1 = 6h_1 = 6 \times 2 = 12$
3. $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial x} = 12 \cdot 2 = 24$

PyTorch는 이 과정을 **자동으로** 수행합니다!

---

## Step 5: requires_grad와 backward()

### Step 5.0: 기호 정리

- `requires_grad`: gradient 계산 여부를 나타내는 불리언 값
- `grad_fn`: gradient를 계산한 함수에 대한 정보
- `grad`: 계산된 gradient 값
- `backward()`: 역전파를 실행하는 함수

### Step 5.1: requires_grad의 의미

**`requires_grad=True`**: 이 변수는 학습 대상입니다.

**의미**:
- 이 변수에서 이루어지는 모든 연산을 추적
- `backward()` 호출 시 이 변수의 gradient를 계산
- 신경망의 가중치(weight)와 편향(bias)에 설정

#### Step 5.1.1: requires_grad 사용 예제

```python
import torch

# requires_grad=True로 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x: {x}")
print(f"x.requires_grad: {x.requires_grad}")

# 연산 수행 (자동으로 추적됨)
y = x * 2
z = y.sum()

print(f"\ny = x * 2: {y}")
print(f"z = sum(y): {z}")
print(f"y.requires_grad: {y.requires_grad}")
print(f"z.requires_grad: {z.requires_grad}")
```

**출력 예시**:
```
x: tensor([1., 2., 3.], requires_grad=True)
x.requires_grad: True

y = x * 2: tensor([2., 4., 6.], grad_fn=<MulBackward0>)
z = sum(y): tensor(12., grad_fn=<SumBackward0>)
y.requires_grad: True
z.requires_grad: True
```

#### Step 5.1.2: requires_grad_() 사용

**`requires_grad_()`**: 기존 텐서의 `requires_grad` 값을 변경 (in-place)

```python
import torch

# 처음에는 requires_grad=False
w = torch.tensor([1.0, 2.0], requires_grad=False)
print(f"처음: requires_grad={w.requires_grad}")

# requires_grad_()로 변경
w.requires_grad_(True)
print(f"변경 후: requires_grad={w.requires_grad}")

# grad_fn 확인
v = w * 3
print(f"\nv = w * 3")
print(f"v.grad_fn: {v.grad_fn}")  # MulBackward 등의 연산 정보
```

### Step 5.2: backward()의 동작 원리

**`backward()`**: 역전파를 실행하여 gradient를 계산합니다.

#### Step 5.2.1: backward() 기본 사용

```python
import torch

# 계산 그래프 구성
a = torch.tensor([2.0], requires_grad=True)
b = a ** 2  # b = a^2
c = b * 3   # c = 3a^2

print(f"a: {a}")
print(f"b = a^2: {b}")
print(f"c = b*3: {c}")

# 역전파 실행
c.backward()

print(f"\nbackward() 실행 후:")
print(f"a.grad: {a.grad}")  # dc/da = 6a = 12

# 이론값 검증
print(f"이론값: dc/da = 6a = 6×{a.item()} = {6 * a.item()}")
```

**출력 예시**:
```
a: tensor([2.], requires_grad=True)
b = a^2: tensor([4.], grad_fn=<PowBackward0>)
c = b*3: tensor([12.], grad_fn=<MulBackward0>)

backward() 실행 후:
a.grad: tensor([12.])
이론값: dc/da = 6a = 6×2.0 = 12.0
```

#### Step 5.2.2: 벡터 입력에 대한 backward()

```python
import torch

# 벡터 입력
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()  # y = x1^2 + x2^2

print(f"x: {x}")
print(f"y = sum(x^2): {y.item()}")

# backward
y.backward()

print(f"\nbackward() 실행 후:")
print(f"x.grad: {x.grad}")  # 각 원소에 대한 gradient

# 이론값 검증: dy/dx = [2x1, 2x2] = [2, 4]
print(f"이론값: dy/dx = [2x1, 2x2] = [2×{x[0].item()}, 2×{x[1].item()}] = [2.0, 4.0]")
```

**출력 예시**:
```
x: tensor([1., 2.], requires_grad=True)
y = sum(x^2): 5.0

backward() 실행 후:
x.grad: tensor([2., 4.])
이론값: dy/dx = [2x1, 2x2] = [2×1.0, 2×2.0] = [2.0, 4.0]
```

### Step 5.3: grad 속성의 의미

**`grad`**: 계산된 gradient 값이 저장되는 속성

**의미**:
- `backward()` 호출 후 `grad` 속성에 gradient가 저장됨
- `grad`는 `requires_grad=True`인 텐서에만 존재
- 여러 번 `backward()`를 호출하면 gradient가 누적됨

#### Step 5.3.1: grad 속성 확인

```python
import torch

# 계산 그래프 구성
x = torch.tensor([1.0], requires_grad=True)
a = x * 2
b = a ** 2
c = b + 3

print(f"x: {x}")
print(f"a = x*2: {a}")
print(f"b = a^2: {b}")
print(f"c = b+3: {c}")

# backward 전 grad 확인
print(f"\nbackward() 전:")
print(f"x.grad: {x.grad}")  # None

# backward 실행
c.backward()

# backward 후 grad 확인
print(f"\nbackward() 후:")
print(f"x.grad: {x.grad}")  # gradient 값
```

#### Step 5.3.2: grad 초기화

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

# 첫 번째 backward
y = (x ** 2).sum()
y.backward()
print(f"첫 번째 backward 후: {x.grad}")

# grad 초기화
x.grad.zero_()
print(f"grad 초기화 후: {x.grad}")

# 두 번째 backward
z = (x ** 3).sum()
z.backward()
print(f"두 번째 backward 후: {x.grad}")
```

**출력 예시**:
```
첫 번째 backward 후: tensor([2., 4.])
grad 초기화 후: tensor([0., 0.])
두 번째 backward 후: tensor([3., 12.])
```

### Step 5.4: grad_fn으로 계산 그래프 확인

**`grad_fn`**: gradient를 계산한 함수에 대한 정보

**의미**:
- Leaf node: `grad_fn = None` (직접 생성된 텐서)
- Non-leaf node: `grad_fn ≠ None` (연산을 통해 생성된 텐서)

#### Step 5.4.1: grad_fn 체인 추적

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
a = x * 2
b = a ** 2
c = b + 3

print(f"계산 그래프:")
print(f"x (leaf): requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
print(f"a: grad_fn={type(a.grad_fn).__name__}")
print(f"b: grad_fn={type(b.grad_fn).__name__}")
print(f"c: grad_fn={type(c.grad_fn).__name__}")

# 역전파
c.backward()
print(f"\nx.grad: {x.grad}")
```

**출력 예시**:
```
계산 그래프:
x (leaf): requires_grad=True, grad_fn=None
a: grad_fn=MulBackward0
b: grad_fn=PowBackward0
c: grad_fn=AddBackward0

x.grad: tensor([4.])
```

---

## Step 6: no_grad()와 detach()

### Step 6.0: 기호 정리

- `no_grad()`: gradient 계산을 비활성화하는 컨텍스트 매니저
- `detach()`: gradient 추적을 중단하는 함수

### Step 6.1: no_grad()의 활용

**`no_grad()`**: gradient 계산이 필요없을 때 사용

**용도**:
- 모델 평가 시 (gradient 불필요)
- 메모리 사용량 감소
- 계산 속도 향상

#### Step 6.1.1: no_grad() 사용 예제

```python
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
    print(f"no_grad 블록 내: z = {z.item()}")

print(f"\nno_grad 블록 밖에서:")
z = (x ** 3).sum()
z.backward()
print(f"normal gradient: {x.grad}")
```

#### Step 6.1.2: 평가 모드에서 no_grad() 사용

```python
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

### Step 6.2: detach()의 활용

**`detach()`**: gradient 추적을 중단하는 함수

**의미**:
- 내용물(content)은 같지만 `requires_grad`가 다른 새로운 텐서 생성
- gradient 계산에 영향 없음

#### Step 6.2.1: detach() 사용 예제

```python
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

#### Step 6.2.2: detach() 실제 활용

```python
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

---

## Step 7: 실제 신경망에서의 활용

### Step 7.0: 기호 정리

- `nn.Linear`: 선형 레이어
- `nn.functional.mse_loss`: 평균 제곱 오차 손실 함수
- `optimizer`: 최적화 알고리즘 (SGD, Adam 등)

### Step 7.1: 간단한 모델에서의 gradient

```python
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

### Step 7.2: 다층 신경망에서의 gradient flow

```python
import torch
import torch.nn as nn

# 간단한 2층 신경망
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
x = torch.randn(10, 5)
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

### Step 7.3: Gradient Accumulation

**Gradient Accumulation**: 여러 배치의 gradient를 누적하여 한 번에 업데이트

```python
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

---

## Step 8: 계산 그래프의 상세 설명

### Step 8.1: 계산 그래프 구조

**계산 그래프**는 다음과 같은 구조를 가집니다:

```
x (입력, leaf node, grad_fn=None)
 ↓
LinearBackward (3→5 변환)
 ↓
h1 (중간 결과, non-leaf node, grad_fn=LinearBackward)
 ↓
ReLUBackward (활성화)
 ↓
h2 (중간 결과, non-leaf node, grad_fn=ReLUBackward)
 ↓
LinearBackward (5→1 변환)
 ↓
output (출력, non-leaf node, grad_fn=LinearBackward)
```

**각 화살표가 체인룰의 한 단계입니다!**

### Step 8.2: 역전파 과정

**`backward()`가 호출되면**:
1. 계산 그래프를 역순으로 순회
2. 각 노드에서 gradient 계산
3. 체인룰을 적용하여 다음 노드로 gradient 전파
4. Leaf node에 도달하면 `grad` 속성에 저장

**예시**:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

x = torch.randn(1, 3)
output = model(x)
loss = output.mean()

# Backward pass
loss.backward()

# 모든 레이어의 gradient 확인
print("모든 레이어의 gradient:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: gradient 계산됨, shape: {param.grad.shape}")
    else:
        print(f"{name}: gradient 없음")
```

**핵심**: Autograd는 각 연산의 합성 형태의 순서가 존재하고 그 연산의 원인이 존재하도록 그래프를 구성합니다.  
**`backward()`가 호출되면 이 그래프를 역순으로 순회하며 체인룰을 적용합니다!**

---

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

---

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

### 6. 체인룰의 적용
- 신경망은 여러 층의 합성 함수
- 체인룰을 반복 적용하여 모든 파라미터의 gradient 계산
- PyTorch의 autograd가 이를 자동으로 수행

다음 단원에서는 **데이터 준비**에 대해 배워보겠습니다.

