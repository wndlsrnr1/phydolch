# 02. 텐서 연산과 조작

이번 단원에서는 텐서의 다양한 연산과 조작 방법을 배워보겠습니다.

## 학습 목표
- 텐서의 기본 연산 (더하기, 곱하기, 행렬곱 등)
- 텐서의 형태 변환 (reshape, view, transpose 등)
- 브로드캐스팅 개념
- 인덱싱과 슬라이싱

## 이 단원을 배우기 전에

**이전 단원 (01) 복습: 텐서의 기본 개념과 초기화 방법을 아셨나요?**

## 이 단원 다음에는

**다음 단원 (03): 이제 텐서 연산을 배웠으니, autograd를 통해 gradient가 어떻게 계산되는지 배워봅시다.**

## 왜 이 단원이 필요한가?

텐서 연산은 신경망의 핵심입니다. 행렬곱, 더하기, 곱하기 등 모든 수학적 연산이 텐서로 이루어집니다.

- **행렬곱**: 신경망의 기본 연산 (y = Wx + b)
- **브로드캐스팅**: 효율적인 배치 처리
- **형태 변환**: reshape, transpose로 데이터 준비
- **인덱싱**: 필요한 데이터만 추출

이 단원을 마스터하면 신경망 구현 시 텐서 조작에 자신감을 갖게 됩니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 텐서 연산 관련 기호

- $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$: $n$차원 벡터 (1D 텐서)
- $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$: $m \times n$ 행렬 (2D 텐서)
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$: 가중치 행렬 (weight matrix)
- $\mathbf{x} \in \mathbb{R}^{d_{in}}$: 입력 벡터
- $\mathbf{y} \in \mathbb{R}^{d_{out}}$: 출력 벡터
- $\mathbf{b} \in \mathbb{R}^{d_{out}}$: 편향 벡터 (bias vector)
- $B$: 배치 크기 (batch size)
- $\cdot$: 내적 (dot product)
- $\odot$: 요소별 곱 (element-wise product)
- $\times$: 행렬곱 (matrix multiplication)

### PyTorch 코드에서의 변수

- `a`, `b`: 일반적인 텐서 변수
- `x`: 입력 텐서
- `y`: 출력 텐서
- `weight`: 가중치 텐서
- `bias`: 편향 텐서
- `dim`: 차원 (dimension)
- `shape`: 텐서의 크기
- `dtype`: 데이터 타입

---

## Step 1: 왜 텐서 연산이 필요한가?

### 문제 상황

**신경망의 기본 연산**:
- 선형 변환: $y = Wx + b$ (행렬곱과 덧셈)
- 활성화 함수: $z = \sigma(y)$ (요소별 연산)
- 손실 계산: $L = \frac{1}{2}\|y - \hat{y}\|^2$ (요소별 연산의 집계)

**배치 처리의 필요성**:
- 하나의 샘플만 처리하면 비효율적
- 여러 샘플을 동시에 처리해야 빠른 학습 가능
- 배치 크기 $B$인 경우: $Y = XW^T + \mathbf{b}$ (여기서 $X \in \mathbb{R}^{B \times d_{in}}$)

**메모리 효율성**:
- 브로드캐스팅으로 메모리 사용량 최소화
- 실제 데이터 복사 없이 연산 가능

### 해결책: 텐서 연산

PyTorch는 다음과 같은 텐서 연산을 제공합니다:
1. **요소별 연산**: 같은 크기의 텐서끼리 요소별로 연산
2. **행렬곱**: 선형 변환 구현
3. **브로드캐스팅**: 서로 다른 크기의 텐서끼리 자동으로 크기 맞춤
4. **집계 연산**: sum, mean, max, min 등
5. **형태 변환**: reshape, view, transpose 등

---

## Step 2: 내적과 행렬곱의 수학적 정의

**이전 단원(00) 복습**: 내적이 두 벡터의 정렬도를 측정한다는 것을 기억하시나요?

### Step 2.0: 기호 정리

- $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$: $n$차원 벡터
- $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{B} \in \mathbb{R}^{n \times p}$: 행렬
- $\mathbf{a} \cdot \mathbf{b}$: 벡터의 내적
- $\mathbf{A} \mathbf{B}$: 행렬곱

### Step 2.1: 내적의 수학적 정의

**내적 (Dot Product)**은 두 벡터의 대응되는 원소를 곱한 후 합한 값입니다:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**기하학적 의미**: 두 벡터의 **정렬도(alignment)** 측정
- 두 벡터가 같은 방향이면 내적이 큼
- 두 벡터가 수직이면 내적이 0
- 두 벡터가 반대 방향이면 내적이 음수

**딥러닝 연결**: 가중치 벡터와 입력 벡터가 얼마나 유사한 패턴인가?

#### Step 2.1.1: PyTorch에서의 내적

```python
import torch

# 벡터 정의
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 내적 계산
dot_product = torch.dot(a, b)
print(f"a: {a}")
print(f"b: {b}")
print(f"a · b: {dot_product.item()}")

# 수학적 계산: 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
print(f"이론값: 1×4 + 2×5 + 3×6 = {1*4 + 2*5 + 3*6}")
```

**출력 예시**:
```
a: tensor([1., 2., 3.])
b: tensor([4., 5., 6.])
a · b: 32.0
이론값: 1×4 + 2×5 + 3×6 = 32
```

### Step 2.2: 행렬곱의 수학적 정의

**행렬곱 (Matrix Multiplication)**은 각 행과 열의 내적로 정의됩니다:

$$\mathbf{C} = \mathbf{A} \mathbf{B}$$

여기서 $\mathbf{C} \in \mathbb{R}^{m \times p}$의 $(i, j)$ 원소는:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj} = \mathbf{A}_{i,:} \cdot \mathbf{B}_{:,j}$$

**의미**: 행렬곱의 각 원소는 $\mathbf{A}$의 $i$번째 행과 $\mathbf{B}$의 $j$번째 열의 내적입니다.

**기하학적 의미**: **선형 변환 (Linear Transformation)**
- 행렬은 공간을 변환하는 함수
- 각 출력은 입력과 가중치 행렬의 행(row)의 내적
- "입력이 각 가중치 패턴과 얼마나 일치하는가?"를 측정

#### Step 2.2.1: 행렬곱의 구체적 예시

**수학적 계산**:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

$$\mathbf{C} = \mathbf{A} \mathbf{B} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

$C_{11} = 1 \times 5 + 2 \times 7 = 5 + 14 = 19$
$C_{12} = 1 \times 6 + 2 \times 8 = 6 + 16 = 22$
$C_{21} = 3 \times 5 + 4 \times 7 = 15 + 28 = 43$
$C_{22} = 3 \times 6 + 4 \times 8 = 18 + 32 = 50$

$$\mathbf{C} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

#### Step 2.2.2: PyTorch에서의 행렬곱

```python
import torch

# 행렬 정의
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 행렬곱 계산
C = torch.matmul(A, B)
# 또는
C = A @ B

print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A @ B:\n{C}")

# 이론값 검증
expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
print(f"이론값:\n{expected}")
print(f"일치 여부: {torch.allclose(C, expected)}")
```

**출력 예시**:
```
A:
tensor([[1., 2.],
        [3., 4.]])
B:
tensor([[5., 6.],
        [7., 8.]])
A @ B:
tensor([[19., 22.],
        [43., 50.]])
이론값:
tensor([[19., 22.],
        [43., 50.]])
일치 여부: True
```

### Step 2.3: 선형 변환과의 연결

**선형 변환**은 다음과 같이 정의됩니다:

$$y = Wx + b$$

여기서:
- $x \in \mathbb{R}^{d_{in}}$: 입력 벡터
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$: 가중치 행렬
- $b \in \mathbb{R}^{d_{out}}$: 편향 벡터
- $y \in \mathbb{R}^{d_{out}}$: 출력 벡터

**의미**: 
- $Wx$는 행렬곱으로 입력을 다른 공간으로 변환
- $+b$는 편향을 더해 평행이동
- 각 출력 $y_i$는 입력 $x$와 가중치 행렬의 $i$번째 행의 내적

#### Step 2.3.1: 딥러닝에서의 활용

```python
import torch
import torch.nn as nn

# 선형 레이어 생성
linear = nn.Linear(in_features=3, out_features=2)

# 입력 벡터
x = torch.tensor([1.0, 2.0, 3.0])

# Forward pass: y = Wx + b
y = linear(x)

print(f"입력 x: {x}")
print(f"가중치 W:\n{linear.weight.data}")
print(f"편향 b: {linear.bias.data}")
print(f"출력 y: {y}")

# 수동 계산으로 검증
y_manual = x @ linear.weight.T + linear.bias
print(f"수동 계산: {y_manual}")
print(f"일치 여부: {torch.allclose(y, y_manual)}")
```

**핵심**: 행렬곱의 각 원소는 내적이므로, "입력이 각 가중치 패턴과 얼마나 일치하는가?"를 측정합니다!

---

## Step 3: 브로드캐스팅 (Broadcasting)

### Step 3.0: 기호 정리

- $A \in \mathbb{R}^{m \times n}$: $m \times n$ 행렬
- $B \in \mathbb{R}^{p \times q}$: $p \times q$ 행렬
- 브로드캐스팅: 크기가 다른 텐서끼리 연산을 가능하게 하는 메커니즘

### Step 3.1: 왜 브로드캐스팅이 필요한가?

**문제 상황**:
- 배치 처리 시 각 샘플에 같은 편향을 더하고 싶습니다
- 예: 배치 크기 32, 특징 수 100인 경우
  - 입력: $(32, 100)$
  - 편향: $(100,)$
  - 원하는 결과: 각 샘플에 편향을 더한 $(32, 100)$

**해결책 없이**:
- 편향을 $(32, 100)$으로 복사해야 함 → 메모리 낭비
- 반복문 사용 → 느린 연산

**브로드캐스팅으로**:
- 편향을 자동으로 $(1, 100)$으로 확장한 뒤 $(32, 100)$과 더함
- 실제 데이터 복사 없이 연산 가능 → 메모리 효율적

### Step 3.2: 브로드캐스팅의 수학적 의미

브로드캐스팅은 **암시적 확장(Implicit Expansion)**입니다:

**수학적으로**:
- $A \in \mathbb{R}^{m \times 1}$, $B \in \mathbb{R}^{1 \times n}$일 때
- $A + B$는 $A$를 $(m \times n)$으로, $B$를 $(m \times n)$으로 확장한 뒤 더합니다

**확장 규칙**:
1. 마지막 차원부터 비교
2. 크기가 같거나 한쪽이 1이면 브로드캐스팅 가능
3. 빈 차원은 1로 간주

### Step 3.3: 구체적 계산 과정

**예시**: `(3, 1)` + `(1, 4)` → `(3, 4)`

#### Step 3.3.1: 단계별 확장 과정

```python
import torch

# 원본 텐서
A = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
B = torch.tensor([[10, 20, 30, 40]])  # shape: (1, 4)

print("Step 1: 원본 텐서")
print(f"A (shape: {A.shape}):\n{A}")
print(f"B (shape: {B.shape}):\n{B}")

# 브로드캐스팅 결과
result = A + B
print(f"\nStep 2: 브로드캐스팅 결과 (shape: {result.shape})")
print(result)

# 확장 과정 설명
print("\nStep 3: 확장 과정")
print("A를 (3, 4)로 확장:")
A_expanded = A.expand(3, 4)
print(A_expanded)
print("\nB를 (3, 4)로 확장:")
B_expanded = B.expand(3, 4)
print(B_expanded)
print("\n더하기:")
print(A_expanded + B_expanded)
```

**출력 예시**:
```
Step 1: 원본 텐서
A (shape: torch.Size([3, 1])):
tensor([[1],
        [2],
        [3]])
B (shape: torch.Size([1, 4])):
tensor([[10, 20, 30, 40]])

Step 2: 브로드캐스팅 결과 (shape: torch.Size([3, 4]))
tensor([[11, 21, 31, 41],
        [12, 22, 32, 42],
        [13, 23, 33, 43]])

Step 3: 확장 과정
A를 (3, 4)로 확장:
tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
B를 (3, 4)로 확장:
tensor([[10, 20, 30, 40],
        [10, 20, 30, 40],
        [10, 20, 30, 40]])
더하기:
tensor([[11, 21, 31, 41],
        [12, 22, 32, 42],
        [13, 23, 33, 43]])
```

**수학적 표현**:
- $A$를 $(3, 4)$로 확장:
  $$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \rightarrow \begin{bmatrix} 1 & 1 & 1 & 1 \\ 2 & 2 & 2 & 2 \\ 3 & 3 & 3 & 3 \end{bmatrix}$$

- $B$를 $(3, 4)$로 확장:
  $$\begin{bmatrix} 10 & 20 & 30 & 40 \end{bmatrix} \rightarrow \begin{bmatrix} 10 & 20 & 30 & 40 \\ 10 & 20 & 30 & 40 \\ 10 & 20 & 30 & 40 \end{bmatrix}$$

- 더하기:
  $$\begin{bmatrix} 1 & 1 & 1 & 1 \\ 2 & 2 & 2 & 2 \\ 3 & 3 & 3 & 3 \end{bmatrix} + \begin{bmatrix} 10 & 20 & 30 & 40 \\ 10 & 20 & 30 & 40 \\ 10 & 20 & 30 & 40 \end{bmatrix} = \begin{bmatrix} 11 & 21 & 31 & 41 \\ 12 & 22 & 32 & 42 \\ 13 & 23 & 33 & 43 \end{bmatrix}$$

### Step 3.4: 신경망에서의 브로드캐스팅 활용

#### Step 3.4.1: 스칼라와 텐서 브로드캐스팅

```python
import torch

# 배치 데이터
x = torch.randn(10, 3)  # shape: (10, 3)
bias = 1.5  # 스칼라

# 브로드캐스팅: 모든 요소에 1.5가 더해짐
result = x + bias

print(f"입력 x (shape: {x.shape}):\n{x[:3, :]}")  # 처음 3개만 출력
print(f"bias: {bias} (스칼라)")
print(f"결과 (shape: {result.shape}):\n{result[:3, :]}")
print(f"검증: x[0, 0] + bias = {x[0, 0].item():.4f} + {bias} = {result[0, 0].item():.4f}")
```

#### Step 3.4.2: 배치 단위 브로드캐스팅

```python
import torch

# 배치 특징 (32개 샘플, 100개 특징)
batch_features = torch.randn(32, 100)  # shape: (32, 100)
mean_features = torch.randn(100)  # 전체 평균 (100개 특징)

# 브로드캐스팅: 각 샘플에서 평균을 빼서 정규화
normalized = batch_features - mean_features

print(f"batch_features (shape: {batch_features.shape})")
print(f"mean_features (shape: {mean_features.shape})")
print(f"normalized (shape: {normalized.shape})")
print(f"검증: normalized[0, 0] = {batch_features[0, 0].item():.4f} - {mean_features[0].item():.4f} = {normalized[0, 0].item():.4f}")
```

### Step 3.5: 브로드캐스팅 규칙 정리

**규칙**:
1. 마지막 차원부터 비교
2. 크기가 같거나 한쪽이 1이면 브로드캐스팅 가능
3. 빈 차원은 1로 간주

**예시**:
- `(3, 1)` + `(1, 4)` → `(3, 4)` ✓
- `(2, 3, 4)` + `(4,)` → `(2, 3, 4)` ✓ (빈 차원은 1로 간주)
- `(2, 3, 4)` + `(3, 1)` → `(2, 3, 4)` ✓
- `(3, 4)` + `(2, 4)` → 에러 ✗ (첫 번째 차원이 다름)

---

## Step 4: 텐서 연산 (Operations)

### Step 4.0: 기호 정리

- `+`, `-`, `*`, `/`: 요소별 연산 (element-wise operations)
- `@`: 행렬곱 (matrix multiplication)
- `sum`, `mean`, `max`, `min`: 집계 연산 (aggregation operations)
- `dim`: 집계할 차원

### Step 4.1: 요소별 연산

**요소별 연산**은 같은 크기의 텐서끼리 각 원소별로 연산합니다.

#### Step 4.1.1: 덧셈 (Addition)

```python
import torch

# 텐서 정의
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 덧셈
result = a + b
# 또는
result = torch.add(a, b)

print(f"a: {a}")
print(f"b: {b}")
print(f"a + b: {result}")
```

**출력 예시**:
```
a: tensor([1., 2., 3.])
b: tensor([4., 5., 6.])
a + b: tensor([5., 7., 9.])
```

#### Step 4.1.2: 결과 텐서를 인자로 제공

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
out = torch.empty_like(a)

# 결과를 out에 저장
torch.add(a, b, out=out)
print(f"a: {a}")
print(f"b: {b}")
print(f"out: {out}")
```

#### Step 4.1.3: In-place 방식

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# In-place 방식: a에 직접 더함
a.add_(b)
print(f"a (변경 후): {a}")
print(f"b: {b}")
```

**주의**: In-place 방식으로 텐서의 값을 변경하는 연산 뒤에는 `_`가 붙습니다 (예: `x.add_(y)`, `x.copy_(y)`, `x.t_()`)

#### Step 4.1.4: 뺄셈 (Subtraction)

```python
import torch

a = torch.tensor([5.0, 7.0, 9.0])
b = torch.tensor([1.0, 2.0, 3.0])

# 뺄셈
result = a - b
# 또는
result = torch.sub(a, b)

print(f"a: {a}")
print(f"b: {b}")
print(f"a - b: {result}")
```

#### Step 4.1.5: 곱셈 (Multiplication)

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 요소별 곱셈
result = a * b
# 또는
result = torch.mul(a, b)

print(f"a: {a}")
print(f"b: {b}")
print(f"a * b (요소별): {result}")

# 행렬곱과의 차이
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

element_wise = A * B  # 요소별 곱
matrix_mult = A @ B   # 행렬곱

print(f"\nA:\n{A}")
print(f"B:\n{B}")
print(f"A * B (요소별):\n{element_wise}")
print(f"A @ B (행렬곱):\n{matrix_mult}")
```

#### Step 4.1.6: 나눗셈 (Division)

```python
import torch

a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([2.0, 4.0, 5.0])

# 나눗셈
result = a / b
# 또는
result = torch.div(a, b)

print(f"a: {a}")
print(f"b: {b}")
print(f"a / b: {result}")
```

### Step 4.2: 집계 연산 (Aggregation Operations)

**집계 연산**은 텐서의 원소들을 집계하여 하나의 값 또는 차원이 줄어든 텐서를 반환합니다.

#### Step 4.2.1: 합 (Sum)

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 전체 합
total_sum = x.sum()
print(f"x:\n{x}")
print(f"전체 합: {total_sum.item()}")

# 차원별 합
sum_dim0 = x.sum(dim=0)  # 각 열의 합
sum_dim1 = x.sum(dim=1)  # 각 행의 합

print(f"dim=0 (각 열의 합): {sum_dim0}")
print(f"dim=1 (각 행의 합): {sum_dim1}")
```

**출력 예시**:
```
x:
tensor([[1., 2., 3.],
        [4., 5., 6.]])
전체 합: 21.0
dim=0 (각 열의 합): tensor([5., 7., 9.])
dim=1 (각 행의 합): tensor([ 6., 15.])
```

#### Step 4.2.2: 평균 (Mean)

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 전체 평균
total_mean = x.mean()
print(f"x:\n{x}")
print(f"전체 평균: {total_mean.item()}")

# 차원별 평균
mean_dim0 = x.mean(dim=0)  # 각 열의 평균
mean_dim1 = x.mean(dim=1)  # 각 행의 평균

print(f"dim=0 (각 열의 평균): {mean_dim0}")
print(f"dim=1 (각 행의 평균): {mean_dim1}")
```

#### Step 4.2.3: 최대값과 최소값 (Max and Min)

```python
import torch

x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

# 전체 최대값
max_val = x.max()
print(f"x:\n{x}")
print(f"전체 최대값: {max_val.item()}")

# 차원별 최대값
max_val_dim0, max_idx_dim0 = x.max(dim=0)  # 각 열의 최대값과 인덱스
max_val_dim1, max_idx_dim1 = x.max(dim=1)  # 각 행의 최대값과 인덱스

print(f"dim=0 (각 열의 최대값): {max_val_dim0}")
print(f"dim=0 (각 열의 최대값 인덱스): {max_idx_dim0}")
print(f"dim=1 (각 행의 최대값): {max_val_dim1}")
print(f"dim=1 (각 행의 최대값 인덱스): {max_idx_dim1}")

# 최소값
min_val = x.min()
min_val_dim0, min_idx_dim0 = x.min(dim=0)

print(f"\n전체 최소값: {min_val.item()}")
print(f"dim=0 (각 열의 최소값): {min_val_dim0}")
print(f"dim=0 (각 열의 최소값 인덱스): {min_idx_dim0}")
```

**주의**: `max`와 `min`은 `dim` 인자를 줄 경우 argmax와 argmin도 함께 리턴합니다.
- **argmax**: 최대값을 가진 인덱스
- **argmin**: 최소값을 가진 인덱스

---

## Step 5: 텐서 조작 (Manipulations)

### Step 5.0: 기호 정리

- `view`: 텐서의 크기를 변경 (메모리 공유)
- `reshape`: 텐서의 크기를 변경 (새 텐서 생성 가능)
- `transpose`: 차원 교환
- `squeeze`: 크기가 1인 차원 제거
- `unsqueeze`: 크기가 1인 차원 추가
- `stack`: 새 차원으로 텐서 쌓기
- `cat`: 특정 차원으로 텐서 연결
- `chunk`: 텐서를 여러 개로 나누기
- `split`: 텐서를 특정 크기로 나누기

### Step 5.1: 인덱싱과 슬라이싱

**인덱싱**은 NumPy와 동일한 문법을 사용합니다.

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"원본 텐서:\n{x}")

# 인덱싱
print(f"x[0]: {x[0]}")  # 첫 번째 행
print(f"x[0, 1]: {x[0, 1].item()}")  # 첫 번째 행, 두 번째 열
print(f"x[:, 1]: {x[:, 1]}")  # 두 번째 열
print(f"x[1:, :2]:\n{x[1:, :2]}")  # 두 번째 행부터, 처음 두 열
```

**출력 예시**:
```
원본 텐서:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
x[0]: tensor([1, 2, 3])
x[0, 1]: 2
x[:, 1]: tensor([2, 5, 8])
x[1:, :2]:
tensor([[4, 5],
        [7, 8]])
```

### Step 5.2: View와 Reshape

**`view`**: 텐서의 크기를 변경하지만 메모리는 공유합니다.

**조건**: 변경 전과 후에 텐서 안의 원소 개수가 유지되어야 합니다.

```python
import torch

x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"원본 텐서 (shape: {x.shape}):\n{x}")

# View로 크기 변경
x_view = x.view(8)  # 1D로 변환
print(f"view(8) (shape: {x_view.shape}): {x_view}")

x_view2 = x.view(4, 2)  # 2D로 변환
print(f"view(4, 2) (shape: {x_view2.shape}):\n{x_view2}")

# -1로 자동 계산
x_view3 = x.view(-1, 2)  # 첫 번째 차원은 자동 계산
print(f"view(-1, 2) (shape: {x_view3.shape}):\n{x_view3}")

# 메모리 공유 확인
x[0, 0] = 100
print(f"\n원본 변경 후:")
print(f"x:\n{x}")
print(f"x_view: {x_view}")  # x_view도 변경됨 (메모리 공유)
```

**출력 예시**:
```
원본 텐서 (shape: torch.Size([2, 4])):
tensor([[1, 2, 3, 4],
        [5, 6, 7, 8]])
view(8) (shape: torch.Size([8])): tensor([1, 2, 3, 4, 5, 6, 7, 8])
view(4, 2) (shape: torch.Size([4, 2])):
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
view(-1, 2) (shape: torch.Size([4, 2])):
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])

원본 변경 후:
x:
tensor([[100,   2,   3,   4],
        [  5,   6,   7,   8]])
x_view: tensor([100,   2,   3,   4,   5,   6,   7,   8])
```

**`reshape`**: `view`와 유사하지만 더 유연합니다. 메모리를 복사할 수 있습니다.

```python
import torch

x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"원본 텐서 (shape: {x.shape}):\n{x}")

# Reshape로 크기 변경
x_reshape = x.reshape(8)
print(f"reshape(8) (shape: {x_reshape.shape}): {x_reshape}")

x_reshape2 = x.reshape(4, 2)
print(f"reshape(4, 2) (shape: {x_reshape2.shape}):\n{x_reshape2}")
```

### Step 5.3: Item

**`item()`**: 텐서에 값이 단 하나만 존재하면 숫자값을 얻을 수 있습니다.

```python
import torch

# 스칼라 텐서
x = torch.tensor([5.0])
print(f"x: {x}")
print(f"x.item(): {x.item()}")

# 단일 원소 텐서
y = torch.tensor([[3.14]])
print(f"y: {y}")
print(f"y.item(): {y.item()}")

# 여러 원소가 있으면 에러
z = torch.tensor([1.0, 2.0, 3.0])
try:
    z.item()
except ValueError as e:
    print(f"에러: {e}")
```

**주의**: 스칼라값 하나만 존재해야 `item()` 사용 가능합니다.

### Step 5.4: Squeeze와 Unsqueeze

**`squeeze`**: 크기가 1인 차원을 제거합니다.

```python
import torch

# 크기가 1인 차원이 있는 텐서
x = torch.tensor([[1, 2, 3]])  # shape: (1, 3)
print(f"원본 텐서 (shape: {x.shape}): {x}")

# 모든 크기 1인 차원 제거
x_squeezed = x.squeeze()
print(f"squeeze() (shape: {x_squeezed.shape}): {x_squeezed}")

# 특정 차원만 제거
x2 = torch.tensor([[[1, 2, 3]]])  # shape: (1, 1, 3)
print(f"\n원본 텐서 (shape: {x2.shape}): {x2}")
x2_squeezed = x2.squeeze(dim=0)  # 첫 번째 차원 제거
print(f"squeeze(dim=0) (shape: {x2_squeezed.shape}): {x2_squeezed}")
```

**`unsqueeze`**: 크기가 1인 차원을 추가합니다.

```python
import torch

x = torch.tensor([1, 2, 3])  # shape: (3,)
print(f"원본 텐서 (shape: {x.shape}): {x}")

# 차원 추가
x_unsqueezed0 = x.unsqueeze(0)  # 첫 번째 차원에 추가
print(f"unsqueeze(0) (shape: {x_unsqueezed0.shape}):\n{x_unsqueezed0}")

x_unsqueezed1 = x.unsqueeze(1)  # 두 번째 차원에 추가
print(f"unsqueeze(1) (shape: {x_unsqueezed1.shape}):\n{x_unsqueezed1}")

x_unsqueezed_1 = x.unsqueeze(-1)  # 마지막 차원에 추가
print(f"unsqueeze(-1) (shape: {x_unsqueezed_1.shape}):\n{x_unsqueezed_1}")
```

### Step 5.5: Stack과 Cat

**`stack`**: 새 차원으로 텐서를 쌓습니다.

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.tensor([7, 8, 9])

print(f"a: {a}")
print(f"b: {b}")
print(f"c: {c}")

# dim=0으로 쌓기
stacked = torch.stack([a, b, c], dim=0)
print(f"stack([a, b, c], dim=0) (shape: {stacked.shape}):\n{stacked}")

# dim=1로 쌓기
stacked2 = torch.stack([a, b, c], dim=1)
print(f"stack([a, b, c], dim=1) (shape: {stacked2.shape}):\n{stacked2}")
```

**출력 예시**:
```
a: tensor([1, 2, 3])
b: tensor([4, 5, 6])
c: tensor([7, 8, 9])
stack([a, b, c], dim=0) (shape: torch.Size([3, 3])):
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
stack([a, b, c], dim=1) (shape: torch.Size([3, 3])):
tensor([[1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]])
```

**`cat`**: 특정 차원으로 텐서를 연결합니다.

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

print(f"a (shape: {a.shape}):\n{a}")
print(f"b (shape: {b.shape}):\n{b}")

# dim=0으로 연결 (행 방향)
cat_dim0 = torch.cat([a, b], dim=0)
print(f"cat([a, b], dim=0) (shape: {cat_dim0.shape}):\n{cat_dim0}")

# dim=1으로 연결 (열 방향)
cat_dim1 = torch.cat([a, b], dim=1)
print(f"cat([a, b], dim=1) (shape: {cat_dim1.shape}):\n{cat_dim1}")
```

**출력 예시**:
```
a (shape: torch.Size([2, 2])):
tensor([[1, 2],
        [3, 4]])
b (shape: torch.Size([2, 2])):
tensor([[5, 6],
        [7, 8]])
cat([a, b], dim=0) (shape: torch.Size([4, 2])):
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
cat([a, b], dim=1) (shape: torch.Size([2, 4])):
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
```

**차이점**: `stack`은 새 차원을 만들어 쌓고, `cat`은 기존 차원을 따라 연결합니다.

### Step 5.6: Chunk와 Split

**`chunk`**: 텐서를 여러 개로 나눕니다 (몇 개로 나눌 것인가?).

```python
import torch

x = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
print(f"원본 텐서 (shape: {x.shape}):\n{x}")

# 3개로 나누기
chunks = torch.chunk(x, 3, dim=1)
print(f"\nchunk(x, 3, dim=1):")
for i, chunk in enumerate(chunks):
    print(f"chunk {i} (shape: {chunk.shape}):\n{chunk}")
```

**출력 예시**:
```
원본 텐서 (shape: torch.Size([2, 6])):
tensor([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12]])
chunk(x, 3, dim=1):
chunk 0 (shape: torch.Size([2, 2])):
tensor([[1, 2],
        [7, 8]])
chunk 1 (shape: torch.Size([2, 2])):
tensor([[ 3,  4],
        [ 9, 10]])
chunk 2 (shape: torch.Size([2, 2])):
tensor([[ 5,  6],
        [11, 12]])
```

**`split`**: 텐서를 특정 크기로 나눕니다 (텐서의 크기는 몇인가?).

```python
import torch

x = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
print(f"원본 텐서 (shape: {x.shape}):\n{x}")

# 크기 2로 나누기
splits = torch.split(x, 2, dim=1)
print(f"\nsplit(x, 2, dim=1):")
for i, split in enumerate(splits):
    print(f"split {i} (shape: {split.shape}):\n{split}")
```

**출력 예시**:
```
원본 텐서 (shape: torch.Size([2, 6])):
tensor([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12]])
split(x, 2, dim=1):
split 0 (shape: torch.Size([2, 2])):
tensor([[1, 2],
        [7, 8]])
split 1 (shape: torch.Size([2, 2])):
tensor([[ 3,  4],
        [ 9, 10]])
split 2 (shape: torch.Size([2, 2])):
tensor([[ 5,  6],
        [11, 12]])
```

**차이점**: `chunk`는 나눌 개수를 지정하고, `split`은 각 조각의 크기를 지정합니다.

---

## Step 6: Torch ↔ NumPy 변환

### Step 6.0: 기호 정리

- `numpy()`: 텐서를 NumPy 배열로 변환
- `from_numpy()`: NumPy 배열을 텐서로 변환
- **중요**: CPU 텐서와 NumPy 배열은 메모리 공간을 공유합니다!

### Step 6.1: 텐서 → NumPy 배열

```python
import torch
import numpy as np

# 텐서 생성
x = torch.tensor([1, 2, 3, 4, 5])
print(f"텐서: {x}")
print(f"텐서 타입: {type(x)}")

# NumPy 배열로 변환
x_np = x.numpy()
print(f"NumPy 배열: {x_np}")
print(f"NumPy 배열 타입: {type(x_np)}")
```

**출력 예시**:
```
텐서: tensor([1, 2, 3, 4, 5])
텐서 타입: <class 'torch.Tensor'>
NumPy 배열: [1 2 3 4 5]
NumPy 배열 타입: <class 'numpy.ndarray'>
```

### Step 6.2: NumPy 배열 → 텐서

```python
import torch
import numpy as np

# NumPy 배열 생성
a = np.array([1, 2, 3, 4, 5])
print(f"NumPy 배열: {a}")
print(f"NumPy 배열 타입: {type(a)}")

# 텐서로 변환
a_tensor = torch.from_numpy(a)
print(f"텐서: {a_tensor}")
print(f"텐서 타입: {type(a_tensor)}")
```

### Step 6.3: 메모리 공유 확인

**중요**: CPU 텐서와 NumPy 배열은 메모리 공간을 공유하므로 하나가 변하면 다른 하나도 변합니다!

```python
import torch
import numpy as np

# 텐서 생성
x = torch.tensor([1, 2, 3, 4, 5])
print(f"원본 텐서: {x}")

# NumPy 배열로 변환
x_np = x.numpy()
print(f"NumPy 배열: {x_np}")

# 텐서 변경
x[0] = 100
print(f"\n텐서 변경 후:")
print(f"텐서: {x}")
print(f"NumPy 배열: {x_np}")  # NumPy 배열도 변경됨!

# NumPy 배열 변경
x_np[1] = 200
print(f"\nNumPy 배열 변경 후:")
print(f"텐서: {x}")  # 텐서도 변경됨!
print(f"NumPy 배열: {x_np}")
```

**출력 예시**:
```
원본 텐서: tensor([1, 2, 3, 4, 5])
NumPy 배열: [1 2 3 4 5]

텐서 변경 후:
텐서: tensor([100,   2,   3,   4,   5])
NumPy 배열: [100   2   3   4   5]

NumPy 배열 변경 후:
텐서: tensor([100, 200,   3,   4,   5])
NumPy 배열: [100 200   3   4   5]
```

**주의**: GPU 텐서는 NumPy 배열로 직접 변환할 수 없습니다. 먼저 CPU로 이동해야 합니다:

```python
import torch

# GPU 텐서 (GPU가 있는 경우)
if torch.cuda.is_available():
    x_gpu = torch.tensor([1, 2, 3, 4, 5]).cuda()
    print(f"GPU 텐서: {x_gpu}")
    
    # CPU로 이동 후 NumPy로 변환
    x_cpu = x_gpu.cpu()
    x_np = x_cpu.numpy()
    print(f"NumPy 배열: {x_np}")
```

---

## 연습 문제

### 문제 1: 요소별 연산 vs 행렬곱
`torch.mul`과 `torch.matmul`의 차이를 코드로 설명하세요.

### 문제 2: 브로드캐스팅
(3, 1) 텐서와 (1, 4) 텐서를 더하면 결과 shape는? 실제로 구현해보세요.

### 문제 3: reshape vs view
reshape와 view의 차이점을 설명하고, 어떤 경우에 view가 실패하는지 예를 들어보세요.

### 문제 4: 인덱싱
3D 텐서에서 마지막 차원만 평균내는 코드를 작성하세요.

### 문제 5: 텐서 조작
두 텐서를 dim=0으로 쌓아 4D 텐서를 만드는 코드를 작성하세요.

---

## 핵심 요약

이번 단원에서 우리는 다음을 배웠습니다:

### 1. 텐서 연산
- **요소별 연산**: `+`, `-`, `*`, `/` (같은 크기의 텐서)
- **행렬곱**: `@` 또는 `torch.matmul()` (내적)
- **기타**: `.sum()`, `.mean()`, `.max()` 등

### 2. 브로드캐스팅
- 서로 다른 크기의 텐서끼리 자동으로 크기를 맞춰 연산
- 예: (3, 1) + (1, 3) → (3, 3)
- 메모리 효율적: 실제로 데이터를 복사하지 않음

### 3. 형태 변환
- `.reshape()`: 새로운 형태로 변환
- `.view()`: 같은 데이터 다른 뷰 (메모리 공유)
- `.transpose()`: 차원 교환
- `.unsqueeze()` / `.squeeze()`: 차원 추가/제거

### 4. 인덱싱과 슬라이싱
- NumPy와 동일한 문법: `tensor[0]`, `tensor[:, 1]` 등
- 불리언 마스크: 조건에 맞는 요소 선택
- `.gather()`, `.index_select()`: 고급 인덱싱

### 5. 텐서 조작
- `.cat()`: 텐서 연결
- `.stack()`: 새 차원으로 쌓기
- `.chunk()` / `.split()`: 텐서 분할

### 6. Torch ↔ NumPy 변환
- `numpy()`: 텐서 → NumPy 배열
- `from_numpy()`: NumPy 배열 → 텐서
- CPU 텐서와 NumPy 배열은 메모리 공유

다음 단원에서는 **자동 미분(Autograd)**에 대해 배워보겠습니다.

