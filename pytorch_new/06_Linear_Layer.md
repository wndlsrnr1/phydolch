# 06. Linear Layer (선형 레이어)

이번 단원에서는 신경망의 기본 구성요소인 Linear Layer에 대해 자세히 배워보겠습니다.

## 학습 목표
- nn.Linear의 동작 원리
- 가중치와 편향의 형태 및 의미
- 아핀 변환의 기하학적 의미
- 행렬곱과 내적의 관계

## 이 단원을 배우기 전에

**이전 단원 (05) 복습: Layer, Module, Model의 차이를 말할 수 있나요?**

**이전 단원 (00, 02) 복습: 행렬이 선형 변환(공간을 변환하는 함수)이라는 것을 기억하시나요?**

## 이 단원 다음에는

**다음 단원 (07): Linear Layer를 이해했으니, 이미지 처리의 핵심인 Convolution Layer를 배워봅시다.**

## 왜 이 단원이 필요한가?

nn.Linear는 신경망의 가장 기본적인 레이어입니다. 거의 모든 신경망의 마지막 단계에서 사용됩니다.

- **분류기**: 최종 클래스 판단
- **기하학적 의미**: 벡터 공간의 변환
- **가중치/편향**: 학습의 핵심 파라미터
- **아핀 변환**: 선형 변환 + 평행이동

Linear Layer를 이해하면 신경망의 작동 원리를 깊이 있게 알 수 있습니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 수학 기호

- $x \in \mathbb{R}^{d_{in}}$: 입력 벡터 (한 샘플 기준)
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$: 가중치 행렬
- $b \in \mathbb{R}^{d_{out}}$: 편향 벡터
- $y \in \mathbb{R}^{d_{out}}$: 출력 벡터 (한 샘플 기준)
- $B$: 배치 크기 (batch size)
- $d_{in}$: 입력 특징 수 (input features)
- $d_{out}$: 출력 특징 수 (output features)

### PyTorch 코드에서의 변수

- `input`: 입력 텐서, shape는 $(B, d_{in})$ (배치 크기 $B$, 입력 특징 수 $d_{in}$)
- `weight`: 가중치 행렬, shape는 $(d_{out}, d_{in})$ (출력 특징 수 $d_{out}$, 입력 특징 수 $d_{in}$)
- `bias`: 편향 벡터, shape는 $(d_{out},)$
- `output`: 출력 텐서, shape는 $(B, d_{out})$
- `linear`: `nn.Linear` 레이어 객체

### 용어 정리

- **특징 차원(feature dimension)**: 한 샘플의 값 개수 (예: 입력이 10개면 특징 차원=10)
- **배치 차원(batch dimension)**: 샘플 묶음의 개수 (예: 배치 크기=32)
- **텐서 차원(rank)**: 배열의 축 개수 (예: `(B, 10)`은 2차원 텐서)

※ "입력 차원 10"은 **특징 10개**를 뜻함 (배치 축 아님)

---

## Step 1: 왜 Linear Layer가 필요한가?

### Step 1.1: 문제 상황

**신경망에서 필요한 변환**:

신경망은 입력 데이터를 다른 공간으로 변환하여 패턴을 학습합니다. 예를 들어:
- 이미지 분류: 784차원 픽셀 벡터 → 10차원 클래스 점수
- 텍스트 분류: 300차원 단어 임베딩 → 5차원 감정 점수

**문제**: 어떻게 입력을 원하는 차원으로 변환할까요?

### Step 1.2: 해결책: 선형 변환

**선형 변환(Linear Transformation)**은 벡터 공간을 다른 벡터 공간으로 변환하는 함수입니다:

$$y = Wx$$

여기서:
- $x \in \mathbb{R}^{d_{in}}$: 입력 벡터
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$: 변환 행렬
- $y \in \mathbb{R}^{d_{out}}$: 출력 벡터

**의미**: 입력을 $d_{in}$차원에서 $d_{out}$차원으로 변환합니다.

### Step 1.3: 아핀 변환 (Affine Transformation)

**아핀 변환**은 선형 변환 + 평행이동입니다:

$$y = Wx + b$$

여기서:
- $b \in \mathbb{R}^{d_{out}}$: 편향 벡터 (평행이동)

**왜 편향이 필요한가?**
- 선형 변환만으로는 원점을 이동할 수 없습니다
- 편향을 추가하면 기준점을 이동할 수 있어 더 유연한 변환이 가능합니다

### Step 1.4: 신경망에서의 역할

Linear Layer는 신경망에서 다음과 같은 역할을 합니다:

1. **특징 추출**: 입력을 새로운 공간으로 변환하여 패턴 발견
2. **차원 변환**: 입력 차원 → 출력 차원 (예: 784 → 10, 10 → 3)
3. **분류 결정**: 마지막 레이어에서 각 클래스에 대한 점수 출력

---

## Step 2: 수학적 정의

### Step 2.1: 선형 변환의 수학적 정의

**선형 변환** $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$은 다음을 만족합니다:

1. **덧셈 보존**: $T(x + y) = T(x) + T(y)$
2. **스칼라 곱 보존**: $T(\alpha x) = \alpha T(x)$

여기서 $x, y \in \mathbb{R}^n$, $\alpha \in \mathbb{R}$입니다.

### Step 2.2: 행렬로 표현하기

모든 선형 변환은 행렬로 표현할 수 있습니다:

$$T(x) = Ax$$

여기서 $A \in \mathbb{R}^{m \times n}$는 변환 행렬입니다.

**의미**: 행렬 곱셈으로 선형 변환을 구현할 수 있습니다.

### Step 2.3: 아핀 변환

**아핀 변환**은 선형 변환 + 평행이동입니다:

$$T(x) = Ax + b$$

여기서:
- $A \in \mathbb{R}^{m \times n}$: 선형 변환 행렬
- $b \in \mathbb{R}^{m}$: 편향 벡터 (평행이동)

**기하학적 의미**:
- $Ax$: 공간을 변환 (회전, 확대/축소, 반사 등)
- $+b$: 공간을 평행이동

### Step 2.4: Linear Layer의 수학적 정의

**Linear Layer**는 아핀 변환을 구현합니다:

$$y = Wx + b$$

여기서:
- $x \in \mathbb{R}^{d_{in}}$: 입력 벡터
- $W \in \mathbb{R}^{d_{out} \times d_{in}}$: 가중치 행렬
- $b \in \mathbb{R}^{d_{out}}$: 편향 벡터
- $y \in \mathbb{R}^{d_{out}}$: 출력 벡터

**각 출력 요소의 계산**:

$$y[j] = \sum_{i=1}^{d_{in}} W[j, i] \cdot x[i] + b[j]$$

**의미**: $j$번째 출력은 입력과 $j$번째 가중치 벡터의 **내적**에 편향을 더한 값입니다.

### Step 2.5: 행렬곱과 내적의 관계

**내적(Dot Product)**은 두 벡터의 유사성을 측정합니다:

$$\langle a, b \rangle = \sum_{i} a_i \cdot b_i$$

**행렬곱과 내적의 관계**:

$$y[j] = \sum_{i=1}^{d_{in}} W[j, i] \cdot x[i] = \langle W[j, :], x \rangle$$

**의미**: 
- 각 출력 $y[j]$는 가중치 벡터 $W[j, :]$와 입력 벡터 $x$의 **내적**입니다
- 내적이 크다 = 입력이 해당 가중치 패턴과 유사
- 내적이 작다 = 다른 패턴

**기하학적 해석**:
- 내적은 한 벡터를 다른 벡터 방향으로 **투영(projection)**한 길이입니다
- Linear Layer는 입력을 여러 가중치 벡터 방향으로 투영하여 새로운 표현을 만듭니다

---

## Step 3: PyTorch 구현

### Step 3.1: nn.Linear 기본 사용법

PyTorch의 `nn.Linear`는 아핀 변환을 구현합니다:

```python
import torch
import torch.nn as nn

# nn.Linear 레이어 생성
# in_features: 입력 특징 수
# out_features: 출력 특징 수
linear = nn.Linear(in_features=10, out_features=5)
print(f"Linear 레이어: {linear}")
```

**수학과 PyTorch의 대응**:
- 수학: $y = Wx + b$
- PyTorch: `output = input @ weight.T + bias`
- $W$ = `weight.T` (전치된 가중치 행렬)
- $b$ = `bias` (편향 벡터)
- $x$ = `input` (입력 벡터)
- $y$ = `output` (출력 벡터)

**왜 `weight.T`를 사용하는가?**
- PyTorch의 `weight`는 $(d_{out}, d_{in})$ 형태로 저장됩니다
- 수학적 수식 $y = Wx$에서 $W$는 $(d_{out}, d_{in})$ 형태입니다
- 하지만 PyTorch의 행렬곱 `@` 연산은 `(B, d_{in}) @ (d_{in}, d_{out})` 형태를 기대합니다
- 따라서 `weight.T`를 사용하여 $(d_{in}, d_{out})$ 형태로 변환합니다

### Step 3.2: 가중치와 편향 확인

```python
# 가중치와 편향 확인
print(f"\n가중치 shape: {linear.weight.shape}")
print(f"편향 shape: {linear.bias.shape}")
print(f"\n가중치 tensor:\n{linear.weight.data}")
print(f"\n편향 tensor:\n{linear.bias.data}")
```

**출력 예시**:
```
가중치 shape: torch.Size([5, 10])
편향 shape: torch.Size([5])
```

**의미**:
- `weight`: $(5, 10)$ - 출력 5개, 입력 10개
- `bias`: $(5,)$ - 출력 5개마다 편향 1개

### Step 3.3: Forward Pass

```python
# 입력 텐서 생성 및 forward pass
# 배치 크기 32, 특징 10개
input_tensor = torch.randn(32, 10)
print(f"입력 shape: {input_tensor.shape}")

# Linear 레이어 통과
output = linear(input_tensor)
print(f"출력 shape: {output.shape}")

# 수동으로 검증: output = input @ weight.T + bias
manual_output = input_tensor @ linear.weight.T + linear.bias
print(f"\n수동 계산 결과와 일치: {torch.allclose(output, manual_output)}")
```

**브로드캐스팅 설명**:
- `input_tensor`: $(32, 10)$
- `weight.T`: $(10, 5)$
- `bias`: $(5,)$
- `input @ weight.T`: $(32, 5)$
- `(input @ weight.T) + bias`: $(32, 5)$ (브로드캐스팅으로 bias가 각 배치에 더해짐)

### Step 3.4: 1차원 입력 처리

Linear Layer는 **1차원 입력만** 받을 수 있습니다. 따라서 2차원 이상의 텐서는 `.view()`를 통해 1차원으로 펼쳐야 합니다:

```python
# 2차원 이미지를 1차원으로 변환 예제
# 예: MNIST 이미지 (28, 28) → (784,)
image = torch.randn(1, 28, 28)  # 배치 1, 높이 28, 너비 28
print(f"원본 이미지 shape: {image.shape}")

# 1차원으로 펼치기
flattened = image.view(1, -1)  # (1, 784)
print(f"펼친 후 shape: {flattened.shape}")

# Linear Layer 적용
linear_mnist = nn.Linear(784, 10)  # 784 → 10 (10개 클래스)
output = linear_mnist(flattened)
print(f"출력 shape: {output.shape}")  # (1, 10)
```

**`.view()`의 의미**:
- 텐서의 shape를 변경하지만 데이터는 그대로 유지합니다
- `-1`은 나머지 차원을 자동으로 계산합니다
- `(1, 28, 28).view(1, -1)` → `(1, 784)`

### Step 3.5: 가중치 초기화

```python
# 가중치 초기화 예제
# Xavier 초기화 (주로 사용)
linear_xavier = nn.Linear(10, 5)
nn.init.xavier_uniform_(linear_xavier.weight)
print("Xavier 초기화된 가중치:")
print(linear_xavier.weight.data[:2, :3])  # 일부만 출력

# He 초기화 (ReLU 활성화 함수와 함께 사용)
linear_he = nn.Linear(10, 5)
nn.init.kaiming_uniform_(linear_he.weight, nonlinearity='relu')
print("\nHe 초기화된 가중치:")
print(linear_he.weight.data[:2, :3])  # 일부만 출력
```

**초기화의 중요성**:
- 잘못된 초기화는 학습이 어렵거나 불안정할 수 있습니다
- Xavier 초기화: 입력/출력 차원에 맞춰 가중치 분산 조정
- He 초기화: ReLU와 함께 사용 시 효과적

---

## Step 4: 구체적 예시

### Step 4.1: 수치 계산 예시

**문제 설정**:
- 입력: $x = [1, 2, 3]$ (3차원)
- 가중치: $W = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{bmatrix}$ (2×3)
- 편향: $b = [0.1, 0.2]$ (2차원)
- 목표: $y = Wx + b$ 계산

**단계별 계산**:

1. **행렬곱 계산**: $Wx$

$$Wx = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

$$= \begin{bmatrix} 0.1 \times 1 + 0.2 \times 2 + 0.3 \times 3 \\ 0.4 \times 1 + 0.5 \times 2 + 0.6 \times 3 \end{bmatrix}$$

$$= \begin{bmatrix} 0.1 + 0.4 + 0.9 \\ 0.4 + 1.0 + 1.8 \end{bmatrix} = \begin{bmatrix} 1.4 \\ 3.2 \end{bmatrix}$$

2. **편향 더하기**: $Wx + b$

$$y = \begin{bmatrix} 1.4 \\ 3.2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 3.4 \end{bmatrix}$$

**최종 결과**: $y = [1.5, 3.4]$

### Step 4.2: PyTorch로 검증

```python
import torch
import torch.nn as nn

# 수동으로 텐서 생성
x = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3) - 배치 크기 1
W = torch.tensor([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6]])   # (2, 3)
b = torch.tensor([0.1, 0.2])          # (2,)

# 수학적 계산: y = Wx + b
y_manual = x @ W.T + b
print(f"수동 계산 결과: {y_manual}")
# 출력: tensor([[1.5000, 3.4000]])

# nn.Linear로 구현
linear = nn.Linear(3, 2, bias=True)
# 가중치와 편향을 수동으로 설정
with torch.no_grad():
    linear.weight.data = W
    linear.bias.data = b

# Forward pass
y_pytorch = linear(x)
print(f"PyTorch 계산 결과: {y_pytorch}")
# 출력: tensor([[1.5000, 3.4000]])

# 검증
print(f"결과 일치: {torch.allclose(y_manual, y_pytorch)}")
# 출력: True
```

### Step 4.3: 내적의 의미 이해

```python
# 내적이 유사도를 측정한다는 것을 보여주는 예제
import torch

# 가중치 벡터 (패턴)
pattern = torch.tensor([1.0, 0.0, 0.0])  # 첫 번째 특징을 찾는 패턴

# 입력 벡터들
input1 = torch.tensor([1.0, 0.0, 0.0])  # 패턴과 동일
input2 = torch.tensor([0.0, 1.0, 0.0])  # 패턴과 다름
input3 = torch.tensor([0.5, 0.5, 0.0])  # 패턴과 부분적으로 유사

# 내적 계산 (유사도 측정)
similarity1 = torch.dot(pattern, input1)  # 1.0 (완전히 유사)
similarity2 = torch.dot(pattern, input2)  # 0.0 (전혀 다름)
similarity3 = torch.dot(pattern, input3)  # 0.5 (부분적으로 유사)

print(f"입력1과 패턴의 유사도: {similarity1:.2f}")
print(f"입력2와 패턴의 유사도: {similarity2:.2f}")
print(f"입력3과 패턴의 유사도: {similarity3:.2f}")
```

**의미**: 
- 내적이 크다 = 입력이 가중치 패턴과 유사
- 내적이 작다 = 입력이 가중치 패턴과 다름
- Linear Layer는 여러 가중치 패턴과의 유사도를 동시에 측정합니다

---

## Step 5: 가중치와 편향의 형태 이해

### Step 5.1: 왜 `weight`가 $(d_{out}, d_{in})$인가?

**한 뉴런 관점**:
- 1개 출력을 만들려면 (가중치 $d_{in}$개) + (편향 1개)가 필요합니다
- 이런 뉴런이 $d_{out}$개 → $W$는 $(d_{out}, d_{in})$, $b$는 $(d_{out},)$

**행렬 구조**:
- **행 = 출력 뉴런 수** ($d_{out}$)
- **열 = 입력 특징 수** ($d_{in}$)
- 각 행 $W[j, :]$은 "$j$번째 출력"을 만들기 위한 길이 $d_{in}$의 가중치 벡터

**예시**: `nn.Linear(10, 5)`
- 입력: 10개 특징
- 출력: 5개 뉴런
- 가중치: $(5, 10)$ - 5개 행(출력), 10개 열(입력)
- 편향: $(5,)$ - 출력 5개마다 편향 1개

### Step 5.2: 왜 `bias`가 $(d_{out},)$인가?

**출력이 $d_{out}$개이므로**, 각 출력에 y절편 1개씩 → `bias`는 $(d_{out},)$

**역할**: 
- 출력의 기준점을 평행이동합니다
- 입력이 0이어도 $y = b$가 가능합니다 (편향만큼 출력)

**예시**: `nn.Linear(10, 5)`
- 편향: $(5,)$ - 각 출력 뉴런마다 편향 1개
- $y[0] = \langle W[0, :], x \rangle + b[0]$
- $y[1] = \langle W[1, :], x \rangle + b[1]$
- $\vdots$
- $y[4] = \langle W[4, :], x \rangle + b[4]$

### Step 5.3: 배치 처리에서의 형태

**배치 입력**:
- 입력 `x`: $(B, d_{in})$ - 배치 크기 $B$, 입력 특징 수 $d_{in}$
- 가중치 `weight`: $(d_{out}, d_{in})$ - 출력 특징 수 $d_{out}$, 입력 특징 수 $d_{in}$
- 편향 `bias`: $(d_{out},)$ - 출력 특징 수 $d_{out}$

**연산**:
- `y = x @ weight.T + bias` → 결과 `y`: $(B, d_{out})$
- `weight.T`는 $(d_{in}, d_{out})$ 형태
- 브로드캐스팅으로 `bias`가 배치 전체에 더해짐

**예시**: 배치 크기 32, `nn.Linear(10, 5)`
- 입력: $(32, 10)$
- 가중치: $(5, 10)$
- 편향: $(5,)$
- 출력: $(32, 5)$ - 각 샘플마다 5차원 출력

### Step 5.4: 전체 그림

```
입력 (한 샘플): x: (10)
         ↓
    [ W: (5×10) ]
         ↓
    Wx: (5)
         ↓
    + b: (5)
         ↓
출력 (한 샘플): y: (5)
```

**배치 처리**:
```
입력 (배치): x: (B, 10)
         ↓
    [ weight.T: (10, 5) ]
         ↓
    x @ weight.T: (B, 5)
         ↓
    + bias: (5) [브로드캐스팅]
         ↓
출력 (배치): y: (B, 5)
```

---

## Step 6: 혼동 포인트 교정

### Step 6.1: "차원"의 두 가지 의미

**"차원"은 두 의미가 있습니다**:

1. **특징 개수** (공간상의 좌표 수)
   - 예: "입력 차원 10" = 입력 특징 10개
   - 수학: $\mathbb{R}^{10}$ 공간의 벡터

2. **텐서 축 개수** (배열 구조)
   - 예: `(32, 10)` = 2차원 텐서 (배치 축, 특징 축)
   - Python: `tensor.ndim` = 2

**주의**: "입력 차원 10"은 **특징 10개**를 뜻함 (배치 축 아님)

### Step 6.2: 수식과 코드의 대응

**수학 수식**: $y = Wx + b$

**PyTorch 코드**: `y = x @ weight.T + bias`

**대응 관계**:
- 수학: $W$ (가중치 행렬, $(d_{out}, d_{in})$)
- PyTorch: `weight` (가중치 텐서, $(d_{out}, d_{in})$)
- PyTorch 연산: `weight.T` (전치, $(d_{in}, d_{out})$)

**왜 전치가 필요한가?**
- 수학: $y = Wx$에서 $W$는 $(d_{out}, d_{in})$, $x$는 $(d_{in},)$
- PyTorch: `x @ weight.T`에서 `x`는 $(B, d_{in})$, `weight.T`는 $(d_{in}, d_{out})$
- 행렬곱 규칙: $(B, d_{in}) @ (d_{in}, d_{out}) = (B, d_{out})$

### Step 6.3: Bias의 의미

**`bias`는 단순 덧셈이 아니라 아핀 변환의 오프셋입니다**

**선형 변환만으로는**:
- 원점 $(0, 0)$을 이동할 수 없습니다
- $T(0) = A \cdot 0 = 0$ (항상 원점)

**아핀 변환 (Bias 추가)**:
- 원점을 이동할 수 있습니다
- $T(0) = A \cdot 0 + b = b$ (편향만큼 이동)

**의미**: 
- Bias는 출력 공간의 기준점을 설정합니다
- 입력이 0이어도 출력이 0이 아닐 수 있습니다

---

## 연습 문제

### 문제 1: Linear 구현
nn.Linear 없이 순수하게 행렬곱만으로 Linear를 구현해보세요.

```python
import torch

def manual_linear(input_tensor, weight, bias):
    """
    수동으로 Linear Layer 구현
    input_tensor: (B, d_in)
    weight: (d_out, d_in)
    bias: (d_out,)
    """
    # TODO: 구현하세요
    pass

# 검증
input_tensor = torch.randn(32, 10)
weight = torch.randn(5, 10)
bias = torch.randn(5)

output = manual_linear(input_tensor, weight, bias)
print(f"출력 shape: {output.shape}")  # (32, 5)이어야 함
```

### 문제 2: Shape 확인
nn.Linear(10, 5)의 weight shape는? 실제로 확인하세요.

### 문제 3: bias 효과
bias=False로 Linear를 만들고 bias=True와의 차이를 확인하세요.

```python
# bias=False
linear_no_bias = nn.Linear(10, 5, bias=False)
print(f"편향 없음: {linear_no_bias.bias}")  # None

# bias=True
linear_with_bias = nn.Linear(10, 5, bias=True)
print(f"편향 있음: {linear_with_bias.bias.shape}")  # (5,)

# 차이 확인
x = torch.zeros(1, 10)
output_no_bias = linear_no_bias(x)
output_with_bias = linear_with_bias(x)

print(f"편향 없을 때 출력: {output_no_bias}")  # 모두 0
print(f"편향 있을 때 출력: {output_with_bias}")  # 편향 값
```

### 문제 4: 내적 의미
가중치 벡터와 입력 벡터의 내적이 왜 '유사도'를 나타내는지 설명하세요.

### 문제 5: 아핀 변환
선형 변환과 아핀 변환의 차이를 코드로 보여주세요.

```python
# 선형 변환만 (bias=False)
linear_transform = nn.Linear(2, 2, bias=False)
x = torch.tensor([[0.0, 0.0]])
y_linear = linear_transform(x)
print(f"선형 변환 (0,0): {y_linear}")  # 항상 (0, 0)

# 아핀 변환 (bias=True)
affine_transform = nn.Linear(2, 2, bias=True)
y_affine = affine_transform(x)
print(f"아핀 변환 (0,0): {y_affine}")  # 편향 값
```

---

## 핵심 요약

### 1. nn.Linear 파라미터
- `in_features`: 입력 특징 수 ($d_{in}$)
- `out_features`: 출력 뉴런 수 ($d_{out}$)

### 2. Weight와 Bias 형태
- `weight`: $(d_{out}, d_{in})$ - 출력×입력
- `bias`: $(d_{out},)$ - 각 출력마다 1개

### 3. 수학적 의미
- $y = Wx + b$: 아핀 변환
- 행렬곱 = 내적 = 유사성/투영 측정
- Bias = 기준점 평행이동

### 4. PyTorch 구현
- `output = input @ weight.T + bias`
- `weight.T`는 전치된 가중치 행렬
- 브로드캐스팅으로 `bias`가 배치 전체에 더해짐

### 5. 기하학적 의미
- 각 출력은 입력과 가중치 벡터의 **내적** (유사도 측정)
- 내적이 크다 = 입력이 해당 가중치 패턴과 유사
- Linear Layer는 입력을 여러 가중치 벡터 방향으로 투영

다음 단원에서는 **Convolution Layer**에 대해 배워보겠습니다.

