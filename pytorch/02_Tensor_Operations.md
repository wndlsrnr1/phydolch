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

## 행렬곱과 내적의 기하학적 의미

**이전 단원(00) 복습**: 내적이 두 벡터의 정렬도를 측정한다는 것을 기억하시나요?

### 내적의 기하학적 의미

**내적 (Dot Product)**: `torch.dot(a, b)` 또는 `a @ b` (1차원 벡터)

- **수학적 정의**: $a \cdot b = a_1 b_1 + a_2 b_2 + \ldots$
- **기하학적 의미**: 두 벡터의 **정렬도(alignment)** 측정
- **딥러닝 연결**: 가중치 벡터와 입력 벡터가 얼마나 유사한 패턴인가?

### 행렬곱의 기하학적 의미

**행렬곱 (Matrix Multiplication)**: `torch.matmul(A, B)` 또는 `A @ B`

- **수학적 정의**: 각 행과 열의 내적
- **기하학적 의미**: **선형 변환 (Linear Transformation)**
  - 행렬은 공간을 변환하는 함수
  - 각 출력은 입력과 가중치 행렬의 행(row)의 내적
- **딥러닝 연결**: 
  - `nn.Linear`는 행렬곱: `output = input @ weight.T`
  - 각 출력 뉴런 = 입력과 가중치 벡터의 내적

**핵심**: 행렬곱의 각 원소는 내적이므로, "입력이 각 가중치 패턴과 얼마나 일치하는가"를 측정합니다!

### 텐서의 연산(Operations)

* 텐서에 대한 수학 연산, 삼각함수, 비트 연산, 비교 연산, 집계 등 제공

```python

```

```python

```

`max`와 `min`은 `dim` 인자를 줄 경우 argmax와 argmin도 함께 리턴
- argmax: 최대값을 가진 인덱스
- argmin: 최소값을 가진 인덱스

```python

```

```python

```

```python

```

`torch.add`: 덧셈

```python

```

결과 텐서를 인자로 제공

```python

```

`in-place` 방식
  - in-place방식으로 텐서의 값을 변경하는 연산 뒤에는 _''가 붙음
  - `x.copy_(y), x.t_()`

```python

```

`torch.sub`: 뺄셈

```python

```

`torch.mul`: 곱셉

```python

```

`torch.div`: 나눗셈

```python

```

`torch.mm`: 내적(dot product)

```python

```

### 브로드캐스팅(Broadcasting)

서로 다른 크기의 텐서끼리 연산할 때 자동으로 크기를 맞춰주는 기능

**규칙**:
1. 마지막 차원부터 비교
2. 크기가 같거나 한쪽이 1이면 브로드캐스팅 가능
3. 빈 차원은 1로 간주

**예시**:
- `(3, 1)` + `(1, 4)` → `(3, 4)`
- `(2, 3, 4)` + `(4,)` → `(2, 3, 4)`
- `(2, 3, 4)` + `(3, 1)` → `(2, 3, 4)`

```python
import torch

# 예제 1: (3, 1) + (1, 4) → (3, 4)
a = torch.tensor([[1], [2], [3]])  # shape: (3, 1)
b = torch.tensor([[10, 20, 30, 40]])  # shape: (1, 4)
result = a + b
print(f"a: {a.shape}")
print(f"b: {b.shape}")
print(f"a + b: {result.shape}")
print(result)

print("\n" + "="*50)

# 예제 2: 스칼라와 텐서 브로드캐스팅 (신경망에서 가장 많이 사용)
x = torch.randn(10, 3)  # shape: (10, 3)
bias = 1.5  # 스칼라
result2 = x + bias  # 모든 요소에 1.5가 더해짐
print(f"x: {x.shape}")
print(f"bias: 스칼라")
print(f"x + bias: {result2.shape}")

print("\n" + "="*50)

# 예제 3: 배치 단위 브로드캐스팅 (신경망 학습에서 필수)
# 배치 가중치 평균을 각 샘플에 더하는 경우
batch_features = torch.randn(32, 100)  # 배치 32개, 특징 100개
mean_features = torch.randn(100)  # 전체 평균 (100개 특징)
normalized = batch_features - mean_features  # broadcasting!
print(f"batch_features: {batch_features.shape}")
print(f"mean_features: {mean_features.shape}")
print(f"normalized: {normalized.shape}")
```

### 텐서의 조작(Manipulations)

인덱싱(Indexing): NumPy처럼 인덱싱 형태로 사용가능

```python

```

`view`: 텐서의 크기(size)나 모양(shape)을 변경

- 기본적으로 변경 전과 후에 텐서 안의 원소 개수가 유지되어야 함
- -1로 설정되면 계산을 통해 해당 크기값을 유추

```python

```

`item`: 텐서에 값이 단 하나라도 존재하면 숫자값을 얻을 수 있음

```python

```

스칼라값 하나만 존재해야 `item()` 사용 가능

```python

```

`squeeze`: 차원을 축소(제거)

```python

```

```python

```

`unsqueeze`: 차원을 증가(생성)

```python

```

```python

```

```python

```

`stack`: 텐서간 결합

```python

```

`cat`: 텐서를 결합하는 메소드(concatenate)

- 넘파이의 `stack`과 유사하지만, 쌓을 `dim`이 존재해야함
- 해당 차원을 늘려준 후 결합

```python

```

```python

```

```python

```

`chunk`: 텐서를 여러 개로 나눌 때 사용 (몇 개로 나눌 것인가?)

```python

```

`split`: `chunk`와 동일한 기능이지만 조금 다름 (텐서의 크기는 몇인가?)

```python

```

torch ↔ numpy
- Torch Tensor(텐서)를 NumPy array(배열)로 변환 가능
  - `numpy()`
  - `from_numpy()`
- Tensor가 CPU상에 있다면 NumPy 배열은 메모리 공간을 공유하므로 하나가 변하면, 다른 하나도 변함

```python

```

```python

```

```python

```

```python

```

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

다음 단원에서는 **자동 미분(Autograd)**에 대해 배워보겠습니다.
