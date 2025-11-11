# 텐서 기초 (Tensor Basics)

![image.png](data:image/png;base64,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)

## 왜 이 단원이 필요한가?

텐서(Tensor)는 PyTorch의 기본 데이터 구조입니다. 모든 PyTorch 작업은 텐서를 다루는 것이고, 다양한 연산과 변환을 수행합니다.

- **NumPy와의 차이**: 텐서는 NumPy 배열과 유사하지만 GPU 가속 지원
- **다차원 데이터**: 이미지(3차원), 비디오(4차원), 배치(5차원) 표현
- **자동 미분**: 텐서가 autograd의 기본 단위

텐서를 자유자재로 다루지 못하면 이후 모든 학습이 어려워집니다.

## 이 단원을 배우기 전에

**이전 단원 (00) 복습: 행렬곱, 선형결합, SVD의 의미를 이해하셨나요?**

## 이 단원 다음에는

**다음 단원 (02): 텐서를 만들 수 있으니, 이제 텐서로 다양한 연산을 수행하는 방법을 배워봅시다.**

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 텐서 관련 기호

- $\mathbf{x} \in \mathbb{R}^n$: $n$차원 벡터 (1D 텐서)
- $\mathbf{X} \in \mathbb{R}^{m \times n}$: $m \times n$ 행렬 (2D 텐서)
- $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$: $k$차원 텐서
- `shape`: 텐서의 차원 크기를 나타내는 튜플
- `dtype`: 텐서의 데이터 타입 (예: `float32`, `int64`)
- `device`: 텐서가 저장된 장치 (예: `cpu`, `cuda:0`)

### PyTorch 코드에서의 변수

- `x`: 입력 텐서
- `y`: 출력 텐서
- `t`: 일반적인 텐서 변수
- `B`: 배치 크기 (batch size)
- `C`: 채널 수 (channel)
- `H`: 높이 (height)
- `W`: 너비 (width)

---

## Step 1: 왜 텐서가 필요한가?

### 문제 상황

**기존 방법의 한계**:
- NumPy 배열: CPU에서만 연산 가능
- Python 리스트: 느린 연산 속도
- 수동 미분: 복잡한 신경망에서 불가능

**딥러닝의 요구사항**:
- 대량의 데이터를 빠르게 처리
- GPU를 활용한 병렬 연산
- 자동 미분 (Autograd)을 통한 역전파
- 다양한 차원의 데이터 표현 (이미지, 텍스트, 시계열 등)

### 해결책: 텐서

텐서는 이러한 문제를 모두 해결합니다:
1. **GPU 가속**: CUDA를 통한 병렬 연산
2. **자동 미분**: 계산 그래프 추적으로 gradient 자동 계산
3. **다차원 데이터**: 스칼라부터 고차원 배열까지 통일된 표현

---

## Step 2: 벡터와 행렬에서 텐서로

**이전 단원(00) 복습**: 벡터와 행렬의 기하학적 의미를 이해하셨나요?

### 수학적 정의

텐서는 **벡터와 행렬을 일반화한 개념**입니다:

- **스칼라 (Scalar)**: 0차원 텐서 (하나의 숫자)
  - 수학: $a \in \mathbb{R}$
  - 예: $a = 3.14$

- **벡터 (Vector)**: 1차원 텐서 (숫자의 배열)
  - 수학: $\mathbf{x} \in \mathbb{R}^n$
  - 예: $\mathbf{x} = [1, 2, 3]^T$

- **행렬 (Matrix)**: 2차원 텐서 (행과 열)
  - 수학: $\mathbf{X} \in \mathbb{R}^{m \times n}$
  - 예: $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$

- **텐서 (Tensor)**: 3차원 이상 (행렬의 배열, 행렬의 배열의 배열, ...)
  - 수학: $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$
  - 예: 3D 텐서는 $d_1 \times d_2 \times d_3$ 크기의 배열

### 기하학적 연결

- **벡터** = 방향과 크기를 가진 화살표
- **행렬** = 선형 변환 (공간을 변환하는 함수)
- **텐서** = 다차원 공간에서의 데이터 표현

### 딥러닝에서의 의미

- **Linear Layer**: 행렬곱 (`input @ weight.T`)
  - 수학: $y = Wx + b$ (여기서 $W \in \mathbb{R}^{d_{out} \times d_{in}}$, $x \in \mathbb{R}^{d_{in}}$, $b \in \mathbb{R}^{d_{out}}$)
- **각 출력 뉴런**: 가중치 벡터와 입력 벡터의 내적
- **텐서 연산**: 여러 샘플(배치)을 한 번에 처리
  - 수학: $Y = XW^T + \mathbf{b}$ (여기서 $X \in \mathbb{R}^{B \times d_{in}}$, $Y \in \mathbb{R}^{B \times d_{out}}$)

---

## Step 3: 텐서란 무엇인가?

### PyTorch 텐서의 정의

**텐서(Tensor)**는 PyTorch의 기본 데이터 구조로, 다차원 배열을 나타냅니다.

**특징**:
- NumPy의 `ndarray`와 유사한 API
- GPU에서 연산 가능 (CUDA 지원)
- 자동 미분 지원 (Autograd)
- 동적 계산 그래프

### PyTorch 모듈 구조

PyTorch는 다음과 같은 주요 모듈로 구성됩니다:

```python
import torch
print(torch.__version__)  # PyTorch 버전 확인
```

---

## Step 4: 텐서 초기화

### Step 4.0: 기호 정리

- `shape`: 텐서의 크기를 나타내는 튜플 (예: `(3, 4)`)
- `dtype`: 데이터 타입 (예: `torch.float32`, `torch.int64`)
- `device`: 텐서가 저장된 장치 (예: `'cpu'`, `'cuda:0'`)

### Step 4.1: 왜 다양한 초기화 방법이 필요한가?

**문제 상황**:
- 랜덤 가중치: 신경망의 가중치는 보통 랜덤 값으로 초기화
- 0으로 초기화: 편향(bias)은 보통 0으로 시작
- 사용자 정의 값: 특정 값으로 초기화해야 하는 경우

**해결책**: PyTorch는 다양한 초기화 방법을 제공합니다.

### Step 4.2: 텐서 초기화 방법

#### 1. 초기화되지 않은 텐서

**왜 필요한가?**: 메모리만 할당하고 값을 설정하지 않아 빠른 초기화가 가능합니다.

```python
import torch

# 초기화되지 않은 텐서 생성 (값이 무작위로 채워짐)
x = torch.empty(3, 4)
print(f"초기화되지 않은 텐서:\n{x}")
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
```

**출력 예시**:
```
초기화되지 않은 텐서:
tensor([[ 0.0000e+00,  0.0000e+00, -1.0842e-19,  4.5697e-41],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]])
Shape: torch.Size([3, 4])
Data type: torch.float32
```

#### 2. 무작위로 초기화된 텐서

**왜 필요한가?**: 신경망의 가중치는 보통 랜덤 값으로 초기화합니다.

```python
# 정규분포에서 무작위로 초기화 (평균 0, 표준편차 1)
x = torch.randn(3, 4)
print(f"무작위 텐서 (정규분포):\n{x}")
print(f"평균: {x.mean():.4f}")
print(f"표준편차: {x.std():.4f}")
```

**출력 예시**:
```
무작위 텐서 (정규분포):
tensor([[-0.2345,  0.6789, -0.1234,  0.4567],
        [ 0.8901, -0.3456,  0.7890, -0.2345],
        [ 0.1234, -0.5678,  0.9012, -0.3456]])
평균: 0.1234
표준편차: 0.5678
```

#### 3. 0으로 채워진 텐서

**왜 필요한가?**: 편향(bias)은 보통 0으로 초기화합니다.

```python
# 0으로 채워진 텐서 (long 타입)
x = torch.zeros(3, 4, dtype=torch.long)
print(f"0으로 채워진 텐서 (long):\n{x}")
print(f"Data type: {x.dtype}")

# 0으로 채워진 텐서 (float 타입)
y = torch.zeros(3, 4, dtype=torch.float32)
print(f"\n0으로 채워진 텐서 (float):\n{y}")
print(f"Data type: {y.dtype}")
```

**출력 예시**:
```
0으로 채워진 텐서 (long):
tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])
Data type: torch.int64

0으로 채워진 텐서 (float):
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
Data type: torch.float32
```

#### 4. 사용자가 입력한 값으로 텐서 초기화

**왜 필요한가?**: 특정 값으로 초기화해야 하는 경우가 있습니다.

```python
# 리스트로부터 텐서 생성
x = torch.tensor([1, 2, 3, 4, 5])
print(f"리스트로부터 생성한 텐서: {x}")

# 2D 텐서 생성
y = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\n2D 텐서:\n{y}")
print(f"Shape: {y.shape}")
```

**출력 예시**:
```
리스트로부터 생성한 텐서: tensor([1, 2, 3, 4, 5])
2D 텐서:
tensor([[1, 2, 3],
        [4, 5, 6]])
Shape: torch.Size([2, 3])
```

#### 5. 1로 채워진 텐서

**왜 필요한가?**: 특정 연산에서 초기값으로 1이 필요한 경우가 있습니다.

```python
# 1로 채워진 텐서 (double 타입)
x = torch.ones(2, 4, dtype=torch.double)
print(f"1로 채워진 텐서 (double):\n{x}")
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
```

**출력 예시**:
```
1로 채워진 텐서 (double):
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]], dtype=torch.float64)
Shape: torch.Size([2, 4])
Data type: torch.float64
```

#### 6. 기존 텐서와 같은 크기로 초기화

**왜 필요한가?**: 기존 텐서와 같은 크기의 새 텐서를 만들 때 유용합니다.

```python
# 기존 텐서 생성
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"기존 텐서:\n{x}")
print(f"Shape: {x.shape}")

# 같은 크기로 무작위 텐서 생성
y = torch.randn_like(x)
print(f"\n같은 크기의 무작위 텐서:\n{y}")
print(f"Shape: {y.shape}")
```

**출력 예시**:
```
기존 텐서:
tensor([[1., 2., 3.],
        [4., 5., 6.]])
Shape: torch.Size([2, 3])

같은 크기의 무작위 텐서:
tensor([[ 0.1234, -0.5678,  0.9012],
        [-0.3456,  0.7890, -0.2345]])
Shape: torch.Size([2, 3])
```

#### 7. 텐서의 크기 계산

**왜 필요한가?**: 텐서의 크기를 확인하여 연산이 가능한지 확인할 수 있습니다.

```python
# 텐서 생성
x = torch.randn(3, 4, 5)
print(f"텐서:\n{x}")
print(f"\nShape: {x.shape}")
print(f"차원 수: {x.ndim}")
print(f"전체 원소 개수: {x.numel()}")
print(f"각 차원의 크기:")
for i, size in enumerate(x.shape):
    print(f"  차원 {i}: {size}")
```

**출력 예시**:
```
텐서:
tensor([[[-0.1234,  0.5678, ...],
         ...]])

Shape: torch.Size([3, 4, 5])
차원 수: 3
전체 원소 개수: 60
각 차원의 크기:
  차원 0: 3
  차원 1: 4
  차원 2: 5
```

---

## Step 5: 데이터 타입 (Data Type)

### Step 5.0: 기호 정리

- `dtype`: 데이터 타입 (예: `torch.float32`, `torch.int64`)
- `float32`: 32-bit 부동소수점 (기본값)
- `int64`: 64-bit 정수
- `bool`: 불리언 (True/False)

### Step 5.1: 왜 데이터 타입이 중요한가?

**문제 상황**:
- **메모리 효율성**: float64는 float32보다 2배 많은 메모리 사용
- **연산 속도**: 작은 데이터 타입일수록 빠른 연산
- **정확도**: float64가 float32보다 높은 정확도

**해결책**: 용도에 맞는 데이터 타입을 선택합니다.

### Step 5.2: 데이터 타입 종류

| Data type | dtype | CPU tensor | GPU tensor |
|-----------|-------|------------|------------|
| 32-bit floating point | `torch.float32` or `torch.float` | `torch.FloatTensor` | `torch.cuda.FloatTensor` |
| 64-bit floating point | `torch.float64` or `torch.double` | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 16-bit floating point | `torch.float16` or `torch.half` | `torch.HalfTensor` | `torch.cuda.HalfTensor` |
| 8-bit integer(unsigned) | `torch.uint8` | `torch.ByteTensor` | `torch.cuda.ByteTensor` |
| 8-bit integer(signed) | `torch.int8` | `torch.CharTensor` | `torch.cuda.CharTensor` |
| 16-bit integer(signed) | `torch.int16` or `torch.short` | `torch.ShortTensor` | `torch.cuda.ShortTensor` |
| 32-bit integer(signed) | `torch.int32` or `torch.int` | `torch.IntTensor` | `torch.cuda.IntTensor` |
| 64-bit integer(signed) | `torch.int64` or `torch.long` | `torch.LongTensor` | `torch.cuda.LongTensor` |

### Step 5.3: 데이터 타입 사용 예시

#### 예시 1: 데이터 타입 지정

```python
import torch

# float32로 텐서 생성 (기본값)
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(f"float32 텐서: {x}")
print(f"Data type: {x.dtype}")

# int64로 텐서 생성
y = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"\nint64 텐서: {y}")
print(f"Data type: {y.dtype}")
```

**출력 예시**:
```
float32 텐서: tensor([1., 2., 3.])
Data type: torch.float32

int64 텐서: tensor([1, 2, 3])
Data type: torch.int64
```

#### 예시 2: 데이터 타입 변환

**왜 필요한가?**: 연산에서 데이터 타입이 일치해야 합니다.

```python
# int64 텐서 생성
x = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"원본 텐서 (int64): {x}")
print(f"Data type: {x.dtype}")

# float32로 변환
y = x.float()
print(f"\n변환된 텐서 (float32): {y}")
print(f"Data type: {y.dtype}")

# 또는 .to() 메소드 사용
z = x.to(torch.float32)
print(f"\n.to() 메소드로 변환: {z}")
print(f"Data type: {z.dtype}")
```

**출력 예시**:
```
원본 텐서 (int64): tensor([1, 2, 3])
Data type: torch.int64

변환된 텐서 (float32): tensor([1., 2., 3.])
Data type: torch.float32

.to() 메소드로 변환: tensor([1., 2., 3.])
Data type: torch.float32
```

#### 예시 3: 데이터 타입 확인

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(f"텐서: {x}")
print(f"Data type: {x.dtype}")
print(f"is_floating_point: {x.is_floating_point()}")
print(f"is_integer: {x.is_integer()}")

y = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"\n텐서: {y}")
print(f"Data type: {y.dtype}")
print(f"is_floating_point: {y.is_floating_point()}")
print(f"is_integer: {y.is_integer()}")
```

**출력 예시**:
```
텐서: tensor([1., 2., 3.])
Data type: torch.float32
is_floating_point: True
is_integer: False

텐서: tensor([1, 2, 3])
Data type: torch.int64
is_floating_point: False
is_integer: True
```

#### 예시 4: 메모리 사용량 비교

```python
# float32 텐서
x_float32 = torch.randn(1000, 1000, dtype=torch.float32)
print(f"float32 텐서 메모리: {x_float32.element_size() * x_float32.numel() / 1024 / 1024:.2f} MB")

# float64 텐서
x_float64 = torch.randn(1000, 1000, dtype=torch.float64)
print(f"float64 텐서 메모리: {x_float64.element_size() * x_float64.numel() / 1024 / 1024:.2f} MB")
```

**출력 예시**:
```
float32 텐서 메모리: 3.81 MB
float64 텐서 메모리: 7.63 MB
```

---

## Step 6: CUDA Tensors

### Step 6.0: 기호 정리

- `device`: 텐서가 저장된 장치 (예: `'cpu'`, `'cuda:0'`)
- `CUDA`: Compute Unified Device Architecture (NVIDIA GPU 프로그래밍 플랫폼)

### Step 6.1: 왜 GPU가 필요한가?

**문제 상황**:
- **CPU 연산**: 순차적 연산, 느린 속도
- **대량의 데이터**: 신경망 학습에는 수천만 개의 파라미터 필요
- **시간 소요**: CPU로는 학습에 며칠이 걸릴 수 있음

**해결책**: GPU를 활용한 병렬 연산
- **GPU 연산**: 수천 개의 코어로 병렬 처리
- **속도 향상**: CPU 대비 10~100배 빠른 연산
- **메모리**: 대용량 데이터 처리 가능

### Step 6.2: CUDA 텐서 사용법

#### 예시 1: GPU 사용 가능 여부 확인

```python
import torch

# CUDA 사용 가능 여부 확인
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 디바이스 수: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"현재 CUDA 디바이스: {torch.cuda.current_device()}")
    print(f"디바이스 이름: {torch.cuda.get_device_name(0)}")
```

**출력 예시 (GPU 있는 경우)**:
```
CUDA 사용 가능: True
CUDA 디바이스 수: 1
현재 CUDA 디바이스: 0
디바이스 이름: NVIDIA GeForce RTX 3080
```

**출력 예시 (GPU 없는 경우)**:
```
CUDA 사용 가능: False
CUDA 디바이스 수: 0
```

#### 예시 2: 텐서를 GPU로 이동

```python
# CPU에서 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0])
print(f"CPU 텐서: {x}")
print(f"Device: {x.device}")

# GPU로 이동 (GPU가 있는 경우)
if torch.cuda.is_available():
    x_gpu = x.cuda()  # 또는 x.to('cuda')
    print(f"\nGPU 텐서: {x_gpu}")
    print(f"Device: {x_gpu.device}")
else:
    print("\nGPU를 사용할 수 없습니다.")
```

**출력 예시 (GPU 있는 경우)**:
```
CPU 텐서: tensor([1., 2., 3.])
Device: cpu

GPU 텐서: tensor([1., 2., 3.], device='cuda:0')
Device: cuda:0
```

#### 예시 3: .to() 메소드 사용

```python
# CPU에서 텐서 생성
x = torch.randn(3, 4)
print(f"원본 텐서 Device: {x.device}")

# GPU로 이동
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(f"GPU 텐서 Device: {x_gpu.device}")
    
    # 다시 CPU로 이동
    x_cpu = x_gpu.to('cpu')
    print(f"CPU 텐서 Device: {x_cpu.device}")
```

**출력 예시 (GPU 있는 경우)**:
```
원본 텐서 Device: cpu
GPU 텐서 Device: cuda:0
CPU 텐서 Device: cpu
```

#### 예시 4: GPU에서 직접 텐서 생성

```python
if torch.cuda.is_available():
    # GPU에서 직접 텐서 생성
    x = torch.randn(3, 4, device='cuda')
    print(f"GPU에서 생성한 텐서:\n{x}")
    print(f"Device: {x.device}")
else:
    print("GPU를 사용할 수 없습니다.")
```

**출력 예시 (GPU 있는 경우)**:
```
GPU에서 생성한 텐서:
tensor([[ 0.1234, -0.5678,  0.9012, -0.3456],
        [ 0.7890, -0.2345,  0.5678, -0.8901],
        [-0.1234,  0.4567, -0.7890,  0.2345]], device='cuda:0')
Device: cuda:0
```

---

## Step 7: 다차원 텐서 표현

### Step 7.0: 기호 정리

- $k$: 텐서의 차원 수
- $d_i$: $i$번째 차원의 크기
- $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$: $k$차원 텐서

### Step 7.1: 왜 다양한 차원이 필요한가?

**문제 상황**:
- **스칼라**: 단일 값
- **벡터**: 1D 데이터 (예: 시계열 데이터의 한 시점)
- **행렬**: 2D 데이터 (예: 이미지의 한 채널)
- **3D 텐서**: 다채널 이미지 (예: RGB 이미지)
- **4D 텐서**: 이미지 배치 (예: 여러 이미지를 한 번에 처리)
- **5D 텐서**: 비디오 데이터 (예: 여러 프레임의 이미지 시퀀스)

**해결책**: 차원에 따라 데이터를 구조화합니다.

### Step 7.2: 0D Tensor (Scalar)

**정의**: 하나의 숫자를 담고 있는 텐서

**수학적 표현**: $a \in \mathbb{R}$

**왜 필요한가?**: 손실 함수의 출력값, 스칼라 연산 결과 등을 나타냅니다.

```python
import torch

# 0D 텐서 생성 (스칼라)
x = torch.tensor(3.14)
print(f"0D 텐서 (스칼라): {x}")
print(f"Shape: {x.shape}")
print(f"차원 수: {x.ndim}")
print(f"값: {x.item()}")  # 스칼라 값을 Python 숫자로 변환
```

**출력 예시**:
```
0D 텐서 (스칼라): tensor(3.1400)
Shape: torch.Size([])
차원 수: 0
값: 3.140000104904175
```

### Step 7.3: 1D Tensor (Vector)

**정의**: 값들을 저장한 리스트와 유사한 텐서

**수학적 표현**: $\mathbf{x} \in \mathbb{R}^n$

**왜 필요한가?**: 시계열 데이터의 한 시점, 특징 벡터 등을 나타냅니다.

```python
# 1D 텐서 생성 (벡터)
x = torch.tensor([1, 2, 3, 4, 5])
print(f"1D 텐서 (벡터): {x}")
print(f"Shape: {x.shape}")
print(f"차원 수: {x.ndim}")
print(f"길이: {len(x)}")
```

**출력 예시**:
```
1D 텐서 (벡터): tensor([1, 2, 3, 4, 5])
Shape: torch.Size([5])
차원 수: 1
길이: 5
```

### Step 7.4: 2D Tensor (Matrix)

**정의**: 행렬과 같은 모양으로 두 개의 축이 존재

**수학적 표현**: $\mathbf{X} \in \mathbb{R}^{m \times n}$

**왜 필요한가?**: 
- 일반적인 수치, 통계 데이터셋
- 주로 샘플(samples)과 특성(features)을 가진 구조

**예시**: 
- $m$개 샘플, 각 샘플은 $n$개 특징
- 수학: $\mathbf{X} = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}$

```python
# 2D 텐서 생성 (행렬)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D 텐서 (행렬):\n{x}")
print(f"Shape: {x.shape}")
print(f"차원 수: {x.ndim}")
print(f"행 수: {x.shape[0]}")
print(f"열 수: {x.shape[1]}")
```

**출력 예시**:
```
2D 텐서 (행렬):
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
Shape: torch.Size([3, 3])
차원 수: 2
행 수: 3
열 수: 3
```

### Step 7.5: 3D Tensor

**정의**: 큐브(cube)와 같은 모양으로 세 개의 축이 존재

**수학적 표현**: $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times d_3}$

**왜 필요한가?**:
- 데이터가 연속된 시퀀스 데이터나 시간 축이 포함된 시계열 데이터
- 주식 가격 데이터셋, 시간에 따른 질병 발병 데이터 등
- 주로 샘플(samples), 타임스텝(timesteps), 특성(features)을 가진 구조

**예시**:
- $d_1$: 채널 수 (예: RGB = 3)
- $d_2$: 높이 (예: 28 픽셀)
- $d_3$: 너비 (예: 28 픽셀)
- 수학: $\mathcal{X} \in \mathbb{R}^{3 \times 28 \times 28}$ (RGB 이미지)

```python
# 3D 텐서 생성 (예: RGB 이미지)
# Shape: (채널, 높이, 너비) = (3, 28, 28)
x = torch.randn(3, 28, 28)
print(f"3D 텐서 Shape: {x.shape}")
print(f"차원 수: {x.ndim}")
print(f"채널 수: {x.shape[0]}")
print(f"높이: {x.shape[1]}")
print(f"너비: {x.shape[2]}")
print(f"전체 원소 개수: {x.numel()}")
```

**출력 예시**:
```
3D 텐서 Shape: torch.Size([3, 28, 28])
차원 수: 3
채널 수: 3
높이: 28
너비: 28
전체 원소 개수: 2352
```

### Step 7.6: 4D Tensor

**정의**: 4개의 축

**수학적 표현**: $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times d_3 \times d_4}$

**왜 필요한가?**:
- 컬러 이미지 데이터가 대표적인 사례
- 주로 샘플(samples), 높이(height), 너비(width), 컬러 채널(channel)을 가진 구조

**예시**:
- $d_1$: 배치 크기 (예: 32개 이미지)
- $d_2$: 채널 수 (예: RGB = 3)
- $d_3$: 높이 (예: 224 픽셀)
- $d_4$: 너비 (예: 224 픽셀)
- 수학: $\mathcal{X} \in \mathbb{R}^{32 \times 3 \times 224 \times 224}$ (32개 RGB 이미지 배치)

```python
# 4D 텐서 생성 (예: 이미지 배치)
# Shape: (배치, 채널, 높이, 너비) = (32, 3, 224, 224)
x = torch.randn(32, 3, 224, 224)
print(f"4D 텐서 Shape: {x.shape}")
print(f"차원 수: {x.ndim}")
print(f"배치 크기: {x.shape[0]}")
print(f"채널 수: {x.shape[1]}")
print(f"높이: {x.shape[2]}")
print(f"너비: {x.shape[3]}")
print(f"전체 원소 개수: {x.numel()}")
print(f"메모리 사용량: {x.element_size() * x.numel() / 1024 / 1024:.2f} MB")
```

**출력 예시**:
```
4D 텐서 Shape: torch.Size([32, 3, 224, 224])
차원 수: 4
배치 크기: 32
채널 수: 3
높이: 224
너비: 224
전체 원소 개수: 4816896
메모리 사용량: 18.38 MB
```

### Step 7.7: 5D Tensor

**정의**: 5개의 축

**수학적 표현**: $\mathcal{X} \in \mathbb{R}^{d_1 \times d_2 \times d_3 \times d_4 \times d_5}$

**왜 필요한가?**:
- 비디오 데이터가 대표적인 사례
- 주로 샘플(samples), 프레임(frames), 높이(height), 너비(width), 채널(channel)을 가진 구조

**예시**:
- 10초짜리 비디오 클립 100개가 있다면, 각 프레임이 240×320 RGB 이미지이고 초당 30프레임이라면
- $d_1$: 샘플 수 (100개)
- $d_2$: 프레임 수 (10초 × 30프레임/초 = 300프레임)
- $d_3$: 높이 (240 픽셀)
- $d_4$: 너비 (320 픽셀)
- $d_5$: 채널 (RGB = 3)
- 수학: $\mathcal{X} \in \mathbb{R}^{100 \times 300 \times 240 \times 320 \times 3}$
- Shape: (100, 300, 240, 320, 3)

```python
# 5D 텐서 예제: 비디오 데이터 시뮬레이션
# 가상의 비디오 데이터: 4개 비디오 클립, 각 5프레임, 64x64 흑백 이미지
# Shape: (4, 5, 64, 64, 1)
video_data = torch.randn(4, 5, 64, 64, 1)
print(f"5D 텐서 Shape: {video_data.shape}")
print(f"5D 텐서의 차원 수: {video_data.ndim}")

# 5D 텐서의 구조 분석
print("\n5D 텐서 구조 분석:")
print(f"- 배치 크기 (샘플 수): {video_data.shape[0]}")
print(f"- 시간 축 (프레임 수): {video_data.shape[1]}")
print(f"- 공간 축 (높이): {video_data.shape[2]}")
print(f"- 공간 축 (너비): {video_data.shape[3]}")
print(f"- 채널 (색상): {video_data.shape[4]}")
print(f"전체 원소 개수: {video_data.numel()}")
```

**출력 예시**:
```
5D 텐서 Shape: torch.Size([4, 5, 64, 64, 1])
5D 텐서의 차원 수: 5

5D 텐서 구조 분석:
- 배치 크기 (샘플 수): 4
- 시간 축 (프레임 수): 5
- 공간 축 (높이): 64
- 공간 축 (너비): 64
- 채널 (색상): 1
전체 원소 개수: 81920
```

---

## 연습 문제

### 문제 1: 텐서 생성

0으로 채워진 3×3 크기의 텐서를 3가지 방법으로 만들어보세요.

**해답**:
```python
import torch

# 방법 1: torch.zeros() 사용
x1 = torch.zeros(3, 3)
print(f"방법 1:\n{x1}")

# 방법 2: torch.tensor()와 리스트 사용
x2 = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
print(f"\n방법 2:\n{x2}")

# 방법 3: torch.empty() 후 fill_() 사용
x3 = torch.empty(3, 3)
x3.fill_(0)
print(f"\n방법 3:\n{x3}")
```

### 문제 2: 데이터 타입 변환

정수 텐서를 float 텐서로 변환하고, 그 차이를 설명하세요.

**해답**:
```python
# 정수 텐서 생성
x_int = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"정수 텐서: {x_int}")
print(f"Data type: {x_int.dtype}")

# float 텐서로 변환
x_float = x_int.float()
print(f"\nFloat 텐서: {x_float}")
print(f"Data type: {x_float.dtype}")

# 차이점:
# - int64: 정수만 저장 가능, 메모리 8 bytes
# - float32: 소수점 저장 가능, 메모리 4 bytes
```

### 문제 3: CUDA 사용

GPU가 있다면 같은 텐서를 GPU로 옮겨보고, 없다면 어떤 에러가 나는지 확인하세요.

**해답**:
```python
# CPU에서 텐서 생성
x = torch.randn(3, 3)
print(f"CPU 텐서 Device: {x.device}")

if torch.cuda.is_available():
    # GPU로 이동
    x_gpu = x.cuda()
    print(f"GPU 텐서 Device: {x_gpu.device}")
else:
    print("GPU를 사용할 수 없습니다.")
    # x.cuda()를 호출하면 에러가 발생합니다
```

### 문제 4: 차원 이해

다음 텐서들의 차원과 의미를 설명하세요:
- `torch.zeros(10)`
- `torch.zeros(5, 3)`
- `torch.zeros(2, 3, 4)`

**해답**:
```python
# 1D 텐서: 길이 10인 벡터
x1 = torch.zeros(10)
print(f"torch.zeros(10): Shape={x1.shape}, 차원={x1.ndim}D")
# 의미: 10개 원소를 가진 1차원 벡터

# 2D 텐서: 5행 3열 행렬
x2 = torch.zeros(5, 3)
print(f"torch.zeros(5, 3): Shape={x2.shape}, 차원={x2.ndim}D")
# 의미: 5개 샘플, 각 샘플은 3개 특징을 가진 데이터

# 3D 텐서: 2×3×4 크기
x3 = torch.zeros(2, 3, 4)
print(f"torch.zeros(2, 3, 4): Shape={x3.shape}, 차원={x3.ndim}D")
# 의미: 2개 채널, 각 채널은 3×4 크기의 이미지
```

### 문제 5: 텐서 정보 확인

텐서의 shape, dtype, device를 한 번에 확인하는 코드를 작성하세요.

**해답**:
```python
x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")

# 또는 한 번에 출력
print(f"\n텐서 정보:\n  Shape: {x.shape}\n  dtype: {x.dtype}\n  device: {x.device}")
```

---

## 핵심 요약

이번 단원에서 우리는 다음을 배웠습니다:

### 1. 텐서(Tensor)란?

- 다차원 배열로 PyTorch의 기본 데이터 구조
- NumPy와 유사하지만 GPU 가속 지원
- 스칼라(0차원), 벡터(1차원), 행렬(2차원), 그리고 고차원 텐서 가능

### 2. 텐서 초기화

- `torch.zeros()`: 0으로 초기화
- `torch.ones()`: 1로 초기화
- `torch.randn()`: 정규분포 난수
- `torch.tensor()`: 리스트/배열로부터 직접 생성
- `torch.empty()`: 초기화되지 않은 텐서

### 3. 데이터 타입

- `torch.float32` (기본): 32-bit 부동소수점
- `torch.int64`: 64-bit 정수
- `dtype` 파라미터로 타입 지정 가능
- `.float()`, `.to(torch.float32)` 등으로 타입 변환 가능

### 4. CUDA

- `.to(device)` 또는 `.cuda()`로 GPU 사용 가능
- GPU는 병렬 연산에 유리
- `torch.cuda.is_available()`로 GPU 사용 가능 여부 확인

### 5. 텐서의 차원

- **0D**: 스칼라 (하나의 숫자)
- **1D**: 벡터 (길이)
- **2D**: 행렬 (행, 열)
- **3D**: (채널, 높이, 너비) 예: RGB 이미지
- **4D**: (배치, 채널, 높이, 너비) 예: 이미지 배치
- **5D**: (배치, 프레임, 높이, 너비, 채널) 예: 비디오 데이터

---

다음 단원에서는 텐서의 연산과 조작에 대해 배워보겠습니다.

