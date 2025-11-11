# 06. Linear Layer (선형 레이어)

이번 단원에서는 신경망의 기본 구성요소인 Linear Layer에 대해 자세히 배워보겠습니다.

## 학습 목표
- nn.Linear의 동작 원리
- 가중치와 편향의 형태 및 의미
- 아핀 변환의 기하학적 의미
- 행렬곱과 내적의 관계

## 이 단원을 배우기 전에

**이전 단원 (05) 복습: Layer, Module, Model의 차이를 말할 수 있나요?**

## 이 단원 다음에는

**다음 단원 (07): Linear Layer를 이해했으니, 이미지 처리의 핵심인 Convolution Layer를 배워봅시다.**

## 왜 이 단원이 필요한가?

nn.Linear는 신경망의 가장 기본적인 레이어입니다. 거의 모든 신경망의 마지막 단계에서 사용됩니다.

- **분류기**: 최종 클래스 판단
- **기하학적 의미**: 벡터 공간의 변환
- **가중치/편향**: 학습의 핵심 파라미터
- **아핀 변환**: 선형 변환 + 평행이동

Linear Layer를 이해하면 신경망의 작동 원리를 깊이 있게 알 수 있습니다.

## 선형 변환의 기하학적 의미

**이전 단원(00, 02) 복습**: 행렬이 선형 변환(공간을 변환하는 함수)이라는 것을 기억하시나요?

### Linear Layer = 선형 변환

**nn.Linear**는 **행렬곱 + 편향**으로 이루어진 선형 변환입니다:

```python
output = input @ weight.T + bias
```

### 기하학적 해석

1. **행렬곱 (weight.T)**: 공간을 변환
   - 각 가중치 벡터는 하나의 "방향"을 정의
   - 입력 벡터가 이 방향으로 투영됨 (내적)
   
2. **편향 (bias)**: 공간을 평행이동
   - 기준점을 이동 (원점이 (0,0)이 아니라 (bias)로 이동)

3. **각 출력 뉴런의 의미**:
   - 입력과 가중치 벡터의 **내적** = 유사도 측정
   - 내적이 크다 = 입력이 해당 가중치 패턴과 유사
   - 내적이 작다 = 다른 패턴

### 시각적 이해

```
입력 벡터 ──[가중치 행렬]──> 변환된 벡터
  [x₁, x₂]        [w₁₁  w₁₂]        [y₁]
                 [w₂₁  w₂₂]        [y₂]
                 [w₃₁  w₃₂]        [y₃]
```

**의미**: 
- 입력을 **3차원 공간의 새로운 표현**으로 변환
- 각 출력은 입력이 어떤 "패턴"을 가지고 있는가를 측정

### 딥러닝에서의 역할

- **특징 추출**: 입력을 새로운 공간으로 변환하여 패턴 발견
- **차원 변환**: 입력 차원 → 출력 차원 (예: 784 → 10, 10 → 3)
- **분류 결정**: 마지막 레이어에서 각 클래스에 대한 점수 출력

### 선형 레이어(Linear layers)

1d만 가능하므로 `.view()`를 통해 1d로 펼쳐줘야함

```python

```

```python
import torch
import torch.nn as nn

# nn.Linear 레이어 생성
linear = nn.Linear(in_features=10, out_features=5)
print(f"Linear 레이어: {linear}")

# 가중치와 편향 확인
print(f"\n가중치 shape: {linear.weight.shape}")
print(f"편향 shape: {linear.bias.shape}")
print(f"\n가중치 tensor:\n{linear.weight.data}")
print(f"\n편향 tensor:\n{linear.bias.data}")
```

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

```python

```

```python

```

# Linear에서 가중치와 bias의 이해 (수식 표기 단순화 버전)
> **질문**  
> `weight: (5, 10)`, `bias: (5,)`는 왜 이런 모양인가?
---

## 1) 용어 정리
- **특징 차원(feature dimension)**: 한 샘플의 값 개수(예: 입력이 10개면 특징 차원=10).
- **배치 차원(batch dimension)**: 샘플 묶음의 개수(예: 배치 크기=32).
- **텐서 차원(rank)**: 배열의 축 개수(예: `(B, 10)`은 2차원 텐서).
※ “입력 차원 10”은 **특징 10개**를 뜻함(배치 축 아님).
---

## 2) 수학적 모델(선형/아핀 변환)
- 기본식: `y = W x + b`
- 모양(한 샘플 기준):
  - 입력 `x`: (10)
  - 가중치 `W`: (5, 10)
  - 편향 `b`: (5)
  - 출력 `y`: (5)
각 출력 `y[j]`는 다음처럼 계산됨:
y[j] = sum_i( W[j, i] * x[i] ) + b[j]
---

## 3) PyTorch 구현 대응
`nn.Linear(in_features=10, out_features=5)`는 배치 입력에서 이렇게 동작:
- 입력 `x`: (B, 10)
- 가중치 `weight`: (5, 10)
- 편향 `bias`: (5)
- 연산: `y = x @ weight.T + bias` → 결과 `y`: (B, 5)
(`weight.T`는 (10, 5). 브로드캐스팅으로 `bias`가 배치 전체에 더해짐.)
---

## 4) 왜 `weight`가 (5, 10)인가?
- **행 = 출력 뉴런 수(5)**, **열 = 입력 특징 수(10)**.
- 각 행 `W[j, :]`은 “j번째 출력”을 만들기 위한 길이 10의 가중치 벡터.
> 한 뉴런 관점:  
> 1개 출력을 만들려면 (가중치 10개) + (편향 1개).  
> 이런 뉴런이 5개 → `W`는 (5,10), `b`는 (5).
---

## 5) 왜 `bias`가 (5)인가?
- 출력이 5개이므로, 각 출력에 y절편 1개씩 → `bias`는 (5).
- 역할: 출력의 기준점을 평행이동(입력이 0이어도 `y = b`가 가능).
---

## 6) 행렬곱의 의미(내적·유사성·투영)
- 각 출력 `y[j]`는 `W[j, :]`와 `x`의 **내적** + `b[j]`.
- 내적은 두 벡터의 **유사성/투영 정도**를 나타냄.
- 결과: 입력 10개를 5개의 “방향(가중치 벡터)”으로 투영한 값.
---

## 7) 전체 그림
x: (10) ──[ W: (5×10) ]──> Wx: (5)
+ b: (5)
= y: (5)
---

## 8) 혼동 포인트 교정
- “차원”은 두 의미가 있음:
  - **특징 개수**(공간상의 좌표 수) vs **텐서 축 개수**(배열 구조).
- 수식 `y = W x + b` 와 코드 `y = x @ W.T + b`의 대응을 항상 명시.
- `bias`는 단순 덧셈이 아니라 **아핀 변환의 오프셋**.
---

## 9) 한 줄 결론
`nn.Linear(10, 5)`에서  
- `weight`는 “출력 5 × 입력 10” → (5, 10),  
- `bias`는 “출력마다 1개” → (5),  
- 각 출력은 “입력과 가중치의 내적(유사성)”에 “자기 편향”을 더해 생성된다.
---

## 10) 체크리스트
- 입력(한 샘플) 모양? → (10) / 배치면 (B, 10)  
- 출력(한 샘플) 모양? → (5) / 배치면 (B, 5)  
- `weight` 모양? → (5, 10)  
- `bias` 모양? → (5)  
- 연산? → `y = x @ weight.T + b`

## 연습 문제

### 문제 1: Linear 구현
nn.Linear 없이 순수하게 행렬곱만으로 Linear를 구현해보세요.

### 문제 2: Shape 확인
nn.Linear(10, 5)의 weight shape는? 실제로 확인하세요.

### 문제 3: bias 효과
bias=False로 Linear를 만들고 bias=True와의 차이를 확인하세요.

### 문제 4: 내적 의미
가중치 벡터와 입력 벡터의 내적이 왜 '유사도'를 나타내는지 설명하세요.

### 문제 5: 아핀 변환
선형 변환과 아핀 변환의 차이를 코드로 보여주세요.

## 핵심 요약

### 1. nn.Linear 파라미터
- `in_features`: 입력 특징 수
- `out_features`: 출력 뉴런 수

### 2. Weight와 Bias 형태
- `weight`: (out_features, in_features) - 출력×입력
- `bias`: (out_features,) - 각 출력마다 1개

### 3. 수학적 의미
- `y = Wx + b`: 아핀 변환
- 행렬곱 = 내적 = 유사성/투영 측정
- Bias = 기준점 평행이동

다음 단원에서는 **Convolution Layer**에 대해 배워보겠습니다.
