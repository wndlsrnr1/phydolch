# 07. Convolution Layer (컨볼루션 레이어)

이번 단원에서는 이미지 처리의 핵심인 Convolution Layer에 대해 배워보겠습니다.

## 학습 목표
- Conv2d의 개념과 동작 원리
- Linear Layer와의 차이점
- 커널, stride, padding, dilation의 의미
- 출력 크기 계산
- 특징 추출기로서의 역할

## 이 단원을 배우기 전에

**이전 단원 (06) 복습: 가중치와 편향의 형태, 행렬곱의 기하학적 의미를 기억하나요?**

**이전 단원 (02) 복습: 텐서의 shape와 브로드캐스팅을 이해하시나요?**

## 이 단원 다음에는

**다음 단원 (08): Conv Layer를 배웠으니, Pooling, Activation 등 기타 레이어들을 알아봅시다.**

## 왜 이 단원이 필요한가?

Conv2d는 이미지 처리의 핵심입니다. 딥러닝이 이미지 인식에서 혁명을 일으킨 주역입니다.

- **이미지 특화**: 공간적 정보 활용
- **효율성**: 전역 연결보다 훨씬 적은 파라미터
- **패턴 탐지**: 가장자리, 형태, 질감 등 자동 추출
- **계층적 학습**: 저수준 → 고수준 특징

이미지 프로젝트라면 반드시 마스터해야 할 레이어입니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 수학 기호

- $H_{in}$, $W_{in}$: 입력 이미지의 높이, 너비 (픽셀 단위)
- $C_{in}$: 입력 채널 수 (예: RGB=3, 흑백=1)
- $C_{out}$: 출력 채널 수 (필터 개수)
- $K$: 커널 크기 (예: $K = 3$이면 3×3 커널)
- $P$: 패딩 크기 (예: $P = 1$이면 상하좌우 1픽셀씩 추가)
- $S$: 스트라이드 (예: $S = 2$이면 2픽셀씩 이동)
- $D$: 다이레이션 (예: $D = 2$이면 커널 내부 간격 2)
- $H_{out}$, $W_{out}$: 출력 이미지의 높이, 너비
- $x_{i,j}^{(c)}$: 입력 이미지의 $(i, j)$ 위치, $c$번째 채널의 픽셀 값
- $w_{u,v}^{(k,c)}$: $k$번째 필터의 $c$번째 채널, $(u, v)$ 위치의 가중치
- $y_{i,j}^{(k)}$: 출력 특징맵의 $(i, j)$ 위치, $k$번째 채널의 값
- $b_k$: $k$번째 필터의 편향

### PyTorch 코드에서의 변수

- `input`: 입력 텐서, shape는 $(B, C_{in}, H_{in}, W_{in})$ (배치, 채널, 높이, 너비)
- `weight`: 가중치 텐서, shape는 $(C_{out}, C_{in}, K, K)$ (출력 채널, 입력 채널, 커널 높이, 커널 너비)
- `bias`: 편향 텐서, shape는 $(C_{out},)$
- `output`: 출력 텐서, shape는 $(B, C_{out}, H_{out}, W_{out})$
- `conv`: `nn.Conv2d` 레이어 객체

### 용어 정리

- **커널(Kernel)**: 필터(Filter)와 동일한 의미, 이미지를 스캔하는 작은 창
- **특징맵(Feature Map)**: 하나의 필터가 생성하는 출력 이미지
- **합성곱(Convolution)**: 커널을 이미지 위에 슬라이딩하며 내적을 계산하는 연산
- **수용 영역(Receptive Field)**: 하나의 출력 픽셀이 "보는" 입력 영역

---

## Step 1: 왜 Conv2d가 필요한가?

### Step 1.1: Linear Layer의 한계

**Linear Layer의 문제점**:

1. **전역 연결**: 모든 입력 픽셀에 가중치를 적용
   - 예: 28×28 이미지 (784 픽셀) → 10 클래스
   - 파라미터 수: $784 \times 10 = 7,840$ (편향 제외)

2. **공간 정보 손실**: 이미지를 1차원 벡터로 평탄화
   - 픽셀 간의 위치 관계가 사라짐
   - 인접 픽셀의 관계를 학습하기 어려움

3. **파라미터 수 과다**: 큰 이미지에서 파라미터가 기하급수적으로 증가
   - 예: 224×224 이미지 (50,176 픽셀) → 1000 클래스
   - 파라미터 수: $50,176 \times 1000 = 50,176,000$ (약 5천만 개!)

### Step 1.2: 이미지의 공간적 특성

**이미지는 공간적 구조를 가집니다**:

- **국소 패턴**: 인접한 픽셀들이 관련이 있음
  - 예: 가장자리(edge), 형태(shape), 질감(texture)
- **평행 이동 불변성**: 같은 패턴이 이미지 어디에 있든 동일하게 인식되어야 함
- **계층적 특징**: 저수준 특징(가장자리) → 고수준 특징(눈, 코, 입)

**문제**: Linear Layer는 이러한 공간적 특성을 활용하지 못합니다.

### Step 1.3: 해결책: Convolution Layer

**Conv2d의 핵심 아이디어**:

1. **지역적 연결**: 작은 커널(예: 3×3)로 국소 영역만 봄
2. **가중치 공유**: 같은 커널을 이미지 전체에 적용 (평행 이동 불변성)
3. **공간 정보 보존**: 2차원 구조를 유지하며 처리

**장점**:
- **파라미터 수 감소**: 커널 크기가 고정되어 입력 크기와 무관
  - 예: 3×3 커널, 64개 필터 → $3 \times 3 \times 64 = 576$ 파라미터 (입력 크기와 무관!)
- **공간 정보 활용**: 인접 픽셀의 관계를 학습
- **계층적 특징 추출**: 여러 Conv Layer를 쌓아 저수준 → 고수준 특징 학습

### Step 1.4: Conv2d의 역할

**Conv2d는 특징 추출기(Feature Extractor)**입니다:

- **저수준 특징**: 가장자리, 색의 변화, 질감
- **고수준 특징**: 형태, 객체 부분 (여러 Conv Layer를 쌓으면)

**Linear Layer와의 비교**:

| 구분 | Linear | Conv2d |
|------|--------|--------|
| 입력 구조 | 1차원 벡터 | 2차원 이미지 |
| 연결 방식 | 모든 입력에 가중치 적용 | 지역적 연결(local receptive field) |
| 파라미터 수 | 매우 많음 (입력×출력) | 상대적으로 적음 (커널 고정 크기) |
| 특징 | 전역적 관계 학습 | 지역적 패턴 학습 |
| 활용 | 마지막 단계 (분류기) | 전단계 (특징 추출기) |

**비유**: 
- Conv2d는 일종의 **패턴 탐지기 (feature extractor)**
- Linear는 **판단자 (decision maker)**

---

## Step 2: 수학적 정의

### Step 2.1: 합성곱 연산의 수학적 정의

**2D 합성곱(2D Convolution)**은 다음과 같이 정의됩니다:

$$y_{i,j}^{(k)} = \sum_{c=1}^{C_{in}} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} w_{u,v}^{(k,c)} \cdot x_{i+u,j+v}^{(c)} + b_k$$

여기서:
- $y_{i,j}^{(k)}$: 출력 특징맵의 $(i, j)$ 위치, $k$번째 채널의 값
- $x_{i+u,j+v}^{(c)}$: 입력 이미지의 $(i+u, j+v)$ 위치, $c$번째 채널의 픽셀 값
- $w_{u,v}^{(k,c)}$: $k$번째 필터의 $c$번째 채널, $(u, v)$ 위치의 가중치
- $b_k$: $k$번째 필터의 편향
- $K$: 커널 크기 (예: $K=3$이면 3×3 커널)

**의미**:
1. 각 입력 채널($c$)에 대해 커널과 내적을 계산
2. 모든 입력 채널의 결과를 합산
3. 편향을 더하여 하나의 출력 채널($k$) 생성

### Step 2.2: 단일 채널 예시

**단일 채널 입력** ($C_{in} = 1$)인 경우:

$$y_{i,j} = \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} w_{u,v} \cdot x_{i+u,j+v} + b$$

**과정**:
1. 커널을 입력 이미지 위에 배치
2. 커널과 겹치는 영역의 픽셀과 가중치를 곱하여 합산
3. 편향을 더함
4. 커널을 이동하여 다음 위치 계산

**시각적 이해**:
```
입력 이미지 (5×5)        커널 (3×3)
[1  2  3  4  5]         [0.1  0.2  0.3]
[6  7  8  9 10]    *    [0.4  0.5  0.6]  =  출력 (3×3)
[11 12 13 14 15]        [0.7  0.8  0.9]
[16 17 18 19 20]
[21 22 23 24 25]
```

### Step 2.3: 다중 채널 예시

**다중 채널 입력** ($C_{in} = 3$, RGB 이미지)인 경우:

각 출력 채널 $k$에 대해:
$$y_{i,j}^{(k)} = \sum_{c=1}^{3} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} w_{u,v}^{(k,c)} \cdot x_{i+u,j+v}^{(c)} + b_k$$

**의미**:
- 각 입력 채널(R, G, B)에 대해 별도의 커널을 적용
- 모든 채널의 결과를 합산하여 하나의 출력 채널 생성
- $C_{out}$개의 필터가 있으면 $C_{out}$개의 출력 채널 생성

### Step 2.4: 내적과의 관계

**합성곱은 내적의 확장**입니다:

- Linear Layer: 전체 입력 벡터와 가중치 벡터의 내적
- Conv2d: 국소 영역(커널 크기)과 커널의 내적

**공통점**:
- 둘 다 "패턴과의 유사도"를 측정
- 내적이 크다 = 해당 패턴이 존재

**차이점**:
- Linear: 전역 패턴 (모든 입력 고려)
- Conv2d: 지역 패턴 (국소 영역만 고려)

---

## Step 3: 출력 크기 계산

### Step 3.1: 왜 출력 크기를 계산해야 하는가?

**문제 상황**:
- 모델 설계 시 출력 크기를 미리 알아야 메모리 할당 가능
- 여러 레이어를 쌓을 때 크기 변화를 추적해야 함
- 목표 출력 크기를 얻기 위해 파라미터 조정 필요

**해결책**: 출력 크기 공식을 이해하고 적용

### Step 3.2: 기본 공식 (패딩, 스트라이드, 다이레이션 없음)

**가장 간단한 경우** (padding=0, stride=1, dilation=1):

$$H_{out} = H_{in} - K + 1$$
$$W_{out} = W_{in} - K + 1$$

**이유**:
- 커널 크기 $K$만큼 경계에서 손실 발생
- 예: 입력 5×5, 커널 3×3 → 출력 3×3

### Step 3.3: 패딩 추가

**패딩(Padding)**은 입력 가장자리에 0을 추가합니다:

**패딩 적용 후 입력 크기**:
$$H_{padded} = H_{in} + 2P$$
$$W_{padded} = W_{in} + 2P$$

**패딩 적용 후 출력 크기** (stride=1, dilation=1):
$$H_{out} = (H_{in} + 2P) - K + 1 = H_{in} + 2P - K + 1$$
$$W_{out} = (W_{in} + 2P) - K + 1 = W_{in} + 2P - K + 1$$

**예시**: 입력 28×28, 커널 5×5, 패딩 2
$$H_{out} = 28 + 2 \times 2 - 5 + 1 = 28$$

**의미**: 패딩을 적절히 설정하면 입력 크기를 보존할 수 있습니다.

### Step 3.4: 스트라이드 추가

**스트라이드(Stride)**는 커널이 이동하는 간격입니다:

**스트라이드 적용 후 출력 크기** (padding=0, dilation=1):
$$H_{out} = \left\lfloor \frac{H_{in} - K}{S} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} - K}{S} \right\rfloor + 1$$

**이유**:
- 커널이 $S$픽셀씩 이동하므로 가능한 위치 수가 $1/S$로 감소
- $\lfloor \cdot \rfloor$는 내림 함수 (정수 나눗셈)

**예시**: 입력 28×28, 커널 5×5, 스트라이드 2
$$H_{out} = \left\lfloor \frac{28 - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{23}{2} \right\rfloor + 1 = 11 + 1 = 12$$

### Step 3.5: 다이레이션 추가

**다이레이션(Dilation)**은 커널 내부 간격을 확장합니다:

**유효 커널 크기**:
$$K_{eff} = K + (K - 1)(D - 1) = D(K - 1) + 1$$

**이유**:
- 다이레이션 $D$는 커널 요소 사이를 $D-1$만큼 띄움
- 예: $K=3$, $D=2$ → 유효 크기 $2(3-1)+1 = 5$ (실제로는 3×3이지만 5×5 영역을 봄)

**다이레이션 적용 후 출력 크기** (padding=0, stride=1):
$$H_{out} = H_{in} - K_{eff} + 1 = H_{in} - (D(K-1) + 1) + 1 = H_{in} - D(K-1)$$
$$W_{out} = W_{in} - D(K-1)$$

### Step 3.6: 전체 공식 (모든 파라미터 포함)

**최종 출력 크기 공식**:

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

**각 파라미터의 영향**:
- $P$ (패딩): 출력 크기 증가 (경계 보존)
- $S$ (스트라이드): 출력 크기 감소 (다운샘플링)
- $D$ (다이레이션): 수용 영역 확대 (출력 크기는 감소)

**예시**: 입력 28×28, 커널 5, 패딩 2, 스트라이드 1, 다이레이션 1
$$H_{out} = \left\lfloor \frac{28 + 2 \times 2 - 1 \times (5-1) - 1}{1} \right\rfloor + 1$$
$$= \left\lfloor \frac{28 + 4 - 4 - 1}{1} \right\rfloor + 1 = \left\lfloor 27 \right\rfloor + 1 = 28$$

---

## Step 4: PyTorch 구현

### Step 4.1: nn.Conv2d 기본 사용법

```python
import torch
import torch.nn as nn

# nn.Conv2d 레이어 생성
# in_channels: 입력 채널 수
# out_channels: 출력 채널 수 (필터 개수)
# kernel_size: 커널 크기
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
print(f"Conv2d 레이어: {conv}")
```

**주요 파라미터**:
- `in_channels`: 입력 채널 수 (RGB=3, 흑백=1)
- `out_channels`: 필터 수 = 추출할 특징 종류
- `kernel_size`: 필터의 크기 = 지역적 패턴 범위
- `stride`: 이동 간격 = 출력 해상도 제어 (기본값: 1)
- `padding`: 가장자리 0 추가 = 경계 정보 보존 (기본값: 0)
- `dilation`: 필터 간격 확장 = 넓은 문맥 파악 (기본값: 1)

### Step 4.2: 다양한 파라미터 조합 예제

```python
import torch
import torch.nn as nn

# 입력 텐서 생성: 배치 1, 채널 3, 28x28 이미지
input_tensor = torch.randn(1, 3, 28, 28)
print(f"입력 shape: {input_tensor.shape}")

# 예제 1: 기본 (padding=0, stride=1, dilation=1)
conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0, stride=1, dilation=1)
output1 = conv1(input_tensor)
print(f"출력 (기본): {output1.shape}")
# 출력: torch.Size([1, 16, 24, 24])
# 계산: ⌊(28 + 2×0 - 1×(5-1) - 1) / 1⌋ + 1 = ⌊23⌋ + 1 = 24

# 예제 2: padding 추가 (크기 보존)
conv2 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1, dilation=1)
output2 = conv2(input_tensor)
print(f"출력 (padding=2): {output2.shape}")
# 출력: torch.Size([1, 16, 28, 28])
# 계산: ⌊(28 + 2×2 - 1×(5-1) - 1) / 1⌋ + 1 = ⌊27⌋ + 1 = 28

# 예제 3: stride 적용 (다운샘플링)
conv3 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2, dilation=1)
output3 = conv3(input_tensor)
print(f"출력 (stride=2): {output3.shape}")
# 출력: torch.Size([1, 16, 14, 14])
# 계산: ⌊(28 + 2×2 - 1×(5-1) - 1) / 2⌋ + 1 = ⌊13⌋ + 1 = 14

# 예제 4: dilation 적용 (수용 영역 확대)
conv4 = nn.Conv2d(3, 16, kernel_size=5, padding=4, stride=1, dilation=2)
output4 = conv4(input_tensor)
print(f"출력 (dilation=2): {output4.shape}")
# 출력: torch.Size([1, 16, 28, 28])
# 계산: ⌊(28 + 2×4 - 2×(5-1) - 1) / 1⌋ + 1 = ⌊27⌋ + 1 = 28
```

### Step 4.3: Weight와 Bias 확인

```python
# Conv2d 레이어 생성
layer = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
print(f"Weight shape: {layer.weight.shape}")
print(f"Bias shape: {layer.bias.shape}")

# 출력 예시:
# Weight shape: torch.Size([20, 1, 5, 5])
# Bias shape: torch.Size([20])
```

**Weight shape 의미**:
- `(out_channels, in_channels, kernel_height, kernel_width)`
- 예: `(20, 1, 5, 5)` = 20개 필터, 각 필터는 1채널 5×5 커널

**Bias shape 의미**:
- `(out_channels,)`
- 예: `(20,)` = 20개 필터마다 편향 1개

### Step 4.4: Weight 시각화

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Conv2d 레이어 생성
layer = nn.Conv2d(1, 20, 5, 1)

# Weight 추출 (detach()를 통해 gradient 추적 해제)
weight = layer.weight.detach()
print(f"Weight shape: {weight.shape}")  # (20, 1, 5, 5)

# NumPy로 변환 (detach() 후 가능)
weight_numpy = weight.numpy()
print(f"NumPy shape: {weight_numpy.shape}")  # (20, 1, 5, 5)

# 첫 번째 필터의 가중치 시각화
first_filter = weight[0, 0, :, :]  # (5, 5)
plt.imshow(first_filter.numpy(), cmap='jet')
plt.colorbar()
plt.title('첫 번째 필터의 가중치')
plt.show()
```

**주의**: `weight`는 `detach()`를 통해 꺼내줘야 `numpy()` 변환이 가능합니다.

### Step 4.5: 다양한 파라미터 조합

```python
import torch
import torch.nn as nn

# 입력: 배치 20, 채널 16, 높이 50, 너비 100
input = torch.randn(20, 16, 50, 100)
print(f"입력 shape: {input.shape}")

# 예제 1: 기본 Conv2d
m1 = nn.Conv2d(16, 33, 3, stride=2)
output1 = m1(input)
print(f"출력 1: {output1.shape}")
# 계산: ⌊(50 + 2×0 - 1×(3-1) - 1) / 2⌋ + 1 = ⌊23.5⌋ + 1 = 24
#       ⌊(100 + 2×0 - 1×(3-1) - 1) / 2⌋ + 1 = ⌊48.5⌋ + 1 = 49

# 예제 2: 비대칭 커널 및 stride
m2 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
output2 = m2(input)
print(f"출력 2: {output2.shape}")

# 예제 3: dilation 추가
m3 = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
output3 = m3(input)
print(f"출력 3: {output3.shape}")
```

**파라미터 설명**:
- `dilation=(3, 1)`: 커널 내부 요소 사이를 세로로 3칸, 가로로 1칸 띄워서 적용
- `padding=(4, 2)`: 입력 가장자리 밖에 세로 4줄, 가로 2줄 추가하여 경계 손실을 줄이고 원하는 출력 크기를 유지/조정
- `stride=(2, 1)`: 커널이 이동하는 간격으로 세로 방향은 한 번에 2칸, 가로 방향은 1칸씩 이동

---

## Step 5: 구체적 예시

### Step 5.1: 출력 크기 계산 예시

**문제 설정**:
- 입력: 28×28 이미지
- 커널: 5×5
- 패딩: 2
- 스트라이드: 1
- 다이레이션: 1

**단계별 계산**:

1. **패딩 적용 후 입력 크기**:
   $$H_{padded} = 28 + 2 \times 2 = 32$$
   $$W_{padded} = 28 + 2 \times 2 = 32$$

2. **유효 커널 크기** (다이레이션=1이므로 커널 크기 그대로):
   $$K_{eff} = 5$$

3. **출력 크기 계산**:
   $$H_{out} = \left\lfloor \frac{28 + 2 \times 2 - 1 \times (5-1) - 1}{1} \right\rfloor + 1$$
   $$= \left\lfloor \frac{28 + 4 - 4 - 1}{1} \right\rfloor + 1$$
   $$= \left\lfloor 27 \right\rfloor + 1 = 28$$

**결과**: 출력 크기는 28×28 (입력 크기와 동일)

### Step 5.2: PyTorch로 검증

```python
import torch
import torch.nn as nn

# 입력 텐서 생성
input_tensor = torch.randn(1, 3, 28, 28)  # 배치 1, 채널 3, 28x28
print(f"입력 shape: {input_tensor.shape}")

# Conv2d 레이어 생성
conv = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1, dilation=1)
output = conv(input_tensor)

print(f"출력 shape: {output.shape}")
# 출력: torch.Size([1, 16, 28, 28])

# 수동 계산 검증
H_in, W_in = 28, 28
K = 5
P = 2
S = 1
D = 1

H_out = ((H_in + 2*P - D*(K-1) - 1) // S) + 1
W_out = ((W_in + 2*P - D*(K-1) - 1) // S) + 1

print(f"수동 계산: ({H_out}, {W_out})")
print(f"PyTorch 출력: {output.shape[2:]}")
print(f"일치 여부: {H_out == output.shape[2] and W_out == output.shape[3]}")
```

### Step 5.3: 파라미터 수 계산

**Conv2d의 파라미터 수**:

$$\text{파라미터 수} = C_{out} \times (C_{in} \times K \times K + 1)$$

여기서:
- $C_{out} \times C_{in} \times K \times K$: 가중치 수
- $C_{out} \times 1$: 편향 수

**예시**: `Conv2d(3, 64, 3)`
- 가중치: $64 \times 3 \times 3 \times 3 = 1,728$
- 편향: $64 \times 1 = 64$
- 총 파라미터: $1,728 + 64 = 1,792$

```python
# 파라미터 수 확인
conv = nn.Conv2d(3, 64, 3)
total_params = sum(p.numel() for p in conv.parameters())
print(f"총 파라미터 수: {total_params}")  # 1,792

# 가중치와 편향 개수
weight_params = conv.weight.numel()  # 64 * 3 * 3 * 3 = 1,728
bias_params = conv.bias.numel()      # 64
print(f"가중치: {weight_params}, 편향: {bias_params}")
```

### Step 5.4: Linear vs Conv 파라미터 수 비교

**같은 입력 크기에서 비교**:

```python
# 입력: 28×28 이미지 (784 픽셀)
input_size = 28 * 28  # 784
output_size = 10

# Linear Layer
linear = nn.Linear(input_size, output_size)
linear_params = sum(p.numel() for p in linear.parameters())
print(f"Linear 파라미터: {linear_params}")  # 784 * 10 + 10 = 7,850

# Conv2d (3×3 커널, 10개 필터)
conv = nn.Conv2d(1, 10, 3, padding=1)  # 크기 보존
conv_params = sum(p.numel() for p in conv.parameters())
print(f"Conv2d 파라미터: {conv_params}")  # 10 * 1 * 3 * 3 + 10 = 100

print(f"비율: Linear는 Conv2d의 {linear_params / conv_params:.1f}배")
```

**결과**: Conv2d가 훨씬 적은 파라미터로 같은 입력을 처리할 수 있습니다.

---

## Step 6: Conv2d의 구조적 이해

### Step 6.1: 입력 (Input)

**입력 형태**: `(batch, channels, height, width)`

- **batch**: 배치 크기 (한 번에 처리할 샘플 수)
- **channels**: 채널 수 (RGB=3, 흑백=1)
- **height, width**: 이미지의 높이, 너비

**예시**:
- RGB 이미지: `(32, 3, 224, 224)` - 배치 32, RGB 3채널, 224×224 픽셀
- 흑백 이미지: `(32, 1, 28, 28)` - 배치 32, 흑백 1채널, 28×28 픽셀

### Step 6.2: 커널 (Kernel, Filter)

**커널은 작은 창(window)**입니다:

- 예: 3×3 크기의 커널은 이미지 위를 슬라이딩하며
- 각 영역의 픽셀과 커널의 가중치를 곱해 더합니다 (내적)
- 이 과정을 **"합성곱(convolution)"**이라고 부릅니다

**의미**:
- 하나의 커널은 이미지 전체를 스캔하며 하나의 **특징맵(feature map)**을 만듭니다
- $C_{out}$개의 커널이 있으면 $C_{out}$개의 특징맵이 생성됩니다

### Step 6.3: 출력 (Output)

**출력 형태**: `(batch, out_channels, out_height, out_width)`

- **batch**: 입력과 동일한 배치 크기
- **out_channels**: 필터 개수 = 추출한 특징 종류
- **out_height, out_width**: 출력 이미지의 높이, 너비 (공식으로 계산)

### Step 6.4: out_channels의 의미

**out_channels는 추출할 특징의 종류 수**입니다:

- **하나의 필터는 하나의 특징 탐지기**
- 예: 가장자리 탐지 필터, 형태 탐지 필터, 질감 탐지 필터 등
- $C_{out}$개의 필터가 있으면 $C_{out}$가지 특징을 동시에 탐지

**의미**:
- 모델이 가지한 의미 있는 특징의 개수
- 이 층이 몇 가지 시각 패턴을 병렬로 학습할 수 있는가를 결정하는 신경의 수

---

## Step 7: Pooling Layer

### Step 7.1: 왜 Pooling이 필요한가?

**문제 상황**:
- Conv Layer를 거치면 특징맵의 크기가 커질 수 있음
- 파라미터 수와 계산량이 증가
- 과적합(overfitting) 위험

**해결책**: Pooling으로 특징맵 크기를 줄입니다

### Step 7.2: MaxPool2d

**MaxPool2d**는 영역 내 최댓값을 선택합니다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 입력: 배치 1, 채널 16, 28×28
input = torch.randn(1, 16, 28, 28)

# 방법 1: nn.MaxPool2d 사용
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
output1 = maxpool(input)
print(f"MaxPool2d 출력: {output1.shape}")  # (1, 16, 14, 14)

# 방법 2: F.max_pool2d 사용
output2 = F.max_pool2d(input, kernel_size=2, stride=2)
print(f"F.max_pool2d 출력: {output2.shape}")  # (1, 16, 14, 14)
```

**주요 파라미터**:
- `kernel_size`: 풀링 영역 크기 (예: 2×2)
- `stride`: 이동 간격 (보통 `kernel_size`와 동일)

**출력 크기**:
$$H_{out} = \left\lfloor \frac{H_{in} - K}{S} \right\rfloor + 1$$

**예시**: 입력 28×28, kernel=2, stride=2
$$H_{out} = \left\lfloor \frac{28 - 2}{2} \right\rfloor + 1 = 14$$

### Step 7.3: MaxPool의 특징

**MaxPool Layer는 weight가 없습니다**:

```python
# MaxPool은 파라미터가 없음
maxpool = nn.MaxPool2d(2, 2)
params = list(maxpool.parameters())
print(f"파라미터 수: {len(params)}")  # 0

# 따라서 detach() 없이 바로 numpy() 변환 가능 (하지만 MaxPool은 가중치가 없으므로 의미 없음)
```

**의미**: MaxPool은 학습 가능한 파라미터가 없는 연산입니다.

### Step 7.4: Conv Block 구성

**Conv Block**은 Convolutional Neural Network의 기본 단위입니다:

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# 사용 예시
block = ConvBlock(3, 16)
input = torch.randn(1, 3, 28, 28)
output = block(input)
print(f"입력: {input.shape}, 출력: {output.shape}")
# 입력: (1, 3, 28, 28), 출력: (1, 16, 14, 14)
```

**구성 요소**:
1. **Conv2d**: 특징 추출
2. **ReLU**: 비선형 활성화
3. **MaxPool2d**: 다운샘플링

---

## Step 8: 빠른 질의응답 정리

### Q1: stride의 효과는?

**A**: 기본값 1에서 벗어나면 커널 이동 폭이 커져 출력 해상도가 `stride` 배로 줄어듭니다.

- stride=1: 출력 크기 ≈ 입력 크기
- stride=2: 출력 크기 ≈ 입력 크기 / 2

### Q2: dilation의 효과는?

**A**: 커널 내부 간격을 `dilation·(kernel−1)+1`로 펼쳐 업스케일 없이 수용 영역을 넓힙니다.

- dilation=1: 일반적인 커널
- dilation=2: 커널 요소 사이를 1칸씩 띄움 (수용 영역 확대)

### Q3: padding의 효과는?

**A**: 입력 테두리를 `pad`만큼 채워 경계 손실을 줄이고 목표 출력 크기를 맞춥니다.

- padding=0: 경계 정보 손실
- padding=1 (3×3 커널): 크기 보존
- padding=2 (5×5 커널): 크기 보존

### Q4: 출력 크기 공식은?

**A**: 
$$H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

### Q5: weight 텐서 구조는?

**A**: `(out_channels, in_channels, kernel_height, kernel_width)`

예: `torch.Size([20, 1, 5, 5])`는 20개 필터가 1채널 5×5 커널을 가진다는 뜻입니다.

### Q6: 커널 시각화는?

**A**: `plt.imshow(weight[0, 0, :, :], "jet")`은 첫 필터의 가중치를 색상으로 표현한 열 지도이며 실제 픽셀 색상은 아닙니다.

---

## 연습 문제

### 문제 1: 출력 크기 계산
입력 28×28, kernel 5, stride 2, padding 1일 때 출력 크기를 계산하고 확인하세요.

```python
import torch
import torch.nn as nn

# 계산
H_in = 28
K = 5
S = 2
P = 1
D = 1

H_out = ((H_in + 2*P - D*(K-1) - 1) // S) + 1
print(f"계산 결과: {H_out}")  # 13

# PyTorch로 검증
conv = nn.Conv2d(1, 1, 5, stride=2, padding=1)
input = torch.randn(1, 1, 28, 28)
output = conv(input)
print(f"PyTorch 출력: {output.shape[2]}")  # 13
```

### 문제 2: 파라미터 수
Conv2d(3, 64, 3)의 총 파라미터 수를 계산하세요 (bias 포함).

```python
conv = nn.Conv2d(3, 64, 3)
total = sum(p.numel() for p in conv.parameters())
print(f"총 파라미터: {total}")  # 1,792
```

### 문제 3: Linear vs Conv
같은 입력 크기에서 Linear와 Conv의 파라미터 수를 비교하세요.

### 문제 4: dilation 효과
dilation=2일 때 실제 수용 영역이 얼마나 커지는지 설명하세요.

**답**: 
- 커널 크기 $K=3$, dilation $D=2$
- 유효 커널 크기: $D(K-1) + 1 = 2(3-1) + 1 = 5$
- 3×3 커널이지만 5×5 영역을 봄

### 문제 5: 시각화
커널의 가중치를 히트맵으로 시각화하고 의미를 해석하세요.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

layer = nn.Conv2d(1, 20, 5, 1)
weight = layer.weight.detach()

# 첫 번째 필터 시각화
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(weight[i, 0, :, :].numpy(), cmap='jet')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## 핵심 요약

### 1. Conv2d의 역할
- 2차원 이미지의 국소 패턴 인식
- Linear: 전역적 연결 / Conv2d: 지역적 연결
- 특징 추출기(feature extractor)

### 2. 주요 파라미터
- `in_channels`: 입력 채널 수
- `out_channels`: 필터 수 = 추출할 특징 종류
- `kernel_size`: 필터 크기 = 지역 패턴 범위
- `stride`: 이동 간격 = 출력 해상도 제어
- `padding`: 가장자리 0 추가 = 경계 정보 보존
- `dilation`: 필터 간격 확장 = 넓은 문맥 파악

### 3. 출력 크기 계산
$$H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

### 4. Weight 텐서 구조
- `(out_channels, in_channels, kernel_height, kernel_width)`
- 각 필터는 하나의 특징 탐지기

### 5. Conv Block
- Conv2d + ReLU + MaxPool2d
- Convolutional Neural Network의 기본 단위

다음 단원에서는 **풀링과 활성화 함수**에 대해 배워보겠습니다.

