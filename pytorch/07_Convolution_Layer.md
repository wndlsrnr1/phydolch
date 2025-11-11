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

## 이 단원 다음에는

**다음 단원 (08): Conv Layer를 배웠으니, Pooling, Activation 등 기타 레이어들을 알아봅시다.**

## 왜 이 단원이 필요한가?

Conv2d는 이미지 처리의 핵심입니다. 딥러닝이 이미지 인식에서 혁명을 일으킨 주역입니다.

- **이미지 특화**: 공간적 정보 활용
- **효율성**: 전역 연결보다 훨씬 적은 파라미터
- **패턴 탐지**: 가장자리, 형태, 질감 등 자동 추출
- **계층적 학습**: 저수준 → 고수준 특징

이미지 프로젝트라면 반드시 마스터해야 할 레이어입니다.

### 컨볼루션 레이어(Convolution Layers)

`nn.Conv2d` 예제

- `in_channels`: channel의 갯수
- `out_channels`: 출력 채널의 갯수
- `kernel_size`: 커널(필터) 사이즈

### 출력 크기 계산 공식

Conv2d의 출력 크기는 다음 공식으로 계산됩니다:

```
output_size = ⌊(input_size + 2×padding - dilation×(kernel_size - 1) - 1) / stride⌋ + 1
```

**각 파라미터의 영향**:
- `padding`: 출력 크기 증가 (경계 보존)
- `stride`: 출력 크기 감소 (다운샘플링)
- `dilation`: 수용 영역 확대 (출력 크기는 감소)

**예시**:
- 입력 28×28, kernel=5, padding=2, stride=1, dilation=1
  - 출력 = ⌊(28 + 2×2 - 1×(5-1) - 1) / 1⌋ + 1 = ⌊(28 + 4 - 4 - 1) / 1⌋ + 1 = 28

```python
import torch
import torch.nn as nn

# 다양한 파라미터 조합으로 출력 크기 확인
input_tensor = torch.randn(1, 3, 28, 28)  # 배치 1, 채널 3, 28x28 이미지

# 예제 1: 기본 (padding=0, stride=1, dilation=1)
conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0, stride=1, dilation=1)
output1 = conv1(input_tensor)
print(f"입력: {input_tensor.shape}")
print(f"출력 (기본): {output1.shape}")

# 예제 2: padding 추가 (크기 보존)
conv2 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1, dilation=1)
output2 = conv2(input_tensor)
print(f"출력 (padding=2): {output2.shape}")

# 예제 3: stride 적용 (다운샘플링)
conv3 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2, dilation=1)
output3 = conv3(input_tensor)
print(f"출력 (stride=2): {output3.shape}")

# 예제 4: dilation 적용 (수용 영역 확대)
conv4 = nn.Conv2d(3, 16, kernel_size=5, padding=4, stride=1, dilation=2)
output4 = conv4(input_tensor)
print(f"출력 (dilation=2): {output4.shape}")
```

```python
import torch
input = torch.rand(20, 16, 50, 10)
print(input.size())

m = nn.conv2d(16, 33, 3, stride=2)
```

```python

```

`weight` 확인

```python

```

`weight`는 `detach()`를 통해 꺼내줘야 `numpy()`변환이 가능

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

### 풀링 레이어(Pooling layers)

- `F.max_pool2d`
  - `stride`
  - `kernel_size`
- `torch.nn.MaxPool2d` 도 많이 사용

```python

```

- MaxPool Layer는 weight가 없기 때문에 바로 `numpy()`변환 가능

### Conv2d란 무엇인가?


Conv2d는 2차원 이미지를 데이터를 입력으로 받아, 작은 "필터(kernel)"을 이용해 특징(feature)를 추출하는 레이어이다. 

Linear가 "모든 입력을 한 번에 연결" 하는 완전 연결층이라면, 

Conv2d는 "이미지의 국소 패턴(local pattern)"을 인식한다. 



예를 들어


가장자리 (edge)

색의 변화 (gradient)

형태 (shape)


같은 시각적 특징을 찾아낸다.


### 구조적 관점 - "필터가 보는 세상"

#### 입력 (Input)

입력은 보통 (batch, channels, height, width) 형태이다. (예: RGB 이미지라면 channels=3, 흑백이면 channels=1)

#### 커널 (Kernel, Filter)

커널은 작은 창(window)입니다.

예를 들어 3x3 크기의 커널은 이미지 위를 슬라이딩하며, 

각 영역의 픽셀과 커널의 가중치를 곱해 더합니다 (내적)

이 과정을 "합성곱(convolution)"이라고 부릅니다.

즉, 하나의 커널은 이미지 전체를 스캔하며 하나의 특징맵(feature map)을 만듭니다. 


### 수학적 표현

yi,j(k)​=c=1∑Cin​​u,v∑​wu,v(k,c)​⋅xi+u,j+v(c)​+bk​

입력 채널별로 합성곱을 한 뒤 더하고, bias를 더해 하나의 출력 채널을 만든다. 


### 주요 파라미터별 개념 정리

#### in_channels
- 입력의 채널 수

즉, Conv2d는 채널 단위로 연산하며, 각 필터가 in_channels 개의 2D 커널을 가집니다. 


#### Linear와 Conv2d의 철학적 차이

| 구분     | Linear        | Conv2d                        |
| ------ | ------------- | ----------------------------- |
| 입력 구조  | 1차원 벡터        | 2차원 이미지                       |
| 연결 방식  | 모든 입력에 가중치 적용 | 지역적 연결(local receptive field) |
| 파라미터 수 | 매우 많음 (입력×출력) | 상대적으로 적음 (커널 고정 크기)           |
| 특징     | 전역적 관계 학습     | 지역적 패턴 학습                     |
| 활용     | 마지막 단계 (분류기)  | 전단계 (특징 추출기)                  |


Conv2d는 일종의 패턴 탐지기 (feature extractor)
Linear는 판단자 (decision maker)라고 생각하면 된다. 

| 파라미터           | 의미             | 효과          |
| -------------- | -------------- | ----------- |
| `in_channels`  | 입력 채널 수        | RGB=3, 흑백=1 |
| `out_channels` | 필터 수 (출력 채널 수) | 추출할 특징 종류   |
| `kernel_size`  | 필터의 크기         | 지역적 패턴 범위   |
| `stride`       | 이동 간격          | 출력 해상도 제어   |
| `padding`      | 가장자리 0 추가      | 경계 정보 보존    |
| `dilation`     | 필터 간격 확장       | 넓은 문맥 파악    |


### 요약 표

| 파라미터           | 의미             | 효과          |
| -------------- | -------------- | ----------- |
| `in_channels`  | 입력 채널 수        | RGB=3, 흑백=1 |
| `out_channels` | 필터 수 (출력 채널 수) | 추출할 특징 종류   |
| `kernel_size`  | 필터의 크기         | 지역적 패턴 범위   |
| `stride`       | 이동 간격          | 출력 해상도 제어   |
| `padding`      | 가장자리 0 추가      | 경계 정보 보존    |
| `dilation`     | 필터 간격 확장       | 넓은 문맥 파악    |


### Conv는 세상을 보는 창

Conv2d는 세상을 스캔하는 눈입니다.

작은 커널이 세상을 창문처럼 훑으며 패턴을 학습합니다. 

stride는 걸음 크기, padding은 시야 확장, dilation은 망원 렌즈와 같습니다.

이렇게 얻은 특징 지도(feature map) 들이 바로

딥러닝이 세상을 이해하는 첫 번째 언어입니다.

RelU

MaxPool2d

conv block

Convolutional Neural Network의 기본 단위 



### out channels

그 감각들을 조합해서 새롭게 만들어내는 시각 패턴의 개수

모델이 가지한 의미 있는 특징의 개수

하나의 필터는 하나의 특징 탐지기

이 층이 몇 가지 시각 패턴을 병렬로 학습 할 수 있는가를 결정하는 신경의 수에 해당한다. 

```python
import torch
import torch.nn as nn
input = torch.randn(20, 16, 50, 100)
print(input.size())

# 배치 20, 채널 16, 높이 50, 너비 100인 난수 입력을 만든다.
input = torch.randn(20, 16, 50, 100)

# 입력 크기 텐서 확인 한다. 
print(input.size())


#
torch.Size([20, 16, 50, 100])
m = nn.Conv2d(16, 33, 3, stride=2)
m = nn.Conv2d(16 ,33, (3, 5), stride=(2, 1), padding=(4, 2))
m = nn.Conv2d(16 ,33, (3, 5), stride=(2, 1), padding=(4, 2), dilation = (3, 1))
output = m(input)
print(output.size())


# dilation = 커널 내부 요소 사이를 세로로 3칸 가로로 1칸 띄워서 적용한다. 
# padding 입력 가장 자리 밖에 세로 4줄 가로 2줄 만듬 덧데어 경계 손실을 줄이고 원하는 출력 크기를 유지/조정한다.
# stride=(2, 1) 커널이 이동하는 간격으로 세로 방향은 한 번에 2칸, 가로 방향은 1칸 씩 이동한다. 따라서 출력이 세로로 더 작아지며 계산량이 줄어준다.
nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
layer = nn.Conv2d(1, 20, 5, 1).to(torch.device("cpu"))

weight = layer.weight
print(weight.shape)

weight = weight.detach()
weight = weight.numpy()
weight.shape

# torch.Size([20, 1, 5, 5])
# (20, 1, 5, 5)

import matplotlib.pyplot as plt

plt.imshow(weight[0, 0, :, :], "jet")
plt.colorbar()
plt.show()


# dilation = kernel 영역을 확장한다. 
# stride = 이동하는 영역이다. 얼마나 촘촘이 보냐를 
# padding = 커널이 이동하는 영역을 확장한다. 

# out = (in + 2 * padding + dilation * (kernal - 1) - 1) / stride + 1
# out = ⌊ (in + 2·pad − dilation·(kernel − 1) − 1) / stride ⌋ + 1
```

### 빠른 질의응답 정리

- **stride**: 기본값 1에서 벗어나면 커널 이동 폭이 커져 출력 해상도가 `stride` 배로 줄어든다.
- **dilation**: 커널 내부 간격을 `dilation·(kernel−1)+1`로 펼쳐 업스케일 없이 수용 영역을 넓힌다.
- **padding**: 입력 테두리를 `pad`만큼 채워 경계 손실을 줄이고 목표 출력 크기를 맞춘다.
- **출력 공식**: `out = ⌊(in + 2·pad − dilation·(kernel − 1) − 1) / stride⌋ + 1`.
- **7×7, 3×3 예시**: `pad=1 → out=7`, `dilation=2 → out=3`, `stride=2 → out=3`, 기본 설정 → `out=5`.
- **weight 텐서**: 구조는 `(out_channels, in_channels, kernel_height, kernel_width)`이고 `torch.Size([20, 1, 5, 5])`는 20개 필터가 1채널 5×5 커널을 가진다는 뜻이다.
- **커널 시각화**: `plt.imshow(weight[0, 0, :, :], "jet")`은 첫 필터의 가중치를 색상으로 표현한 열 지도이며 실제 픽셀 색상은 아니다.

## 연습 문제

### 문제 1: 출력 크기 계산
입력 28x28, kernel 5, stride 2, padding 1일 때 출력 크기를 계산하고 확인하세요.

### 문제 2: 파라미터 수
Conv2d(3, 64, 3)의 총 파라미터 수를 계산하세요 (bias 포함).

### 문제 3: Linear vs Conv
같은 입력 크기에서 Linear와 Conv의 파라미터 수를 비교하세요.

### 문제 4: dilation 효과
dilation=2일 때 실제 수용 영역이 얼마나 커지는지 설명하세요.

### 문제 5: 시각화
커널의 가중치를 히트맵으로 시각화하고 의미를 해석하세요.

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
```
out = ⌊(in + 2·pad − dilation·(kernel − 1) − 1) / stride⌋ + 1
```

### 4. Weight 텐서 구조
- `(out_channels, in_channels, kernel_height, kernel_width)`
- 각 필터는 하나의 특징 탐지기

다음 단원에서는 **풀링과 활성화 함수**에 대해 배워보겠습니다.
