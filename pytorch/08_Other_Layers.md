# 08. 기타 레이어들

이번 단원에서는 신경망의 추가 구성요소들을 배워보겠습니다.

## 학습 목표
- Pooling Layer (풀링 레이어)
- Non-linear Activations (비선형 활성화 함수)
- 신경망 종류 개요

## 이 단원을 배우기 전에

**이전 단원 (07) 복습: Conv2d의 파라미터(stride, padding 등)를 이해하셨나요?**

## 이 단원 다음에는

**다음 단원 (09): 레이어를 배웠으니, 실제 프로젝트에서 필요한 데이터 라벨링을 배워봅시다.**

## 왜 이 단원이 필요한가?

신경망은 다양한 레이어들의 조합으로 이루어집니다. 각 레이어가 하는 역할을 알아야 모델을 설계할 수 있습니다.

- **Pooling**: 다운샘플링, 과적합 방지
- **Activation**: 비선형성 추가 (필수!)
- **Dropout, BatchNorm**: 학습 안정화

이 레이어들을 조합해 다양한 아키텍처를 만들 수 있습니다.

### 풀링 레이어(Pooling layers)

- `F.max_pool2d`
  - `stride`
  - `kernel_size`
- `torch.nn.MaxPool2d` 도 많이 사용

```python

```

- MaxPool Layer는 weight가 없기 때문에 바로 `numpy()`변환 가능

```python

```

```python

```

### 선형 레이어(Linear layers)

1d만 가능하므로 `.view()`를 통해 1d로 펼쳐줘야함

```python

```

```python

```

```python

```

```python

```

### 비선형 활성화 (Non-linear Activations)

`F.softmax`와 같은 활성화 함수 등

```python

```

```python

```

`F.relu`

- ReLU 함수를 적용하는 레이어
- `nn.ReLU`로도 사용 가능

```python

```

```python

```

### 정규화 및 규제 레이어

신경망 학습을 안정화하고 과적합을 방지하는 레이어들

#### Dropout
- 학습 시 랜덤하게 일부 뉴런을 비활성화 (과적합 방지)
- 테스트 시에는 모든 뉴런 활성화 (평균 예측)
- 확률 p로 뉴런을 0으로 만듦

#### BatchNorm (Batch Normalization)
- 배치 내 통계량(평균, 분산)으로 정규화
- 학습 안정화, 학습 속도 향상
- 내부 파라미터 (gamma, beta)로 스케일/시프트 조정

```python
import torch
import torch.nn as nn

# Dropout 예제
input_tensor = torch.randn(10, 20)

# Dropout 레이어 생성 (확률 0.5)
dropout = nn.Dropout(p=0.5)

# 학습 모드 (training mode)
dropout.train()
output_train = dropout(input_tensor)
print(f"입력 shape: {input_tensor.shape}")
print(f"Dropout 출력 (학습 모드):")
print(f"  0의 개수: {(output_train == 0).sum().item()} / {output_train.numel()}")

# 평가 모드 (eval mode)
dropout.eval()
output_eval = dropout(input_tensor)
print(f"\nDropout 출력 (평가 모드):")
print(f"  0의 개수: {(output_eval == 0).sum().item()} / {output_eval.numel()}")
print(f"모든 값 활성화: {torch.allclose(output_eval, input_tensor)}")
```

```python
# BatchNorm 예제
# 1D BatchNorm: Linear 레이어와 함께 사용
x_1d = torch.randn(32, 128)  # 배치 32, 특징 128
bn1d = nn.BatchNorm1d(num_features=128)
output_1d = bn1d(x_1d)
print(f"BatchNorm1d:")
print(f"  입력 shape: {x_1d.shape}")
print(f"  출력 shape: {output_1d.shape}")
print(f"  평균: {output_1d.mean():.6f}, 표준편차: {output_1d.std():.6f}")

# 2D BatchNorm: Conv 레이어와 함께 사용 (이미지)
x_2d = torch.randn(8, 16, 32, 32)  # 배치 8, 채널 16, 32x32
bn2d = nn.BatchNorm2d(num_features=16)
output_2d = bn2d(x_2d)
print(f"\nBatchNorm2d:")
print(f"  입력 shape: {x_2d.shape}")
print(f"  출력 shape: {output_2d.shape}")

# BatchNorm의 학습 파라미터
print(f"\n학습 파라미터:")
print(f"  가중치 (gamma): {bn2d.weight.shape}")
print(f"  편향 (beta): {bn2d.bias.shape}")
```

```python
# Softmax 예제
logits = torch.randn(4, 10)  # 배치 4, 클래스 10

# Softmax 함수 적용
softmax = nn.Softmax(dim=1)
probs = softmax(logits)

print(f"입력 (logits) shape: {logits.shape}")
print(f"출력 (probabilities) shape: {probs.shape}")
print(f"각 샘플의 합: {probs.sum(dim=1)}")  # 모든 행의 합이 1인지 확인

# 실제 예측 (가장 높은 확률의 클래스)
predictions = probs.argmax(dim=1)
print(f"예측된 클래스: {predictions}")
```

### 신경망 종류

## 모델 정의

### `nn.Module` 상속 클래스 정의
- `nn.Module`을 상속받는 클래스 정의
- `__init__()`: 모델에서 사용될 모듈과 활성화 함수 등을 정의
- `forward()`: 모델에서 실행되어야 하는 연산을 정의

```python

```

```python

```

## 연습 문제

### 문제 1: Pooling 비교
MaxPool과 AvgPool의 출력 차이를 설명하고 예시로 보여주세요.

### 문제 2: ReLU 동작
음수 값을 가진 텐서에 ReLU를 적용하고 변화를 관찰하세요.

### 문제 3: Sigmoid vs Softmax
Sigmoid와 Softmax의 차이를 설명하고 각각 어떤 경우에 사용하나요?

### 문제 4: 비선형성 필요성
ReLU 없이 Linear만으로 쌓은 모델의 한계를 설명하세요.

### 문제 5: Layer 조합
Conv → ReLU → MaxPool 블록을 만들고 출력을 확인하세요.

## 핵심 요약

### 1. Pooling Layer
- MaxPool: 지역 최댓값 선택
- AvgPool: 지역 평균 계산
- 역할: 다운샘플링 + 과적합 방지

### 2. 비선형 활성화 함수
- ReLU: 최대(0, x)
- Sigmoid: 0~1 출력
- Tanh: -1~1 출력
- Softmax: 다중 분류 확률 계산
- 비선형성의 필요성 이해

### 3. 정규화 및 규제 레이어
- **Dropout**: 과적합 방지, 학습 시 랜덤 비활성화
- **BatchNorm**: 학습 안정화, 평균/분산 정규화

### 4. 신경망 종류
- CNN, RNN, GAN 등

**축하합니다!** PyTorch의 기본 구조에 대한 모든 단원을 완료했습니다!
