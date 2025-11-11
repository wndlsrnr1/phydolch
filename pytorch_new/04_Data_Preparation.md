# 04. 데이터 준비

이번 단원에서는 모델 학습을 위한 데이터 준비 과정을 배워보겠습니다.

## 학습 목표
- Dataset과 DataLoader의 개념
- 데이터 전처리와 변환(transforms)
- 배치 처리의 의미
- 데이터 시각화

## 이 단원을 배우기 전에

**이전 단원 (03) 복습: backward(), requires_grad의 의미를 이해하셨나요?**

## 이 단원 다음에는

**다음 단원 (05): 데이터를 준비했으니, 이제 신경망 구조를 설계하는 방법을 배워봅시다.**

## 왜 이 단원이 필요한가?

데이터는 머신러닝의 생명입니다. 모델은 데이터가 없으면 학습할 수 없고, 좋은 데이터 없이는 좋은 모델을 만들 수 없습니다.

- **데이터셋 구축**: 필요 데이터 수집 및 정리
- **전처리**: 모델이 이해할 수 있는 형태로 변환
- **배치 처리**: 효율적인 학습을 위한 데이터 묶음 생성
- **증강(Augmentation)**: 적은 데이터로 더 많은 정보 만들기

이 단원에서 배우는 것은 모든 프로젝트에서 필수적입니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 데이터 관련 기호

- $D = \{(x_i, y_i)\}_{i=1}^{N}$: 데이터셋 (Dataset), $N$개의 샘플-레이블 쌍
- $x_i \in \mathbb{R}^{C \times H \times W}$: $i$번째 입력 샘플 (예: 이미지)
- $y_i \in \{1, 2, \ldots, K\}$: $i$번째 레이블 (예: 클래스 번호)
- $B$: 배치 크기 (batch size), 한 번에 처리하는 샘플 수
- $N$: 전체 데이터셋 크기
- $\mathcal{T}$: 변환 함수 (transform function)
- $\mathcal{T}(x)$: 입력 $x$에 변환을 적용한 결과

### PyTorch 코드에서의 변수

- `dataset`: Dataset 객체
- `dataloader`: DataLoader 객체
- `batch_size`: 배치 크기
- `transform`: 데이터 변환 함수
- `shuffle`: 데이터 셔플 여부
- `num_workers`: 데이터 로딩에 사용할 프로세스 수
- `img`: 이미지 텐서
- `label`: 레이블 값

---

## Step 1: 왜 데이터 전처리가 필요한가?

### Step 1.1: 문제 상황

**데이터의 다양한 형태**:
- 이미지 데이터: PIL Image, NumPy 배열, 파일 경로
- 텍스트 데이터: 문자열, 토큰 ID
- 수치 데이터: CSV 파일, Excel 파일

**모델의 요구사항**:
- PyTorch 모델은 **Tensor 형태의 입력만** 받을 수 있음
- 배치로 묶으려면 **모든 샘플의 크기가 같아야** 함
- 학습 안정성을 위해 **데이터 정규화** 필요

### Step 1.2: 구체적 문제 예시

**문제 1: 데이터 타입 불일치**
```python
# 이미지 데이터가 PIL Image 형태
from PIL import Image
img = Image.open("image.jpg")  # PIL Image 객체

# 모델은 Tensor만 받을 수 있음
model(img)  # ❌ 오류 발생!
```

**문제 2: 이미지 크기 불일치**
```python
# 서로 다른 크기의 이미지들
img1 = torch.randn(3, 224, 224)  # 224×224
img2 = torch.randn(3, 256, 256)  # 256×256

# 배치로 묶을 수 없음
batch = torch.stack([img1, img2])  # ❌ 오류 발생!
```

**문제 3: 픽셀 값 범위 문제**
```python
# 이미지 픽셀 값이 0~255 범위
img = torch.randint(0, 255, (3, 224, 224))  # 0~255

# 신경망은 보통 -1~1 또는 0~1 범위에서 학습
# 큰 값으로 인해 학습이 불안정해질 수 있음
```

### Step 1.3: 해결책: 데이터 전처리 (Transforms)

**전처리의 목적**:
1. **타입 변환**: PIL Image → Tensor
2. **크기 통일**: 모든 이미지를 같은 크기로 리사이즈
3. **값 정규화**: 픽셀 값을 적절한 범위로 변환
4. **데이터 증강**: 회전, 자르기 등으로 데이터 다양성 증가

---

## Step 2: transforms란 무엇인가?

### Step 2.0: 기호 정리

- $\mathcal{T}$: 변환 함수 (transform function)
- $\mathcal{T}(x)$: 입력 $x$에 변환을 적용한 결과
- $\mathcal{T}_1 \circ \mathcal{T}_2$: 두 변환의 합성 (composition)
- $x_{raw}$: 원본 데이터
- $x_{processed}$: 전처리된 데이터

### Step 2.1: transforms의 수학적 정의

**변환 함수 (Transform Function)**:

$$\mathcal{T}: \mathcal{X} \rightarrow \mathcal{X}'$$

여기서:
- $\mathcal{X}$: 원본 데이터 공간 (예: PIL Image)
- $\mathcal{X}'$: 변환된 데이터 공간 (예: Tensor)

**변환의 합성**:

여러 변환을 순차적으로 적용:

$$x_{processed} = \mathcal{T}_n(\mathcal{T}_{n-1}(\cdots \mathcal{T}_1(x_{raw})\cdots))$$

**의미**: 원본 데이터에 여러 변환을 순차적으로 적용하여 최종 전처리된 데이터를 얻습니다.

### Step 2.2: torchvision.transforms 패키지

**torchvision.transforms**는 PyTorch에서 제공하는 데이터 변환 함수들의 모음입니다.

**주요 변환 함수**:
- `ToTensor()`: PIL Image 또는 NumPy 배열 → Tensor
- `Resize(size)`: 이미지 크기 조정
- `Normalize(mean, std)`: 정규화 (평균 빼기, 표준편차로 나누기)
- `RandomHorizontalFlip()`: 랜덤 수평 뒤집기 (데이터 증강)
- `Compose([...])`: 여러 변환을 순차적으로 적용

**공식 문서**: https://pytorch.org/docs/stable/torchvision/transforms.html

### Step 2.3: ToTensor()가 필요한 이유

**문제**: torchvision 데이터셋은 PIL Image 형태로만 입력을 받습니다.

**해결**: `ToTensor()`를 사용하여 PIL Image를 Tensor로 변환합니다.

**변환 과정**:
1. PIL Image: `(H, W, C)` 형태, 픽셀 값 0~255 (uint8)
2. ToTensor() 적용
3. Tensor: `(C, H, W)` 형태, 픽셀 값 0.0~1.0 (float32)

**수학적 표현**:

$$\text{ToTensor}(I) = \frac{I}{255.0}$$

여기서 $I$는 PIL Image의 픽셀 값 (0~255 범위)입니다.

### Step 2.4: Compose를 통한 변환 체인

**Compose**는 여러 변환을 하나의 변환으로 묶어주는 함수입니다.

**수학적 표현**:

$$\text{Compose}([\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_n]) = \mathcal{T}_n \circ \mathcal{T}_{n-1} \circ \cdots \circ \mathcal{T}_1$$

**의미**: 여러 변환을 순차적으로 적용하는 하나의 변환 함수를 만듭니다.

---

## Step 3: transforms 사용법 (PyTorch 구현)

### Step 3.1: 기본 transforms 사용

```python
import torch
from torchvision import transforms
from PIL import Image

# Step 0: 기호 정리
# - img: 원본 PIL Image
# - transform: 변환 함수
# - img_tensor: 변환된 Tensor

# 원본 이미지 로드 (PIL Image)
img = Image.open("example.jpg")
print(f"원본 이미지 타입: {type(img)}")
print(f"원본 이미지 크기: {img.size}")  # (W, H)

# ToTensor() 변환 적용
transform = transforms.ToTensor()
img_tensor = transform(img)

print(f"\n변환 후 타입: {type(img_tensor)}")
print(f"변환 후 shape: {img_tensor.shape}")  # (C, H, W)
print(f"픽셀 값 범위: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
```

**출력 예시**:
```
원본 이미지 타입: <class 'PIL.Image.Image'>
원본 이미지 크기: (224, 224)

변환 후 타입: <class 'torch.Tensor'>
변환 후 shape: torch.Size([3, 224, 224])
픽셀 값 범위: [0.00, 1.00]
```

### Step 3.2: Compose를 사용한 여러 변환 적용

```python
import torch
from torchvision import transforms
from PIL import Image

# Step 0: 기호 정리
# - transform: 여러 변환을 순차적으로 적용하는 함수
# - img: 원본 PIL Image
# - img_processed: 전처리된 Tensor

# 여러 변환을 Compose로 묶기
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 1. 크기 조정
    transforms.ToTensor(),              # 2. Tensor 변환
    transforms.Normalize(               # 3. 정규화
        mean=[0.485, 0.456, 0.406],    # ImageNet 평균
        std=[0.229, 0.224, 0.225]       # ImageNet 표준편차
    )
])

# 원본 이미지
img = Image.open("example.jpg")
print(f"원본 크기: {img.size}")

# 변환 적용
img_processed = transform(img)
print(f"\n전처리 후 shape: {img_processed.shape}")
print(f"전처리 후 값 범위: [{img_processed.min():.2f}, {img_processed.max():.2f}]")
```

**출력 예시**:
```
원본 크기: (512, 384)

전처리 후 shape: torch.Size([3, 224, 224])
전처리 후 값 범위: [-2.12, 2.64]
```

**정규화 수식**:

$$\text{Normalize}(x) = \frac{x - \mu}{\sigma}$$

여기서:
- $\mu = [0.485, 0.456, 0.406]$: 각 채널의 평균
- $\sigma = [0.229, 0.224, 0.225]$: 각 채널의 표준편차

### Step 3.3: 커스텀 transform 만들기

torchvision에서 제공하는 클래스 이외의 변환은 일반적으로 클래스를 따로 만들어 전처리 단계를 진행합니다.

```python
import torch
from torchvision import transforms
from PIL import Image

# Step 0: 기호 정리
# - CustomTransform: 커스텀 변환 클래스
# - img: 입력 이미지
# - img_transformed: 변환된 이미지

# 커스텀 transform 클래스 정의
class CustomNormalize:
    """픽셀 값을 -1~1 범위로 정규화하는 커스텀 transform"""
    
    def __call__(self, tensor):
        """
        Args:
            tensor: (C, H, W) 형태의 Tensor, 값 범위 0~1
        Returns:
            normalized: (C, H, W) 형태의 Tensor, 값 범위 -1~1
        """
        # 수식: normalized = 2 * tensor - 1
        # 0 → -1, 0.5 → 0, 1 → 1
        return 2 * tensor - 1

# Compose에 커스텀 transform 포함
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    CustomNormalize()  # 커스텀 변환 적용
])

# 사용 예시
img = Image.open("example.jpg")
img_tensor = transform(img)
print(f"정규화 후 값 범위: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
```

**출력 예시**:
```
정규화 후 값 범위: [-1.00, 1.00]
```

---

## Step 4: Dataset과 DataLoader

### Step 4.0: 기호 정리

- $D = \{(x_i, y_i)\}_{i=1}^{N}$: 데이터셋, $N$개의 샘플-레이블 쌍
- $B$: 배치 크기 (batch size)
- $\lceil N/B \rceil$: 배치 개수 (전체 샘플 수를 배치 크기로 나눈 값의 올림)
- $X_b \in \mathbb{R}^{B \times C \times H \times W}$: 배치 $b$의 입력 데이터
- $Y_b \in \mathbb{R}^{B}$: 배치 $b$의 레이블

### Step 4.1: 왜 DataLoader가 필요한가?

**상황**: 60,000개의 MNIST 이미지가 있다고 가정

**문제**: 한 번에 하나씩 처리하면?

```python
# 비효율적!
for i in range(60000):
    image = load_image(i)
    prediction = model(image)  # 한 번씩 처리
```

**문제점**:
- 60,000번의 forward pass → 느림
- GPU 활용 불가 (한 번에 하나씩만 처리)
- 메모리 효율성 낮음

**해결**: 배치로 묶어서 처리

```python
# 효율적!
for batch in dataloader:  # batch_size=64
    images, labels = batch  # 64개를 한 번에
    predictions = model(images)  # GPU 병렬 처리!
```

**장점**:
- 배치 개수: $\lceil 60000/64 \rceil = 938$번의 배치 → 빠름
- GPU가 64개를 동시에 계산 (병렬 처리)
- 메모리 효율적

**DataLoader는 배치 생성 + 메모리 효율 + 병렬 처리의 핵심입니다!**

### Step 4.2: Dataset 클래스의 수학적 정의

**Dataset**은 데이터 샘플과 레이블을 저장하는 클래스입니다.

**수학적 정의**:

$$D = \{(x_i, y_i)\}_{i=1}^{N}$$

여기서:
- $N$: 전체 샘플 수
- $x_i$: $i$번째 입력 샘플
- $y_i$: $i$번째 레이블

**Dataset 클래스의 역할**:
- `__len__()`: 데이터셋 크기 반환 ($N$)
- `__getitem__(i)`: $i$번째 샘플-레이블 쌍 $(x_i, y_i)$ 반환

### Step 4.3: DataLoader의 수학적 의미

**DataLoader**는 Dataset에서 배치를 생성하는 클래스입니다.

**배치 생성 과정**:

전체 데이터셋 $D = \{(x_i, y_i)\}_{i=1}^{N}$를 배치 크기 $B$로 나눕니다:

$$D = D_1 \cup D_2 \cup \cdots \cup D_{\lceil N/B \rceil}$$

여기서 각 배치 $D_b$는:

$$D_b = \{(x_{(b-1)B+1}, y_{(b-1)B+1}), \ldots, (x_{bB}, y_{bB})\}$$

**배치 텐서**:

각 배치를 텐서로 묶으면:

$$X_b = \begin{bmatrix} x_{(b-1)B+1} \\ x_{(b-1)B+2} \\ \vdots \\ x_{bB} \end{bmatrix} \in \mathbb{R}^{B \times C \times H \times W}$$

$$Y_b = \begin{bmatrix} y_{(b-1)B+1} \\ y_{(b-1)B+2} \\ \vdots \\ y_{bB} \end{bmatrix} \in \mathbb{R}^{B}$$

### Step 4.4: Dataset과 DataLoader 사용법

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Step 0: 기호 정리
# - trainset: 훈련 데이터셋
# - trainloader: 훈련 데이터 로더
# - batch: 배치 데이터 (images, labels)
# - images: 배치의 이미지들 (B, C, H, W)
# - labels: 배치의 레이블들 (B,)

# 1. Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST는 그레이스케일
])

# 2. Dataset 생성
trainset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print(f"데이터셋 크기: {len(trainset)}")  # 60,000

# 3. DataLoader 생성
trainloader = DataLoader(
    trainset,
    batch_size=64,      # 배치 크기 B = 64
    shuffle=True,       # 데이터 셔플
    num_workers=2       # 병렬 로딩 프로세스 수
)

# 4. 배치 반복 처리
for batch_idx, (images, labels) in enumerate(trainloader):
    print(f"\n배치 {batch_idx + 1}:")
    print(f"  이미지 shape: {images.shape}")  # (64, 1, 28, 28)
    print(f"  레이블 shape: {labels.shape}")    # (64,)
    print(f"  레이블 값: {labels[:5].tolist()}")  # 처음 5개 레이블
    
    if batch_idx == 0:  # 첫 번째 배치만 출력
        break
```

**출력 예시**:
```
데이터셋 크기: 60000

배치 1:
  이미지 shape: torch.Size([64, 1, 28, 28])
  레이블 shape: torch.Size([64])
  레이블 값: [3, 7, 2, 1, 9]
```

**DataLoader의 주요 파라미터**:
- `batch_size`: 배치 크기 $B$ (기본값: 1)
- `shuffle`: 데이터 셔플 여부 (기본값: False)
- `num_workers`: 병렬 로딩 프로세스 수 (기본값: 0)
- `drop_last`: 마지막 배치를 버릴지 여부 (기본값: False)

### Step 4.5: shuffle=True와 False의 차이

**shuffle=False** (기본값):
```python
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

# 첫 번째 배치
for images, labels in trainloader:
    print(f"첫 번째 배치 레이블: {labels[:10].tolist()}")
    break
```

**출력**: `[5, 0, 4, 1, 9, 2, 1, 3, 1, 4]` (순서대로)

**shuffle=True**:
```python
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# 첫 번째 배치
for images, labels in trainloader:
    print(f"첫 번째 배치 레이블: {labels[:10].tolist()}")
    break
```

**출력**: `[3, 7, 2, 1, 9, 4, 5, 8, 6, 0]` (랜덤 순서)

**왜 shuffle이 필요한가?**
- 훈련 데이터가 클래스별로 정렬되어 있으면 학습이 불안정할 수 있음
- 각 배치에 다양한 클래스가 포함되도록 하여 학습 안정성 향상

---

## Step 5: 데이터 시각화 - Matplotlib

### Step 5.0: 기호 정리

- `figure`: matplotlib의 그림 전체 (도화지)
- `subplot`: 그림 내의 개별 플롯 영역
- `img`: 이미지 텐서
- `cmap`: 컬러맵 (예: "gray"는 그레이스케일)

### Step 5.1: 왜 시각화가 필요한가?

**문제 상황**:
- 데이터가 제대로 로드되었는지 확인 필요
- 전처리가 올바르게 적용되었는지 확인 필요
- 레이블이 올바른지 확인 필요

**해결책**: Matplotlib을 사용하여 데이터를 시각적으로 확인

### Step 5.2: Matplotlib 기본 사용법

**matplotlib**은 Python에서 데이터를 시각화하는 라이브러리입니다.

**pyplot**은 matplotlib의 submodule로, 단계별로 그리기용 도화지처럼 다루는 기능을 제공합니다.

**주요 함수**:
- `pyplot.figure()`: 새로운 그림 생성
- `pyplot.imshow()`: 이미지 표시
- `pyplot.title()`: 제목 설정
- `pyplot.axis()`: 축 설정
- `pyplot.show()`: 그림 표시

### Step 5.3: 데이터셋 샘플 시각화

```python
import torch
import matplotlib.pyplot as pyplot
from torchvision import datasets, transforms

# Step 0: 기호 정리
# - dataset: 데이터셋 객체
# - sample_idx: 샘플 인덱스
# - img: 이미지 텐서
# - label: 레이블 값
# - figure: matplotlib 그림 객체
# - cols, rows: 그리드의 열과 행 수

# 데이터셋 로드
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 4 x 2 그리드로 이미지 배치
cols, rows = 4, 2
figure = pyplot.figure(figsize=(12, 6))  # 가로 12인치, 세로 6인치

# 도화지의 추상화. 즉 하나의 그림 전체를 의미한다.
# 도화지의 크기를 "인치 단위"로 설정한다.
# (12, 6) 가로 12인치 세로 6인치
# 큰 이미지를 여러 개 배치할 때 넓게 보이도록 하는 설정.

# 8개의 랜덤 샘플 선택하여 표시
for i in range(1, cols * rows + 1):
    # 데이터셋 크기에서 랜덤 인덱스 선택
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    
    # 샘플 가져오기
    img, label = dataset[sample_idx]
    
    # 전체 이미지에 나눈 이미지를 넣는다. rows, cols로 나눈 이미지에 i 번째에 지금의 이미지를 할당
    figure.add_subplot(rows, cols, i)
    
    # 라벨 이름을 표시
    pyplot.title(f"Label: {label}")
    
    # 눈금선을 숨김
    pyplot.axis("off")
    
    # 이미지를 화면에 표시
    # squeeze()로 차원 축소: (1, 28, 28) → (28, 28)
    # cmap="gray"로 그레이스케일 표시
    pyplot.imshow(img.squeeze(), cmap="gray")

# 실제로 이미지를 화면에 띄움
pyplot.show()
```

**출력**: 4×2 그리드로 8개의 MNIST 이미지가 표시됩니다.

**코드 설명**:
- `figure = pyplot.figure(figsize=(12, 6))`: 12×6 인치 크기의 그림 생성
- `figure.add_subplot(rows, cols, i)`: $i$번째 서브플롯 생성
- `img.squeeze()`: 차원이 1인 축 제거 (예: (1, 28, 28) → (28, 28))
- `cmap="gray"`: 그레이스케일 컬러맵 사용

### Step 5.4: 배치 데이터 시각화

```python
import torch
import matplotlib.pyplot as pyplot
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 0: 기호 정리
# - batch: 배치 데이터 (images, labels)
# - images: 배치의 이미지들 (B, C, H, W)
# - labels: 배치의 레이블들 (B,)

# 데이터셋과 DataLoader 생성
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

# 배치 가져오기
images, labels = next(iter(dataloader))

# 3 x 3 그리드로 표시
figure = pyplot.figure(figsize=(9, 9))
cols, rows = 3, 3

for i in range(9):
    figure.add_subplot(rows, cols, i + 1)
    pyplot.title(f"Label: {labels[i].item()}")
    pyplot.axis("off")
    pyplot.imshow(images[i].squeeze(), cmap="gray")

pyplot.show()
```

**출력**: 3×3 그리드로 9개의 MNIST 이미지가 표시됩니다.

---

## 연습 문제

### 문제 1: Dataset 클래스
CSV 파일을 읽어 PyTorch Dataset을 만드는 클래스를 작성하세요.

**힌트**:
```python
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # CSV 파일 읽기
        # transform 저장
        pass
    
    def __len__(self):
        # 데이터셋 크기 반환
        pass
    
    def __getitem__(self, idx):
        # idx번째 샘플 반환
        pass
```

### 문제 2: DataLoader
shuffle=True와 False의 차이점을 코드로 설명하세요.

**힌트**: 위의 Step 4.5 섹션을 참고하세요.

### 문제 3: Transform
이미지를 224x224로 리사이즈하고 정규화하는 transform을 만드세요.

**힌트**:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[...], std=[...])
])
```

### 문제 4: 시각화
배치 데이터에서 랜덤으로 9개 이미지를 그리드로 시각화하세요.

**힌트**: 위의 Step 5.4 섹션을 참고하세요.

### 문제 5: 데이터 분할
전체 데이터를 70:20:10으로 train/val/test로 나누는 코드를 작성하세요.

**힌트**:
```python
from torch.utils.data import random_split

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
```

---

## 핵심 요약

이번 단원에서 우리는 다음을 배웠습니다:

### 1. 데이터 파이프라인
- **Dataset**: 데이터 샘플과 레이블 저장 ($D = \{(x_i, y_i)\}_{i=1}^{N}$)
- **DataLoader**: 배치 생성, 셔플, 멀티프로세싱 제공
- **Transform**: 데이터 전처리 함수 ($x_{processed} = \mathcal{T}(x_{raw})$)

### 2. 배치 처리의 중요성
- 한 번에 여러 샘플 처리하여 효율성 향상
- GPU 활용 극대화
- 배치 크기 $B$로 나누면 $\lceil N/B \rceil$번의 반복으로 처리

### 3. 데이터 전처리
- **Transforms**: 이미지 변환, 정규화 등
- **ToTensor()**: PIL Image → Tensor 변환
- **Normalize**: 데이터 정규화 ($\frac{x - \mu}{\sigma}$)
- **Compose**: 여러 변환을 순차적으로 적용

### 4. 시각화
- Matplotlib으로 데이터 확인
- 이미지와 레이블을 시각적으로 검증
- 배치 데이터의 구조 확인

다음 단원에서는 신경망 구조를 배워보겠습니다.

