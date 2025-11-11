# 09. 데이터 라벨링 (Data Labeling)

이번 단원에서는 머신러닝 모델을 학습시키기 위한 **데이터 라벨링**에 대해 배워보겠습니다.

## 학습 목표
- 데이터 라벨링의 중요성과 목적
- 라벨링 유형별 방법론
- 라벨링 도구 사용법
- 라벨 품질 관리
- PyTorch Dataset 준비

## 이 단원을 배우기 전에

**이전 단원 (08) 복습: Pooling과 Activation의 역할을 이해하셨나요?**

**이전 단원 (04) 복습: Dataset과 DataLoader의 개념을 기억하시나요?**

## 이 단원 다음에는

**다음 단원 (10): 모든 기초를 배웠으니, 이제 첫 번째 완전한 프로젝트를 시작합시다!**

## 왜 이 단원이 필요한가?

**"좋은 데이터가 좋은 모델을 만든다"** - 머신러닝에서 데이터는 모델보다 더 중요할 수 있습니다. 정확하고 일관된 라벨링은 모델 성능의 기반이 됩니다.

- **지도 학습의 필수 요소**: 대부분의 머신러닝 모델은 지도 학습을 사용
- **모델 성능의 기반**: 정확한 라벨 없이는 좋은 모델을 만들 수 없음
- **프로젝트 성공의 핵심**: 데이터 품질이 프로젝트 성공을 좌우

이 단원에서는 AI 프로젝트에서 필수적인 데이터 라벨링의 기초를 배웁니다.

---

## Step 0: 기호 정리

이 단원에서 사용되는 주요 기호와 변수를 정의합니다:

### 수학 기호

#### 데이터셋 관련
- $D = \{(x_i, y_i)\}_{i=1}^{N}$: 데이터셋 (Dataset), $N$개의 샘플-레이블 쌍
- $x_i$: $i$번째 입력 샘플 (예: 이미지, 텍스트)
- $y_i$: $i$번째 레이블 (정답)
- $N$: 전체 데이터셋 크기
- $K$: 클래스 개수 (분류 문제의 경우)

#### 지도 학습 관련
- $X = \{x_1, x_2, \ldots, x_N\}$: 입력 데이터 집합
- $Y = \{y_1, y_2, \ldots, y_N\}$: 레이블 집합
- $f: X \rightarrow Y$: 목표 함수 (ground truth function)
- $\hat{f}: X \rightarrow Y$: 학습된 모델 함수

#### 데이터 분할 관련
- $D_{train}$: 학습 데이터셋 (training set)
- $D_{val}$: 검증 데이터셋 (validation set)
- $D_{test}$: 테스트 데이터셋 (test set)
- $N_{train}$, $N_{val}$, $N_{test}$: 각 데이터셋의 크기

#### 라벨 품질 관련
- $\kappa$: Cohen's Kappa 계수 (라벨러 간 일치도 측정)
- $p_o$: 관찰된 일치 비율 (observed agreement)
- $p_e$: 기대 일치 비율 (expected agreement)
- $IAA$: Inter-Annotator Agreement (라벨러 간 일치도)

### PyTorch 코드에서의 변수

- `dataset`: Dataset 객체
- `dataloader`: DataLoader 객체
- `root_dir`: 데이터셋 루트 디렉토리
- `transform`: 데이터 변환 함수
- `classes`: 클래스 목록
- `class_to_idx`: 클래스 이름을 인덱스로 매핑하는 딕셔너리
- `samples`: (이미지 경로, 레이블) 튜플의 리스트

### 용어 정리

- **라벨링(Labeling)**: 원시 데이터에 정답(label)을 부여하는 작업
- **어노테이션(Annotation)**: 라벨링과 동일한 의미
- **지도 학습(Supervised Learning)**: 입력-출력 쌍으로 학습하는 방법
- **라벨러(Annotator)**: 라벨링을 수행하는 사람 또는 시스템
- **IAA (Inter-Annotator Agreement)**: 여러 라벨러 간의 일치도

---

## Step 1: 데이터 라벨링이란?

### Step 1.1: 왜 라벨링이 필요한가?

#### 문제 상황: 지도 학습의 필요성

**머신러닝 모델의 학습 방식**:
- 대부분의 머신러닝 모델은 **지도 학습(Supervised Learning)**을 사용합니다
- 모델은 입력 데이터와 정답 레이블의 쌍 $(x, y)$을 학습합니다
- 학습 후 새로운 입력 $x'$에 대해 정답 $y'$를 예측합니다

**문제**: 정답 레이블 $y$가 없으면 모델을 학습할 수 없습니다!

#### 해결책: 데이터 라벨링

**라벨링의 목적**:
- 원시 데이터에 정답 레이블을 부여
- 모델이 학습할 수 있는 형태로 데이터 준비
- 모델의 성능을 평가할 수 있는 기준 제공

**예시**:
- 이미지 분류: 이미지에 클래스 레이블 부여 (예: '고양이', '강아지')
- 객체 탐지: 이미지에 바운딩 박스와 클래스 레이블 부여
- 텍스트 분류: 텍스트에 감정 레이블 부여 (예: '긍정', '부정')

### Step 1.2: 지도 학습의 수학적 정의

#### 지도 학습 (Supervised Learning)

**지도 학습**은 다음과 같이 정의됩니다:

주어진 데이터셋 $D = \{(x_i, y_i)\}_{i=1}^{N}$에서:
- $x_i \in \mathcal{X}$: 입력 데이터 (예: 이미지, 텍스트)
- $y_i \in \mathcal{Y}$: 정답 레이블 (예: 클래스, 좌표)
- 목표: 함수 $f: \mathcal{X} \rightarrow \mathcal{Y}$를 학습하여 새로운 입력 $x'$에 대해 정답 $y' = f(x')$를 예측

**학습 과정**:
1. 모델 $\hat{f}$를 초기화
2. 손실 함수 $L(\hat{f}(x_i), y_i)$를 최소화하도록 모델 학습
3. 학습된 모델 $\hat{f}$로 새로운 데이터 예측

**의미**: 입력-출력 쌍을 학습하여 입력에서 출력을 예측하는 패턴을 학습합니다.

### Step 1.3: 라벨링의 유형

#### 1) 이미지 분류 (Image Classification)

**목적**: 이미지 전체를 하나의 클래스로 분류

**수학적 표현**:
- 입력: $x_i \in \mathbb{R}^{C \times H \times W}$ (이미지)
- 출력: $y_i \in \{1, 2, \ldots, K\}$ (클래스 번호)
- 예시: 고양이 vs 강아지, 정상 vs 이상

**라벨 형태**: 단일 클래스 이름 또는 번호

**예시**:
```
이미지: cat_001.jpg → 레이블: "cat"
이미지: dog_001.jpg → 레이블: "dog"
```

#### 2) 객체 탐지 (Object Detection)

**목적**: 이미지 내 객체의 위치와 클래스 찾기

**수학적 표현**:
- 입력: $x_i \in \mathbb{R}^{C \times H \times W}$ (이미지)
- 출력: $y_i = \{(b_j, c_j)\}_{j=1}^{M}$ (바운딩 박스와 클래스 쌍의 집합)
  - $b_j = (x_{min}, y_{min}, x_{max}, y_{max})$: 바운딩 박스 좌표
  - $c_j \in \{1, 2, \ldots, K\}$: 클래스 번호
- 예시: 자율주행, 보안 카메라

**라벨 형태**: 바운딩 박스 좌표 + 클래스

**예시**:
```
이미지: street_001.jpg → 레이블: 
  [
    {"bbox": [100, 50, 200, 150], "class": "car"},
    {"bbox": [300, 100, 400, 200], "class": "person"}
  ]
```

#### 3) 세그멘테이션 (Segmentation)

**목적**: 픽셀 단위로 영역 구분

**수학적 표현**:
- 입력: $x_i \in \mathbb{R}^{C \times H \times W}$ (이미지)
- 출력: $y_i \in \{1, 2, \ldots, K\}^{H \times W}$ (픽셀별 클래스 마스크)
- 예시: 의료 영상, 자율주행

**라벨 형태**: 픽셀별 클래스 마스크

**예시**:
```
이미지: medical_001.jpg → 레이블: (H, W) 크기의 마스크
  각 픽셀: 0 (배경), 1 (장기1), 2 (장기2), ...
```

#### 4) 텍스트 분류/태깅

**목적**: 텍스트의 감정, 주제, 의도 분류

**수학적 표현**:
- 입력: $x_i$ (텍스트 문자열)
- 출력: $y_i \in \{1, 2, \ldots, K\}$ (클래스 번호) 또는 태그 집합
- 예시: 리뷰 감정 분석, 스팸 필터링

**라벨 형태**: 클래스 또는 태그

**예시**:
```
텍스트: "이 영화 정말 재미있어요!" → 레이블: "positive"
텍스트: "배송이 너무 느려요" → 레이블: "negative"
```

---

## Step 2: 라벨링 도구

### Step 2.1: 왜 라벨링 도구가 필요한가?

#### 문제 상황: 수동 라벨링의 한계

**수동 라벨링의 문제**:
1. **비효율성**: 수동으로 라벨링하면 시간이 많이 걸림
2. **일관성 부족**: 여러 사람이 라벨링하면 기준이 다를 수 있음
3. **품질 관리 어려움**: 라벨링 오류를 찾기 어려움
4. **협업 어려움**: 여러 사람이 동시에 작업하기 어려움

#### 해결책: 라벨링 도구

**라벨링 도구의 장점**:
1. **효율성**: 도구를 사용하면 라벨링이 빠름
2. **일관성**: 도구가 기준을 통일하여 일관성 유지
3. **품질 관리**: 도구가 라벨링 오류를 자동으로 검사
4. **협업**: 여러 사람이 동시에 작업 가능

### Step 2.2: 무료 오픈소스 도구

#### CVAT (Computer Vision Annotation Tool)

**용도**: 이미지, 비디오 어노테이션

**특징**:
- 객체 탐지, 세그멘테이션, 키포인트, 3D 쿠베이션
- Docker 또는 웹 버전으로 설치
- URL: https://app.cvat.ai

**장점**:
- 다양한 작업 유형 지원
- 협업 기능 (팀 라벨링)
- 라벨링 품질 관리
- 내보내기 형식 다양 (COCO, YOLO, Pascal VOC 등)

**사용 예시**:
```python
# CVAT는 웹 기반 도구이므로 Python 코드로는 직접 사용하지 않음
# 웹 브라우저에서 https://app.cvat.ai 접속하여 사용

# CVAT에서 내보낸 데이터를 PyTorch에서 사용하는 방법:
# 1. CVAT에서 COCO 형식으로 내보내기
# 2. PyTorch Dataset 클래스에서 COCO 형식 읽기
```

#### Label Studio

**용도**: 이미지, 텍스트, 오디오, 비디오

**특징**:
- 매우 유연한 커스터마이징
- pip install label-studio로 설치

**장점**:
- 다양한 미디어 타입 지원
- 워크플로우 커스터마이징
- ML 모델 통합

**설치 및 사용**:
```python
# Label Studio 설치
# pip install label-studio

# Label Studio 실행 (웹 서버 시작)
# label-studio

# 웹 브라우저에서 http://localhost:8080 접속하여 사용
```

### Step 2.3: 상용 도구

**주요 상용 도구**:
- **Amazon SageMaker Ground Truth**: AWS 통합
- **Google Cloud AutoML**: 자동 라벨링 기능
- **Scale AI**: 전문 라벨링 서비스
- **Labelbox**: 엔터프라이즈급 도구

**상용 도구의 장점**:
- 전문 라벨러 제공
- 자동 라벨링 기능
- 높은 품질 보장
- 빠른 처리 속도

---

## Step 3: 라벨 품질 관리

### Step 3.1: 왜 라벨 품질 관리가 필요한가?

#### 문제 상황: 라벨링 오류의 영향

**라벨링 오류의 영향**:
1. **모델 성능 저하**: 잘못된 라벨로 학습하면 모델 성능이 떨어짐
2. **학습 불안정**: 일관성 없는 라벨로 인해 학습이 불안정해짐
3. **평가 왜곡**: 테스트 데이터의 라벨이 잘못되면 평가가 왜곡됨

**예시**:
- 1000개 샘플 중 50개가 잘못 라벨링됨 (5% 오류)
- 모델이 이 오류를 학습하여 성능이 크게 저하될 수 있음

#### 해결책: 라벨 품질 관리

**품질 관리의 목적**:
1. **오류 감지**: 라벨링 오류를 찾아 수정
2. **일관성 유지**: 여러 라벨러 간의 기준 통일
3. **품질 향상**: 지속적인 개선을 통한 품질 향상

### Step 3.2: 라벨링 가이드라인 수립

**명확하고 상세한 가이드라인**이 필요합니다:

1. **클래스 정의**: 각 클래스의 정확한 정의와 예시
2. **경계 케이스**: 모호한 경우 처리 방법
3. **바운딩 박스 규칙**: 객체를 얼마나 포괄할지
4. **라벨링 정확도 목표**: 95% 이상 권장

**예시: 이미지 분류 가이드라인**:
```
클래스: "cat" (고양이)
- 정의: 집고양이, 길고양이 등 모든 고양이 종류 포함
- 제외: 호랑이, 사자 등 큰 고양이과 동물은 제외
- 경계 케이스: 고양이 털만 보이는 경우 → "cat"으로 라벨링
```

### Step 3.3: 검수(Review) 프로세스

**다층 검수 프로세스**:
1. **Self-review**: 라벨러가 자신의 작업 검수
2. **Peer-review**: 다른 라벨러가 검수
3. **Expert-review**: 전문가가 최종 검수

**검수 기준**:
- 라벨링 정확도: 95% 이상
- 일관성: 여러 라벨러 간 일치도 90% 이상
- 완성도: 모든 데이터에 라벨이 부여되어 있는가?

### Step 3.4: 품질 지표

#### Inter-Annotator Agreement (IAA)

**IAA란?**: 여러 라벨러가 같은 데이터를 라벨링했을 때 일치도 측정

**수학적 정의**:
- $p_o$: 관찰된 일치 비율 (observed agreement)
- $p_e$: 기대 일치 비율 (expected agreement)
- Cohen's Kappa: $\kappa = \frac{p_o - p_e}{1 - p_e}$

**Cohen's Kappa 해석**:
- $\kappa < 0$: 일치도가 기대보다 낮음 (나쁨)
- $0 \leq \kappa < 0.2$: 매우 낮은 일치도
- $0.2 \leq \kappa < 0.4$: 낮은 일치도
- $0.4 \leq \kappa < 0.6$: 보통 일치도
- $0.6 \leq \kappa < 0.8$: 높은 일치도
- $0.8 \leq \kappa \leq 1.0$: 매우 높은 일치도 (우수)

**예시 계산**:
```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

# 두 라벨러의 라벨링 결과
labeler1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 라벨러 1
labeler2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 라벨러 2 (완전히 일치)

# Cohen's Kappa 계산
kappa = cohen_kappa_score(labeler1, labeler2)
print(f"Cohen's Kappa: {kappa:.2f}")  # 1.0 (완전 일치)

# 부분적으로 일치하는 경우
labeler1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
labeler2 = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  # 하나만 다름

kappa = cohen_kappa_score(labeler1, labeler2)
print(f"Cohen's Kappa: {kappa:.2f}")  # 0.8 정도 (높은 일치도)
```

#### 오류 분석

**오류 분석의 목적**:
1. 어떤 클래스에서 오류가 많은지
2. 어떤 라벨러의 정확도가 낮은지
3. 가이드라인 개선 포인트

**예시**:
```python
# 오류 분석 예시
import pandas as pd
from sklearn.metrics import confusion_matrix

# 실제 레이블과 예측 레이블
true_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
predicted_labels = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]  # 하나 오류

# 혼동 행렬 (Confusion Matrix)
cm = confusion_matrix(true_labels, predicted_labels)
print("혼동 행렬:")
print(cm)
# 출력:
# [[5 0]  # 클래스 0: 5개 정확, 0개 오류
#  [1 4]] # 클래스 1: 1개 오류, 4개 정확

# 오류율 계산
error_rate = 1 - (cm[0, 0] + cm[1, 1]) / cm.sum()
print(f"오류율: {error_rate:.2%}")  # 10%
```

---

## Step 4: 라벨링 워크플로우

### Step 4.1: 전체 프로세스

**라벨링 워크플로우**:

```
1. 데이터 수집
   ↓
2. 데이터 전처리
   ↓
3. 가이드라인 작성
   ↓
4. 도구 선택 및 설정
   ↓
5. 샘플 라벨링 (파일럿)
   ↓
6. 품질 검수 및 가이드라인 개선
   ↓
7. 전체 라벨링
   ↓
8. 최종 검수
   ↓
9. 데이터셋 내보내기
   ↓
10. PyTorch Dataset 준비
```

### Step 4.2: 데이터 분할

#### 왜 데이터를 분할해야 하는가?

**문제 상황**: 같은 데이터로 학습하고 평가하면?

**문제점**:
- 모델이 학습 데이터를 외워버림 (과적합)
- 실제 성능을 제대로 평가할 수 없음
- 일반화 성능을 알 수 없음

**해결책**: 데이터를 Train/Val/Test로 분할

#### 데이터 분할의 수학적 정의

**데이터셋 분할**:

$$D = D_{train} \cup D_{val} \cup D_{test}$$

여기서:
- $D_{train} \cap D_{val} = \emptyset$
- $D_{train} \cap D_{test} = \emptyset$
- $D_{val} \cap D_{test} = \emptyset$

**각 데이터셋의 역할**:
- **Train (학습)**: 모델 학습용 (보통 70-80%)
- **Validation (검증)**: 하이퍼파라미터 튜닝용 (10-15%)
- **Test (테스트)**: 최종 성능 평가용 (10-15%)

**중요**: Test 세트는 모델 학습 전에 분리하여 보관!

#### 데이터 분할 예시

```python
import torch
from torch.utils.data import Dataset, random_split

# 전체 데이터셋 크기
total_size = 1000
train_size = int(0.7 * total_size)  # 70%
val_size = int(0.15 * total_size)   # 15%
test_size = total_size - train_size - val_size  # 15%

print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

# 데이터셋 분할 (예시)
# 실제로는 Dataset 클래스를 사용하여 분할
class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

# 전체 데이터셋 생성
full_dataset = DummyDataset(total_size)

# 분할
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, val_size, test_size]
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
```

---

## Step 5: 실습: 이미지 분류 데이터셋 준비

### Step 5.1: 데이터셋 구조

#### 이미지 분류를 위한 데이터셋 구조

**일반적인 데이터셋 구조**:

```
dataset/
  train/
    class1/
      image1.jpg
      image2.jpg
    class2/
      image3.jpg
      image4.jpg
  val/
    class1/
      image5.jpg
    class2/
      image6.jpg
  test/
    class1/
      image7.jpg
    class2/
      image8.jpg
```

**의미**: 각 클래스별로 폴더를 만들고 이미지를 배치합니다.

#### 데이터셋 구조 생성 예시

```python
from pathlib import Path
import shutil
import os

# 데이터셋 구조 예시 (실제로는 데이터를 준비해야 함)
dataset_root = Path("dataset")
train_dir = dataset_root / "train"
val_dir = dataset_root / "test"
test_dir = dataset_root / "test"

# 클래스 목록
classes = ["cat", "dog"]

# 디렉토리 생성 (실제로는 데이터를 준비해야 함)
# for split in [train_dir, val_dir, test_dir]:
#     for cls in classes:
#         (split / cls).mkdir(parents=True, exist_ok=True)

print("이것이 이미지 분류를 위한 일반적인 데이터셋 구조입니다.")
print("각 클래스별로 폴더를 만들고 이미지를 배치합니다.")
```

### Step 5.2: PyTorch Dataset 구현

#### Step 5.2.1: Dataset 클래스의 수학적 정의

**Dataset 클래스**:

데이터셋 $D = \{(x_i, y_i)\}_{i=1}^{N}$에서:
- `__len__()`: 데이터셋 크기 $N$ 반환
- `__getitem__(i)`: $i$번째 샘플 $(x_i, y_i)$ 반환

**의미**: 데이터셋을 인덱스로 접근할 수 있는 구조를 제공합니다.

#### Step 5.2.2: ImageClassificationDataset 구현

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ImageClassificationDataset(Dataset):
    """
    이미지 분류를 위한 커스텀 Dataset
    
    Args:
        root_dir: 데이터셋 루트 디렉토리
        transform: 이미지 변환 (전처리)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 클래스 목록 수집
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 이미지 경로와 라벨 수집
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            label = self.class_to_idx[cls_name]
            
            # 이미지 파일만 수집
            for img_path in cls_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 오류: {img_path}, {e}")
            # 오류 발생 시 더미 이미지 반환
            image = Image.new('RGB', (224, 224), color='black')
        
        # 전처리
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """클래스 이름 목록 반환"""
        return self.classes

# 사용 예시 (실제 데이터가 있을 때)
# dataset = ImageClassificationDataset(
#     root_dir='dataset/train',
#     transform=None
# )
# print(f"데이터셋 크기: {len(dataset)}")
# print(f"클래스: {dataset.get_class_names()}")
```

#### Step 5.2.3: Dataset 사용 예시

```python
from torchvision import transforms
from torch.utils.data import DataLoader

# 데이터 변환 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Dataset 생성 (실제 경로로 교체)
# train_dataset = ImageClassificationDataset(
#     root_dir='dataset/train',
#     transform=train_transform
# )
# 
# val_dataset = ImageClassificationDataset(
#     root_dir='dataset/val',
#     transform=val_transform
# )

# DataLoader 생성
# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=32, 
#     shuffle=True,
#     num_workers=4
# )
# 
# val_loader = DataLoader(
#     val_dataset, 
#     batch_size=32, 
#     shuffle=False,
#     num_workers=4
# )

print("Dataset과 DataLoader 사용 예시입니다.")
print("실제 데이터 경로를 지정하면 바로 사용할 수 있습니다.")
```

### Step 5.3: 데이터셋 검증

#### 데이터셋 품질 검사

```python
import torch
from torch.utils.data import DataLoader
from collections import Counter

# 데이터셋 검증 함수
def validate_dataset(dataset):
    """데이터셋의 품질을 검증"""
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"클래스: {dataset.get_class_names()}")
    
    # 클래스별 샘플 수 확인
    labels = [dataset[i][1] for i in range(len(dataset))]
    label_counts = Counter(labels)
    
    print("\n클래스별 샘플 수:")
    for idx, count in sorted(label_counts.items()):
        class_name = dataset.idx_to_class[idx]
        print(f"  {class_name}: {count}")
    
    # 데이터 불균형 확인
    counts = list(label_counts.values())
    if len(counts) > 1:
        imbalance_ratio = max(counts) / min(counts)
        print(f"\n불균형 비율: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2.0:
            print("  경고: 데이터가 불균형합니다!")

# 사용 예시
# validate_dataset(train_dataset)
```

---

## 연습 문제

### 문제 1: 데이터셋 구조 이해

다음 데이터셋 구조가 무엇을 의미하는지 설명하세요:

```
dataset/
  cat/
    cat001.jpg
    cat002.jpg
  dog/
    dog001.jpg
    dog002.jpg
```

**답**:
- 이미지 분류 데이터셋
- 클래스별로 폴더를 나눔
- `cat` 폴더: 고양이 이미지
- `dog` 폴더: 강아지 이미지
- 각 이미지의 파일명이 클래스를 나타냄

### 문제 2: 라벨링 도구 비교

CVAT와 Label Studio의 차이점을 조사하고, 각각의 장단점을 정리하세요.

**답**:
- **CVAT**: 이미지/비디오 전문, 객체 탐지/세그멘테이션에 특화
- **Label Studio**: 범용 도구, 다양한 미디어 타입 지원, 커스터마이징 용이

### 문제 3: 커스텀 Dataset 작성

다른 형태의 데이터셋(예: CSV 파일로 경로와 라벨이 저장된 경우)을 위한 Dataset 클래스를 작성하세요.

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CSVDataset(Dataset):
    """
    CSV 파일로 경로와 라벨이 저장된 데이터셋
    CSV 형식: image_path, label
    """
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # 클래스 목록 추출
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label_str = self.df.iloc[idx]['label']
        label = self.class_to_idx[label_str]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 전처리
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 사용 예시
# dataset = CSVDataset('data.csv', transform=train_transform)
```

### 문제 4: 데이터 분할

전체 데이터를 train(70%), val(20%), test(10%)로 나누는 코드를 작성하세요.

```python
import torch
from torch.utils.data import Dataset, random_split

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    데이터셋을 train/val/test로 분할
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이어야 함"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드
    )
    
    return train_dataset, val_dataset, test_dataset

# 사용 예시
# full_dataset = ImageClassificationDataset('dataset', transform=None)
# train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
```

### 문제 5: 라벨 품질 검사

데이터셋에 불균형(imbalance)이 있을 때 어떻게 처리할 수 있을지 설명하세요.

**답**:
1. **오버샘플링 (Oversampling)**: 소수 클래스의 샘플을 증가
2. **언더샘플링 (Undersampling)**: 다수 클래스의 샘플을 감소
3. **가중치 조정 (Class Weighting)**: 손실 함수에 클래스 가중치 적용
4. **데이터 증강 (Data Augmentation)**: 소수 클래스의 데이터를 변환하여 증가

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

def create_weighted_sampler(dataset):
    """불균형 데이터셋을 위한 가중치 샘플러 생성"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# 사용 예시
# sampler = create_weighted_sampler(train_dataset)
# train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

---

## 핵심 요약

### 1. 라벨링의 중요성
- 좋은 라벨 = 좋은 모델
- 정확하고 일관된 라벨링 필수
- 지도 학습의 기반

### 2. 라벨링 도구
- **CVAT**: 이미지/비디오 전문
- **Label Studio**: 범용 도구
- 상용 도구: SageMaker, AutoML 등

### 3. 품질 관리
- 명확한 가이드라인
- 다층 검수 프로세스
- IAA 측정 (Cohen's Kappa)
- 오류 분석

### 4. 워크플로우
- 파일럿 → 검수 → 전체 라벨링 → 검수 → 내보내기
- Train/Val/Test 분할
- 데이터셋 구조화

### 5. PyTorch Dataset
- 커스텀 Dataset 클래스 구현
- `__len__()`, `__getitem__()` 메서드 구현
- transforms로 전처리
- DataLoader로 배치 처리

다음 단원에서는 **첫 번째 프로젝트**를 시작해보겠습니다!

