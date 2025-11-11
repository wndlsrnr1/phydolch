# 09. 데이터 라벨링 (Data Labeling)

이번 단원에서는 머신러닝 모델을 학습시키기 위한 **데이터 라벨링**에 대해 배워보겠습니다.

## 학습 목표
- 데이터 라벨링의 중요성과 목적
- 라벨링 유형별 방법론
- 라벨링 도구 사용법
- 라벨 품질 관리
- PyTorch Dataset 준비

## 왜 이 단원이 필요한가?

**"좋은 데이터가 좋은 모델을 만든다"** - 머신러닝에서 데이터는 모델보다 더 중요할 수 있습니다. 정확하고 일관된 라벨링은 모델 성능의 기반이 됩니다. 이 단원에서는 AI 프로젝트에서 필수적인 데이터 라벨링의 기초를 배웁니다.

## 이 단원을 배우기 전에

**이전 단원 (08) 복습: Pooling과 Activation의 역할을 이해하셨나요?**

## 이 단원 다음에는

**다음 단원 (10): 모든 기초를 배웠으니, 이제 첫 번째 완전한 프로젝트를 시작합시다!**

## 1. 데이터 라벨링이란?

**데이터 라벨링(Data Labeling)**은 원시 데이터에 **정답(label)**을 부여하는 작업입니다.

### 1.1 학습 지도 (Supervised Learning)의 필요성

대부분의 머신러닝 모델은 **지도 학습(Supervised Learning)**을 사용합니다:

- **입력(X)**: 이미지, 텍스트, 음성 등 원시 데이터
- **출력(y)**: 정답 레이블 (예: '고양이', '강아지' 또는 좌표, 클래스 등)

모델은 (X, y) 쌍을 학습하여 입력에서 출력을 예측하는 패턴을 학습합니다.

### 1.2 라벨링의 유형

#### 1) 이미지 분류 (Image Classification)
- **목적**: 이미지 전체를 하나의 클래스로 분류
- **예시**: 고양이 vs 강아지, 정상 vs 이상
- **라벨**: 단일 클래스 이름

#### 2) 객체 탐지 (Object Detection)
- **목적**: 이미지 내 객체의 위치와 클래스 찾기
- **예시**: 자율주행, 보안 카메라
- **라벨**: 바운딩 박스 좌표 + 클래스

#### 3) 세그멘테이션 (Segmentation)
- **목적**: 픽셀 단위로 영역 구분
- **예시**: 의료 영상, 자율주행
- **라벨**: 픽셀별 클래스 마스크

#### 4) 텍스트 분류/태깅
- **목적**: 텍스트의 감정, 주제, 의도 분류
- **예시**: 리뷰 감정 분석, 스팸 필터링
- **라벨**: 클래스 또는 태그

## 2. 라벨링 도구

### 2.1 무료 오픈소스 도구

#### CVAT (Computer Vision Annotation Tool)
- **용도**: 이미지, 비디오 어노테이션
- **특징**: 객체 탐지, 세그멘테이션, 키포인트, 3D 쿠베이션
- **설치**: Docker 또는 웹 버전
- **URL**: https://app.cvat.ai

**장점**:
- 다양한 작업 유형 지원
- 협업 기능 (팀 라벨링)
- 라벨링 품질 관리
- 내보내기 형식 다양 (COCO, YOLO, Pascal VOC 등)

#### Label Studio
- **용도**: 이미지, 텍스트, 오디오, 비디오
- **특징**: 매우 유연한 커스터마이징
- **설치**: pip install label-studio

**장점**:
- 다양한 미디어 타입 지원
- 워크플로우 커스터마이징
- ML 모델 통합

### 2.2 상용 도구

- **Amazon SageMaker Ground Truth**: AWS 통합
- **Google Cloud AutoML**: 자동 라벨링 기능
- **Scale AI**: 전문 라벨링 서비스
- **Labelbox**: 엔터프라이즈급 도구

## 3. 라벨 품질 관리

### 3.1 라벨링 가이드라인 수립

**명확하고 상세한 가이드라인**이 필요합니다:

1. **클래스 정의**: 각 클래스의 정확한 정의와 예시
2. **경계 케이스**: 모호한 경우 처리 방법
3. **바운딩 박스 규칙**: 객체를 얼마나 포괄할지
4. **라벨링 정확도 목표**: 95% 이상 권장

### 3.2 검수(Review) 프로세스

- **Self-review**: 라벨러가 자신의 작업 검수
- **Peer-review**: 다른 라벨러가 검수
- **Expert-review**: 전문가가 최종 검수

### 3.3 품질 지표

#### Inter-Annotator Agreement (IAA)
- 여러 라벨러가 같은 데이터를 라벨링했을 때 일치도 측정
- Cohen's Kappa, Fleiss' Kappa 등 사용

#### 오류 분석
- 어떤 클래스에서 오류가 많은지
- 어떤 라벨러의 정확도가 낮은지
- 가이드라인 개선 포인트

## 4. 라벨링 워크플로우

### 4.1 전체 프로세스

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

### 4.2 데이터 분할

라벨링 후에는 반드시 데이터를 나눕니다:

- **Train (학습)**: 모델 학습용 (보통 70-80%)
- **Validation (검증)**: 하이퍼파라미터 튜닝용 (10-15%)
- **Test (테스트)**: 최종 성능 평가용 (10-15%)

**중요**: Test 세트는 모델 학습 전에 분리하여 보관!

## 5. 실습: 이미지 분류 데이터셋 준비

간단한 예제로 이미지 분류 데이터셋을 만들어보겠습니다.

```python
from pathlib import Path
import shutil

# 데이터셋 구조 예시
# dataset/
#   train/
#     class1/
#       image1.jpg
#       image2.jpg
#     class2/
#       image3.jpg
#       image4.jpg
#   val/
#     class1/
#     class2/
#   test/
#     class1/
#     class2/

print("이것이 이미지 분류를 위한 일반적인 데이터셋 구조입니다.")
print("각 클래스별로 폴더를 만들고 이미지를 배치합니다.")
```

### 5.1 PyTorch Dataset 구현

라벨링된 데이터를 PyTorch가 사용할 수 있도록 Dataset 클래스를 만들어봅시다.

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

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
        
        # 이미지 경로와 라벨 수집
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            label = self.class_to_idx[cls_name]
            
            for img_path in cls_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 전처리
        if self.transform:
            image = self.transform(image)
        
        return image, label

print("Dataset 클래스 구현 완료!")
```

### 5.2 사용 예시

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

# Dataset 생성 (실제 경로로 교체)
# train_dataset = ImageClassificationDataset(
#     root_dir='data/train',
#     transform=train_transform
# )

# DataLoader 생성
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Dataset과 DataLoader 사용 예시입니다.")
print("실제 데이터 경로를 지정하면 바로 사용할 수 있습니다.")
```

## 6. 연습 문제

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

### 문제 2: 라벨링 도구 비교

CVAT와 Label Studio의 차이점을 조사하고, 각각의 장단점을 정리하세요.

### 문제 3: 커스텀 Dataset 작성

다른 형태의 데이터셋(예: CSV 파일로 경로와 라벨이 저장된 경우)을 위한 Dataset 클래스를 작성하세요.

### 문제 4: 데이터 분할

전체 데이터를 train(70%), val(20%), test(10%)로 나누는 코드를 작성하세요.

### 문제 5: 라벨 품질 검사

데이터셋에 불균형(imbalance)이 있을 때 어떻게 처리할 수 있을지 설명하세요.

## 핵심 요약

### 1. 라벨링의 중요성
- 좋은 라벨 = 좋은 모델
- 정확하고 일관된 라벨링 필수

### 2. 라벨링 도구
- CVAT: 이미지/비디오 전문
- Label Studio: 범용 도구
- 상용 도구: SageMaker, AutoML 등

### 3. 품질 관리
- 명확한 가이드라인
- 다층 검수 프로세스
- IAA 측정

### 4. 워크플로우
- 파일럿 → 검수 → 전체 라벨링 → 검수 → 내보내기
- Train/Val/Test 분할

### 5. PyTorch Dataset
- 커스텀 Dataset 클래스 구현
- transforms로 전처리
- DataLoader로 배치 처리

다음 단원에서는 **첫 번째 프로젝트**를 시작해보겠습니다!
