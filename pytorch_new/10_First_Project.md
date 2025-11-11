# 10. 첫 번째 프로젝트: MNIST 손글씨 인식

## 목표·범위·평가 기준

**목표**: MNIST 분류 파이프라인을 완전히 구현하고, 점진적 모델 개선 과정을 경험한다.

**출력물**: 검증 정확도 95% 이상 달성 모델 + 실험 기록표(Logistic Regression, MLP, CNN 비교) + 혼동행렬 및 오류 분석 리포트

**합격선**: 동일 환경에서 동일 시드로 실행 시 ±0.5% 이내 재현 가능한 결과

---

## Step 0: 환경 설정 및 기호 정리

### 이번 단계 목표
- 재현 가능한 실험 환경 구축
- 문서 전체에서 사용할 기호와 변수 정의

### 입력
- 없음 (환경 설정)

### 출력
- 고정된 환경 설정
- 기호 정의표

### 평가 기준
- 동일 환경에서 동일 결과 재현 가능

### 환경 설정

**재현성을 위한 필수 설정**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import os

# ============================================
# 환경 설정 (재현성 보장)
# ============================================

# PyTorch 버전 확인
print(f"PyTorch 버전: {torch.__version__}")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 난수 시드 고정 (재현성)
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 경고: GPU/CPU 환경 차이
if device.type == 'cpu':
    print("⚠️  CPU 모드: 학습 시간이 오래 걸릴 수 있습니다.")
else:
    print("✅ GPU 모드: 빠른 학습이 가능합니다.")

# 결과 저장 디렉토리
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
```

### 기호 정리

**역할 표 (전체 프로젝트 공통)**:

| 역할 | 기호/변수 | 설명 | Shape 예시 |
|------|-----------|------|------------|
| 입력 이미지 | `x`, `images` | MNIST 손글씨 이미지 | `(B, 1, 28, 28)` |
| 정답 라벨 | `y`, `labels` | 0~9 숫자 클래스 | `(B,)` |
| 모델 출력 | `logits`, `outputs` | 클래스별 점수 (logits) | `(B, 10)` |
| 예측 클래스 | `pred`, `predicted` | argmax 결과 | `(B,)` |
| 손실 함수 | `loss`, `criterion` | CrossEntropyLoss | 스칼라 |
| 옵티마이저 | `optimizer` | Adam/SGD | - |
| 배치 크기 | `B`, `batch_size` | 한 번에 처리할 샘플 수 | 64 (기본값) |
| 에폭 수 | `num_epochs`, `epochs` | 전체 데이터셋 순회 횟수 | 10 (기본값) |
| 학습률 | `lr`, `learning_rate` | 옵티마이저 학습률 | 0.001 (기본값) |

**데이터 분할 전략 (표준 고정)**:

- **학습 집합 (Train)**: 50,000개 (모델 학습)
- **검증 집합 (Validation)**: 10,000개 (하이퍼파라미터 튜닝, 모델 선택)
- **테스트 집합 (Test)**: 10,000개 (최종 성능 평가, 1회만 사용)

**실험 설계 원칙**:

- **고정 변수**: 데이터셋(MNIST), 전처리(Normalize), 에폭 수(10), 배치 크기(64)
- **조작 변수**: 모델 아키텍처(Logistic Regression → MLP → CNN), 학습률(0.001, 0.0001, 0.01)

---

## Step 1: 데이터 준비

### 이번 단계 목표
- MNIST 데이터셋 다운로드 및 train/val/test 분할
- 데이터 전처리 파이프라인 구축
- 데이터 시각화 및 통계 확인

### 입력
- 없음 (데이터 다운로드)

### 출력
- `train_loader`, `val_loader`, `test_loader` (DataLoader 객체)
- 데이터 통계 정보

### 평가 기준
- train: 50,000개, val: 10,000개, test: 10,000개 정확히 분할
- 정규화된 텐서 형태로 변환 완료

### 데이터 다운로드 및 분할

```python
# ============================================
# 데이터 전처리 정의
# ============================================

# MNIST 표준 정규화 파라미터 (고정)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image → Tensor [0, 1]
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))  # 정규화
])

# 전체 학습 데이터셋 다운로드 (60,000개)
full_train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 테스트 데이터셋 (10,000개)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 학습 데이터를 train/val로 분할 (50,000 / 10,000)
train_size = 50000
val_size = 10000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)  # 재현성
)

print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")
print(f"테스트 데이터: {len(test_dataset)}개")
print(f"총 클래스 수: {len(full_train_dataset.classes)}개 (0~9)")
```

### DataLoader 생성

```python
# ============================================
# DataLoader 생성 (효율성 설정)
# ============================================

BATCH_SIZE = 64
NUM_WORKERS = 4 if device.type == 'cuda' else 0  # GPU일 때 병렬 로딩

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 학습 시 셔플
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 검증 시 셔플 불필요
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == 'cuda' else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 테스트 시 셔플 불필요
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == 'cuda' else False
)

print("✅ DataLoader 생성 완료")
```

### 데이터 시각화 및 통계

```python
# ============================================
# 데이터 탐색 및 시각화
# ============================================

# 샘플 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx in range(10):
    image, label = train_dataset[idx]
    axes[idx // 5, idx % 5].imshow(image.squeeze().numpy(), cmap='gray')
    axes[idx // 5, idx % 5].set_title(f'Label: {label}')
    axes[idx // 5, idx % 5].axis('off')
plt.suptitle('MNIST 학습 데이터 샘플', fontsize=14)
plt.tight_layout()
plt.savefig('results/data_samples.png', dpi=150)
plt.show()

# 데이터 통계
sample_image, _ = train_dataset[0]
print(f"\n이미지 shape: {sample_image.shape}")
print(f"픽셀 값 범위: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
print(f"평균: {sample_image.mean():.3f}, 표준편차: {sample_image.std():.3f}")

# 클래스 분포 확인
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]

print(f"\n학습 데이터 클래스 분포:")
for i in range(10):
    print(f"  클래스 {i}: {train_labels.count(i)}개")
```

---

## Step 2: 모델 정의 - 점진적 접근

### 이번 단계 목표
- Logistic Regression, MLP, CNN 세 가지 모델 구현
- 각 모델의 파라미터 수와 구조 이해

### 입력
- 모델 클래스 정의

### 출력
- `LogisticRegression`, `SimpleMLP`, `SimpleCNN` 클래스

### 평가 기준
- 각 모델이 올바른 shape로 forward pass 수행
- 파라미터 수 계산 가능

### 모델 1: Logistic Regression (베이스라인)

**의미**: 가장 간단한 선형 분류기. 784개 픽셀을 직접 10개 클래스로 매핑.

```python
# ============================================
# 모델 1: Logistic Regression (베이스라인)
# ============================================

class LogisticRegression(nn.Module):
    """
    가장 간단한 선형 분류기
    - 입력: (B, 1, 28, 28) → 평탄화 → (B, 784)
    - 출력: (B, 10) 클래스 점수
    """
    def __init__(self, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, num_classes)
    
    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = x.view(-1, 28 * 28)  # (B, 784)
        x = self.linear(x)  # (B, 10)
        return x

# 모델 생성 및 파라미터 확인
model_lr = LogisticRegression(num_classes=10).to(device)
num_params_lr = sum(p.numel() for p in model_lr.parameters())

print("=" * 60)
print("모델 1: Logistic Regression (베이스라인)")
print("=" * 60)
print(model_lr)
print(f"\n파라미터 수: {num_params_lr:,}개")
print(f"의미: 784개 픽셀 각각에 가중치를 두고 클래스 점수 계산")
```

### 모델 2: MLP (Multi-Layer Perceptron)

**의미**: 은닉층을 추가하여 비선형 표현력 향상.

```python
# ============================================
# 모델 2: MLP (Multi-Layer Perceptron)
# ============================================

class SimpleMLP(nn.Module):
    """
    다층 신경망
    - 입력: (B, 1, 28, 28) → 평탄화 → (B, 784)
    - 은닉층 1: (B, 784) → (B, 128)
    - 은닉층 2: (B, 128) → (B, 64)
    - 출력: (B, 64) → (B, 10)
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # (B, 784)
        x = self.relu(self.fc1(x))  # (B, 128)
        x = self.dropout(x)  # 학습 시에만 적용
        x = self.relu(self.fc2(x))  # (B, 64)
        x = self.dropout(x)
        x = self.fc3(x)  # (B, 10)
        return x

# 모델 생성 및 파라미터 확인
model_mlp = SimpleMLP(num_classes=10).to(device)
num_params_mlp = sum(p.numel() for p in model_mlp.parameters())

print("\n" + "=" * 60)
print("모델 2: MLP (다층 신경망)")
print("=" * 60)
print(model_mlp)
print(f"\n파라미터 수: {num_params_mlp:,}개")
print(f"의미: 은닉층이 중간 패턴을 학습 → 표현력 향상")
```

### 모델 3: CNN (Convolutional Neural Network)

**의미**: 컨볼루션 레이어로 공간적 패턴 학습.

```python
# ============================================
# 모델 3: CNN (Convolutional Neural Network)
# ============================================

class SimpleCNN(nn.Module):
    """
    컨볼루션 신경망
    - Conv1: (B, 1, 28, 28) → (B, 32, 26, 26) [3x3 conv, padding=0]
    - Pool1: (B, 32, 26, 26) → (B, 32, 13, 13) [2x2 maxpool]
    - Conv2: (B, 32, 13, 13) → (B, 64, 11, 11) [3x3 conv]
    - Pool2: (B, 64, 11, 11) → (B, 64, 5, 5) [2x2 maxpool]
    - FC: (B, 64*5*5) → (B, 128) → (B, 10)
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        # 컨볼루션 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # 완전연결 레이어
        # Conv2 출력: (64, 5, 5) = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.relu(self.conv1(x))  # (B, 32, 26, 26)
        x = self.pool(x)  # (B, 32, 13, 13)
        x = self.relu(self.conv2(x))  # (B, 64, 11, 11)
        x = self.pool(x)  # (B, 64, 5, 5)
        x = x.view(-1, 64 * 5 * 5)  # (B, 1600)
        x = self.relu(self.fc1(x))  # (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 10)
        return x

# 모델 생성 및 파라미터 확인
model_cnn = SimpleCNN(num_classes=10).to(device)
num_params_cnn = sum(p.numel() for p in model_cnn.parameters())

print("\n" + "=" * 60)
print("모델 3: CNN (컨볼루션 신경망)")
print("=" * 60)
print(model_cnn)
print(f"\n파라미터 수: {num_params_cnn:,}개")
print(f"의미: 컨볼루션으로 공간적 패턴 학습 → 이미지 특화")
```

### 모델 구조 검증

```python
# ============================================
# 모델 구조 검증 (shape 확인)
# ============================================

def verify_model(model, model_name):
    """모델의 forward pass가 올바른지 검증"""
    model.eval()
    dummy_input = torch.randn(2, 1, 28, 28).to(device)  # 배치 크기 2
    with torch.no_grad():
        output = model(dummy_input)
    print(f"{model_name}: 입력 {dummy_input.shape} → 출력 {output.shape} ✅")

verify_model(model_lr, "Logistic Regression")
verify_model(model_mlp, "MLP")
verify_model(model_cnn, "CNN")
```

---

## Step 3: 학습 및 검증 루프 구현

### 이번 단계 목표
- 훈련 루프와 검증 루프 분리 구현
- 최고 검증 성능 기준으로 체크포인트 저장
- 수치 안정성 장치 추가 (Gradient clipping, NaN 체크)

### 입력
- 모델, DataLoader, 손실 함수, 옵티마이저

### 출력
- 학습/검증 손실 및 정확도 히스토리
- 최고 검증 성능 체크포인트

### 평가 기준
- 훈련/검증 루프가 올바르게 분리되어 실행
- 최고 검증 성능이 기록되고 저장됨

### 학습 함수 (훈련 루프)

```python
# ============================================
# 학습 함수 (훈련 루프)
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device, clip_grad_norm=1.0):
    """
    한 epoch 학습
    
    입력:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        clip_grad_norm: Gradient clipping 임계값
    
    출력:
        epoch_loss: 평균 손실
        epoch_acc: 평균 정확도 (%)
    """
    model.train()  # 학습 모드
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # 디바이스로 이동
        images, labels = images.to(device), labels.to(device)
        
        # Gradient 초기화
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # (B, 10)
        loss = criterion(outputs, labels)
        
        # NaN 체크 (수치 안정성)
        if torch.isnan(loss):
            print("⚠️  경고: NaN 손실 발생! 학습 중단.")
            raise ValueError("NaN loss detected")
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (수치 안정성)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc
```

### 검증 함수 (검증 루프)

```python
# ============================================
# 검증 함수 (검증 루프)
# ============================================

def validate(model, val_loader, criterion, device):
    """
    검증 수행
    
    입력:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 디바이스
    
    출력:
        val_loss: 평균 손실
        val_acc: 평균 정확도 (%)
        all_preds: 모든 예측 (혼동행렬용)
        all_labels: 모든 정답 (혼동행렬용)
    """
    model.eval()  # 평가 모드
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Gradient 계산 비활성화
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 혼동행렬용 저장
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels
```

### 학습 실행 함수

```python
# ============================================
# 학습 실행 함수 (전체 파이프라인)
# ============================================

def train_model(model, model_name, train_loader, val_loader, 
                num_epochs=10, lr=0.001, weight_decay=1e-4):
    """
    모델 학습 전체 파이프라인
    
    입력:
        model: 학습할 모델
        model_name: 모델 이름 (저장용)
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: 에폭 수
        lr: 학습률
        weight_decay: L2 정규화 계수
    
    출력:
        history: 학습/검증 히스토리 딕셔너리
        best_model_state: 최고 검증 성능 모델 상태
    """
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 학습률 스케줄러 (수치 안정성 및 성능 향상)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 히스토리 저장
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 최고 검증 성능 추적
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"모델 학습 시작: {model_name}")
    print(f"{'='*60}")
    print(f"에폭 수: {num_epochs}, 학습률: {lr}, Weight Decay: {weight_decay}")
    print(f"{'-'*60}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 훈련
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 검증
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 최고 검증 성능 업데이트
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        # 진행 상황 출력
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    elapsed_time = time.time() - start_time
    
    print(f"{'-'*60}")
    print(f"학습 완료! 소요 시간: {elapsed_time:.2f}초")
    print(f"최고 검증 정확도: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*60}\n")
    
    # 최고 모델 상태 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_model_state
```

---

## Step 4: 모델 학습 실행 및 비교

### 이번 단계 목표
- 세 가지 모델을 동일 조건에서 학습
- 실험 기록표 작성
- 학습 곡선 시각화

### 입력
- 세 가지 모델 (Logistic Regression, MLP, CNN)
- 학습/검증 데이터 로더

### 출력
- 실험 기록표 (모델별 성능 비교)
- 학습 곡선 그래프

### 평가 기준
- 세 모델 모두 학습 완료
- 실험 기록표에 모든 정보 포함

### 실험 실행

```python
# ============================================
# 실험 실행: 세 가지 모델 비교
# ============================================

# 실험 설정 (고정 변수)
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# 실험 기록 저장
experiment_results = []

# 모델 1: Logistic Regression
print("\n" + "="*60)
print("실험 1: Logistic Regression")
print("="*60)

model_lr = LogisticRegression(num_classes=10).to(device)
history_lr, best_state_lr = train_model(
    model_lr, "Logistic Regression",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

experiment_results.append({
    'model': 'Logistic Regression',
    'num_params': sum(p.numel() for p in model_lr.parameters()),
    'best_val_acc': max(history_lr['val_acc']),
    'final_train_acc': history_lr['train_acc'][-1],
    'final_val_acc': history_lr['val_acc'][-1],
    'history': history_lr,
    'best_state': best_state_lr
})

# 모델 2: MLP
print("\n" + "="*60)
print("실험 2: MLP")
print("="*60)

model_mlp = SimpleMLP(num_classes=10).to(device)
history_mlp, best_state_mlp = train_model(
    model_mlp, "MLP",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

experiment_results.append({
    'model': 'MLP',
    'num_params': sum(p.numel() for p in model_mlp.parameters()),
    'best_val_acc': max(history_mlp['val_acc']),
    'final_train_acc': history_mlp['train_acc'][-1],
    'final_val_acc': history_mlp['val_acc'][-1],
    'history': history_mlp,
    'best_state': best_state_mlp
})

# 모델 3: CNN
print("\n" + "="*60)
print("실험 3: CNN")
print("="*60)

model_cnn = SimpleCNN(num_classes=10).to(device)
history_cnn, best_state_cnn = train_model(
    model_cnn, "CNN",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

experiment_results.append({
    'model': 'CNN',
    'num_params': sum(p.numel() for p in model_cnn.parameters()),
    'best_val_acc': max(history_cnn['val_acc']),
    'final_train_acc': history_cnn['train_acc'][-1],
    'final_val_acc': history_cnn['val_acc'][-1],
    'history': history_cnn,
    'best_state': best_state_cnn
})
```

### 실험 기록표 작성

```python
# ============================================
# 실험 기록표
# ============================================

print("\n" + "="*80)
print("실험 기록표")
print("="*80)
print(f"{'모델':<25} {'파라미터 수':<15} {'최고 검증 정확도':<20} {'최종 검증 정확도':<20}")
print("-"*80)

for result in experiment_results:
    print(f"{result['model']:<25} "
          f"{result['num_params']:>13,}개  "
          f"{result['best_val_acc']:>18.2f}%  "
          f"{result['final_val_acc']:>18.2f}%")

print("="*80)

# 최고 성능 모델 확인
best_model_result = max(experiment_results, key=lambda x: x['best_val_acc'])
print(f"\n✅ 최고 성능 모델: {best_model_result['model']} "
      f"(검증 정확도: {best_model_result['best_val_acc']:.2f}%)")
```

### 학습 곡선 시각화

```python
# ============================================
# 학습 곡선 시각화
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 손실 곡선
axes[0, 0].plot(history_lr['train_loss'], label='Logistic Regression (Train)', linestyle='--')
axes[0, 0].plot(history_lr['val_loss'], label='Logistic Regression (Val)')
axes[0, 0].plot(history_mlp['train_loss'], label='MLP (Train)', linestyle='--')
axes[0, 0].plot(history_mlp['val_loss'], label='MLP (Val)')
axes[0, 0].plot(history_cnn['train_loss'], label='CNN (Train)', linestyle='--')
axes[0, 0].plot(history_cnn['val_loss'], label='CNN (Val)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('학습/검증 손실 비교')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 정확도 곡선
axes[0, 1].plot(history_lr['train_acc'], label='Logistic Regression (Train)', linestyle='--')
axes[0, 1].plot(history_lr['val_acc'], label='Logistic Regression (Val)')
axes[0, 1].plot(history_mlp['train_acc'], label='MLP (Train)', linestyle='--')
axes[0, 1].plot(history_mlp['val_acc'], label='MLP (Val)')
axes[0, 1].plot(history_cnn['train_acc'], label='CNN (Train)', linestyle='--')
axes[0, 1].plot(history_cnn['val_acc'], label='CNN (Val)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('학습/검증 정확도 비교')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 검증 손실만 비교
axes[1, 0].plot(history_lr['val_loss'], label='Logistic Regression', marker='o')
axes[1, 0].plot(history_mlp['val_loss'], label='MLP', marker='s')
axes[1, 0].plot(history_cnn['val_loss'], label='CNN', marker='^')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Validation Loss')
axes[1, 0].set_title('검증 손실 비교 (낮을수록 좋음)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 검증 정확도만 비교
axes[1, 1].plot(history_lr['val_acc'], label='Logistic Regression', marker='o')
axes[1, 1].plot(history_mlp['val_acc'], label='MLP', marker='s')
axes[1, 1].plot(history_cnn['val_acc'], label='CNN', marker='^')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Validation Accuracy (%)')
axes[1, 1].set_title('검증 정확도 비교 (높을수록 좋음)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
plt.show()
```

---

## Step 5: 최종 평가 및 오류 분석

### 이번 단계 목표
- 최고 성능 모델로 테스트셋 평가 (1회만)
- 혼동행렬 및 클래스별 정확도 계산
- 오류 패턴 분석 (혼동 페어, 대표 샘플)

### 입력
- 최고 성능 모델 (검증 기준)
- 테스트 데이터 로더

### 출력
- 테스트 정확도
- 혼동행렬
- 클래스별 정확도 리포트
- 오류 샘플 시각화

### 평가 기준
- 테스트셋은 최종 1회만 사용
- 혼동행렬과 클래스별 지표 모두 계산

### 최고 모델 선택 및 테스트 평가

```python
# ============================================
# 최고 성능 모델 선택 (검증 기준)
# ============================================

best_model_name = best_model_result['model']
print(f"\n최고 성능 모델 선택: {best_model_name}")

# 모델 재생성 및 최고 상태 로드
if best_model_name == 'Logistic Regression':
    final_model = LogisticRegression(num_classes=10).to(device)
    final_model.load_state_dict(best_model_result['best_state'])
elif best_model_name == 'MLP':
    final_model = SimpleMLP(num_classes=10).to(device)
    final_model.load_state_dict(best_model_result['best_state'])
else:  # CNN
    final_model = SimpleCNN(num_classes=10).to(device)
    final_model.load_state_dict(best_model_result['best_state'])

print("✅ 최고 모델 상태 로드 완료")
```

### 테스트셋 평가 (최종 1회)

```python
# ============================================
# 테스트셋 평가 (최종 1회만 사용)
# ============================================

criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_preds, test_labels = validate(
    final_model, test_loader, criterion, device
)

print(f"\n{'='*60}")
print("최종 테스트셋 평가 결과")
print(f"{'='*60}")
print(f"테스트 손실: {test_loss:.4f}")
print(f"테스트 정확도: {test_acc:.2f}%")
print(f"{'='*60}\n")
```

### 혼동행렬 및 클래스별 지표

```python
# ============================================
# 혼동행렬 및 클래스별 지표
# ============================================

# 혼동행렬 계산
cm = confusion_matrix(test_labels, test_preds)

# 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('예측 클래스')
plt.ylabel('실제 클래스')
plt.title(f'혼동행렬 (테스트셋, 전체 정확도: {test_acc:.2f}%)')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.show()

# 클래스별 정확도 계산
class_accuracies = []
for i in range(10):
    class_mask = np.array(test_labels) == i
    if class_mask.sum() > 0:
        class_acc = (np.array(test_preds)[class_mask] == i).mean() * 100
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0.0)

# 클래스별 정확도 시각화
plt.figure(figsize=(10, 6))
bars = plt.bar(range(10), class_accuracies, color='steelblue', alpha=0.7)
plt.xlabel('클래스 (숫자)')
plt.ylabel('정확도 (%)')
plt.title('클래스별 테스트 정확도')
plt.xticks(range(10))
plt.ylim([0, 100])
plt.grid(True, alpha=0.3, axis='y')

# 값 표시
for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/class_accuracy.png', dpi=150)
plt.show()

# 클래스별 리포트
print("\n클래스별 성능 리포트:")
print("-" * 60)
for i in range(10):
    print(f"클래스 {i}: {class_accuracies[i]:.2f}%")
print("-" * 60)
```

### 혼동 페어 분석

```python
# ============================================
# 혼동 페어 분석 (Top-k)
# ============================================

# 혼동 행렬에서 가장 많이 혼동되는 페어 찾기
confusion_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((i, j, cm[i, j]))

# 혼동 횟수 기준 정렬
confusion_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n가장 많이 혼동되는 클래스 페어 (Top-5):")
print("-" * 60)
for idx, (true_class, pred_class, count) in enumerate(confusion_pairs[:5], 1):
    print(f"{idx}. 실제 {true_class} → 예측 {pred_class}: {count}회")
print("-" * 60)
```

### 오류 샘플 시각화

```python
# ============================================
# 오류 샘플 시각화
# ============================================

final_model.eval()
wrong_samples = []
wrong_labels_true = []
wrong_labels_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = final_model(images)
        _, predicted = torch.max(outputs, 1)
        
        wrong_mask = (predicted != labels)
        if wrong_mask.sum() > 0:
            wrong_samples.append(images[wrong_mask].cpu())
            wrong_labels_true.append(labels[wrong_mask].cpu())
            wrong_labels_pred.append(predicted[wrong_mask].cpu())
        
        if len(wrong_samples) >= 20:  # 최대 20개
            break

if wrong_samples:
    wrong_images = torch.cat(wrong_samples, dim=0)[:20]
    wrong_true = torch.cat(wrong_labels_true, dim=0)[:20]
    wrong_pred = torch.cat(wrong_labels_pred, dim=0)[:20]
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for idx in range(min(20, len(wrong_images))):
        ax = axes[idx // 5, idx % 5]
        ax.imshow(wrong_images[idx].squeeze().numpy(), cmap='gray')
        ax.set_title(f'True: {wrong_true[idx].item()}, Pred: {wrong_pred[idx].item()}', 
                     color='red' if wrong_true[idx] != wrong_pred[idx] else 'green')
        ax.axis('off')
    
    plt.suptitle('오류 샘플 (실제 라벨 vs 예측 라벨)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/error_samples.png', dpi=150)
    plt.show()
else:
    print("✅ 모든 샘플을 올바르게 예측했습니다!")
```

---

## Step 6: 모델 저장 및 로드 (표준화)

### 이번 단계 목표
- 체크포인트 형식 표준화
- 저장/로드 호환성 규약 명시
- 재현 가능한 모델 저장

### 입력
- 학습된 모델, 옵티마이저, 히스토리, 메타데이터

### 출력
- 표준화된 체크포인트 파일

### 평가 기준
- 체크포인트에 모든 필수 정보 포함
- 다른 세션에서 로드 가능

### 체크포인트 저장 (표준 형식)

```python
# ============================================
# 체크포인트 저장 (표준 형식)
# ============================================

def save_checkpoint(model, optimizer, history, best_val_acc, test_acc,
                    model_name, epoch, filepath):
    """
    표준화된 체크포인트 저장
    
    저장 내용:
    - model_state_dict: 모델 가중치
    - optimizer_state_dict: 옵티마이저 상태
    - history: 학습/검증 히스토리
    - metadata: 모델 정보, 성능, 환경 정보
    """
    checkpoint = {
        # 모델 상태
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'num_classes': 10,
        
        # 옵티마이저 상태
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_type': 'Adam',
        
        # 학습 정보
        'epoch': epoch,
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        
        # 메타데이터 (재현성)
        'metadata': {
            'pytorch_version': torch.__version__,
            'device': str(device),
            'seed': SEED,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_epochs': NUM_EPOCHS,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
        }
    }
    
    torch.save(checkpoint, filepath)
    print(f"✅ 체크포인트 저장 완료: {filepath}")

# 최고 모델 저장
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

save_checkpoint(
    final_model, optimizer, 
    best_model_result['history'],
    best_model_result['best_val_acc'],
    test_acc,
    best_model_name,
    NUM_EPOCHS,
    f'checkpoints/best_model_{best_model_name.lower().replace(" ", "_")}.pth'
)
```

### 체크포인트 로드 (호환성 검증)

```python
# ============================================
# 체크포인트 로드 (호환성 검증)
# ============================================

def load_checkpoint(filepath, device):
    """
    체크포인트 로드 및 호환성 검증
    
    입력:
        filepath: 체크포인트 파일 경로
        device: 디바이스
    
    출력:
        checkpoint: 로드된 체크포인트 딕셔너리
        model: 로드된 모델
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # 메타데이터 확인
    print("="*60)
    print("체크포인트 메타데이터:")
    print("="*60)
    for key, value in checkpoint['metadata'].items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # 모델 재생성
    model_name = checkpoint['model_name']
    if model_name == 'Logistic Regression':
        model = LogisticRegression(num_classes=checkpoint['num_classes']).to(device)
    elif model_name == 'MLP':
        model = SimpleMLP(num_classes=checkpoint['num_classes']).to(device)
    elif model_name == 'CNN':
        model = SimpleCNN(num_classes=checkpoint['num_classes']).to(device)
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✅ 모델 로드 완료: {model_name}")
    print(f"   검증 정확도: {checkpoint['best_val_acc']:.2f}%")
    print(f"   테스트 정확도: {checkpoint['test_acc']:.2f}%")
    
    return checkpoint, model

# 로드 테스트
loaded_checkpoint, loaded_model = load_checkpoint(
    f'checkpoints/best_model_{best_model_name.lower().replace(" ", "_")}.pth',
    device
)

# 로드된 모델로 재평가 (검증)
val_loss_loaded, val_acc_loaded, _, _ = validate(
    loaded_model, val_loader, criterion, device
)
print(f"\n로드된 모델 검증 정확도: {val_acc_loaded:.2f}% "
      f"(원본: {best_model_result['best_val_acc']:.2f}%)")
```

---

## Step 7: 실험 기록표 최종 정리

### 이번 단계 목표
- 모든 실험 결과를 표로 정리
- 고정 변수와 조작 변수 명시
- 관찰 사항 기록

### 입력
- 실험 결과 딕셔너리

### 출력
- 완전한 실험 기록표

### 평가 기준
- 모든 실험 정보 포함
- 재현 가능한 형태

### 실험 기록표 (최종)

```python
# ============================================
# 실험 기록표 (최종)
# ============================================

print("\n" + "="*100)
print("실험 기록표 (최종)")
print("="*100)

# 고정 변수
print("\n【고정 변수】")
print(f"  - 데이터셋: MNIST")
print(f"  - 전처리: ToTensor + Normalize(mean={MNIST_MEAN}, std={MNIST_STD})")
print(f"  - 데이터 분할: Train 50,000 / Val 10,000 / Test 10,000")
print(f"  - 배치 크기: {BATCH_SIZE}")
print(f"  - 에폭 수: {NUM_EPOCHS}")
print(f"  - 학습률: {LEARNING_RATE}")
print(f"  - Weight Decay: {WEIGHT_DECAY}")
print(f"  - 시드: {SEED}")
print(f"  - 디바이스: {device}")

# 조작 변수
print("\n【조작 변수】")
print(f"  - 모델 아키텍처: Logistic Regression, MLP, CNN")

# 실험 결과
print("\n【실험 결과】")
print(f"{'모델':<25} {'파라미터 수':<15} {'최고 검증 정확도':<20} {'테스트 정확도':<20}")
print("-"*100)

for result in experiment_results:
    # 테스트 정확도는 최고 모델만 계산
    if result['model'] == best_model_name:
        test_acc_str = f"{test_acc:.2f}%"
    else:
        test_acc_str = "N/A (미평가)"
    
    print(f"{result['model']:<25} "
          f"{result['num_params']:>13,}개  "
          f"{result['best_val_acc']:>18.2f}%  "
          f"{test_acc_str:>18}")

print("="*100)

# 관찰 사항
print("\n【관찰 사항】")
print(f"  1. 모델 복잡도가 증가할수록 검증 정확도 향상")
print(f"  2. CNN이 가장 높은 성능 달성 (공간적 패턴 학습 효과)")
print(f"  3. 모든 모델에서 학습/검증 곡선이 수렴 (과적합 없음)")
print(f"  4. 최고 성능 모델: {best_model_name} (검증: {best_model_result['best_val_acc']:.2f}%, "
      f"테스트: {test_acc:.2f}%)")

print("\n" + "="*100)
```

---

## 추가 도전 과제

### 과제 1: 모델 개선 (평가 가능)

**목표**: CNN 모델을 개선하여 검증 정확도 98% 이상 달성

**산출물**:
- 개선된 모델 코드
- 실험 기록표 (기존 CNN vs 개선 CNN 비교)
- 검증 정확도 98% 이상 스크린샷

**제한 조건**:
- 데이터 증강 사용 가능
- 모델 파라미터 수는 500,000개 이하
- 학습 에폭은 20 이하

**채점 기준**:
- 검증 정확도 ≥ 98%: 만점
- 검증 정확도 97-98%: 80점
- 검증 정확도 96-97%: 60점
- 그 외: 40점

### 과제 2: 하이퍼파라미터 튜닝 (실험 기록)

**목표**: 학습률을 0.0001, 0.001, 0.01로 변경하며 학습 곡선 비교

**산출물**:
- 실험 기록표 (고정: 모델/데이터, 조작: 학습률)
- 학습 곡선 비교 그래프
- 최적 학습률 선택 근거 (1문단)

**제한 조건**:
- 동일 모델(MLP) 사용
- 동일 시드 사용
- 각 실험 10 에폭

**채점 기준**:
- 실험 기록표 완성: 50점
- 그래프 및 근거 제시: 50점

### 과제 3: 데이터 증강 효과 분석

**목표**: 데이터 증강을 추가하여 성능 변화 관찰

**산출물**:
- 증강 전/후 성능 비교표
- 증강된 이미지 샘플 시각화
- 증강 효과 분석 (1문단)

**제한 조건**:
- RandomRotation, RandomAffine 등 사용
- 동일 모델(CNN) 사용

### 과제 4: 의도된 실패 실험 (안티패턴 학습)

**목표**: 검증 없이 학습하여 과적합 관찰

**산출물**:
- 검증 없이 학습한 모델의 학습 곡선
- 검증 포함 모델과 비교 그래프
- 과적합 증상 설명 (1문단)

**실험 설계**:
- 모델: MLP
- 에폭: 20
- 검증 없이 학습 → 학습 손실만 기록
- 이후 검증셋으로 평가하여 과적합 확인

**채점 기준**:
- 과적합 관찰 및 설명: 만점
- 그래프만 제시: 50점

### 과제 5: Fashion-MNIST 전환

**목표**: MNIST 코드를 Fashion-MNIST로 전환

**산출물**:
- Fashion-MNIST 학습 코드
- MNIST vs Fashion-MNIST 성능 비교표
- 난이도 차이 분석 (1문단)

---

## 핵심 요약

### 구현한 내용

✅ **환경 설정**: 재현 가능한 실험 환경 (시드, 디바이스, 버전)  
✅ **데이터 준비**: train/val/test 분할, 전처리 파이프라인  
✅ **모델 정의**: Logistic Regression, MLP, CNN 점진적 구현  
✅ **학습 파이프라인**: 훈련/검증 루프 분리, 최고 성능 체크포인트 저장  
✅ **평가**: 테스트셋 평가, 혼동행렬, 클래스별 정확도  
✅ **오류 분석**: 혼동 페어, 오류 샘플 시각화  
✅ **모델 저장/로드**: 표준화된 체크포인트 형식  
✅ **실험 기록**: 고정/조작 변수 명시, 결과 비교표

### 배운 점

1. **재현성의 중요성**: 시드 고정으로 동일 결과 보장
2. **데이터 분리**: train/val/test 역할 구분 및 검증 기준 모델 선택
3. **점진적 개선**: 간단한 모델부터 시작하여 복잡도 증가
4. **수치 안정성**: Gradient clipping, NaN 체크, 학습률 스케줄러
5. **체계적 평가**: 정확도 외 혼동행렬, 클래스별 지표로 진단

### 다음 단계

이제 다양한 방향으로 확장할 수 있습니다:
- 더 복잡한 모델 (ResNet, Transformer)
- 다른 데이터셋 (CIFAR-10, ImageNet)
- 다른 태스크 (물체 탐지, 세그멘테이션)
- 하이퍼파라미터 자동 튜닝 (Optuna, Ray Tune)

