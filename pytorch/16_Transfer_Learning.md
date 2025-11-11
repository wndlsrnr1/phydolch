# 16. 전이 학습 (Transfer Learning)

이번 단원에서는 **사전 학습된 모델을 활용하여 새로운 작업에 적용하는 전이 학습**을 배워보겠습니다.

## 학습 목표
- 전이 학습의 개념과 원리
- Pre-trained 모델 활용법
- Fine-tuning 전략
- Feature Extraction vs Fine-tuning
- 실전 이미지 분류 프로젝트

## 왜 이 단원이 필요한가?

**처음부터 학습하는 것은 비효율적**입니다:

- **데이터 부족**: 수백만 장의 이미지 확보 어려움
- **계산 비용**: 학습에 수 주 소요
- **전문 지식**: 최적의 아키텍처 설계 어려움

**전이 학습**은 이 문제를 해결합니다:
- ImageNet으로 학습된 모델 활용
- 적은 데이터로 높은 성능
- 빠른 학습 (수 분 ~ 수 시간)

백엔드 개발자 관점: **라이브러리 재사용**과 유사합니다!

---
# 1. 전이 학습의 원리
---

## 1.1 왜 전이 학습이 작동하는가?

### CNN의 계층적 특징 학습

```
초기 레이어 (Low-level features)
  ↓
  엣지, 코너, 색상 패턴
  → 모든 이미지에 공통적
  
중간 레이어 (Mid-level features)
  ↓
  텍스처, 패턴, 부분적 형태
  → 많은 작업에 유용
  
후기 레이어 (High-level features)
  ↓
  객체 부분, 전체 형태
  → 작업 특화적
```

### 전이 학습 전략

**1. Feature Extraction (특징 추출)**
- Pre-trained 모델을 **고정** (freeze)
- 마지막 분류기만 교체 및 학습
- 데이터가 **매우 적을 때**

**2. Fine-tuning (미세 조정)**
- Pre-trained 모델의 일부 레이어 **해동** (unfreeze)
- 전체 또는 일부를 재학습
- 데이터가 **충분할 때**

## 1.2 언제 어떤 전략을 사용할까?

| 데이터 크기 | 유사도 | 전략 |
|------------|--------|------|
| 적음 | 높음 | Feature Extraction |
| 적음 | 낮음 | Feature Extraction + 초기 레이어 fine-tune |
| 많음 | 높음 | Fine-tuning (후기 레이어) |
| 많음 | 낮음 | Fine-tuning (전체) |

---
# 2. Pre-trained 모델 로드
---

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# 1. Pre-trained 모델 로드
# ImageNet으로 학습된 ResNet18
model = models.resnet18(pretrained=True)

print("=== ResNet18 구조 ===")
print(model)

# 모델 정보
total_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터: {total_params:,}")

# 2. 다양한 Pre-trained 모델들
print("\n=== 사용 가능한 Pre-trained 모델 ===")

available_models = {
    'ResNet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
    'VGG': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
    'MobileNet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
    'EfficientNet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b7'],
    'Vision Transformer': ['vit_b_16', 'vit_b_32']
}

for family, models_list in available_models.items():
    print(f"\n{family}:")
    for m in models_list[:3]:  # 처음 3개만
        print(f"  - {m}")

# 3. 모델 선택 가이드
print("\n=== 모델 선택 가이드 ===")
print("ResNet18/34: 빠르고 가벼움, 일반적 용도")
print("ResNet50+: 높은 정확도, 더 많은 리소스")
print("MobileNet: 모바일/엣지 디바이스용")
print("EfficientNet: 최고 효율성")
print("ViT: 최신 Transformer 기반, 큰 데이터셋")
```

---
# 3. Feature Extraction
---

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Pre-trained 모델 로드
model = models.resnet18(pretrained=True)

# === 방법 1: 모든 파라미터 고정 ===
for param in model.parameters():
    param.requires_grad = False

# 마지막 분류기만 교체
num_classes = 10  # 새로운 작업의 클래스 수
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

print("=== Feature Extraction 모델 ===")
print(f"입력 특징: {num_features}")
print(f"출력 클래스: {num_classes}")

# 학습 가능한 파라미터 확인
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\n총 파라미터: {total_params:,}")
print(f"학습 가능 파라미터: {trainable_params:,}")
print(f"고정된 파라미터: {total_params - trainable_params:,}")
print(f"학습 비율: {trainable_params / total_params * 100:.2f}%")

# === 방법 2: 커스텀 분류기 ===
class CustomClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

model2 = models.resnet18(pretrained=True)
for param in model2.parameters():
    param.requires_grad = False

model2.fc = CustomClassifier(num_features, num_classes)

print("\n=== 커스텀 분류기 ===")
print(model2.fc)

# Optimizer 설정 (학습 가능한 파라미터만)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

print("\n✓ Feature Extraction 모델 준비 완료")
print("장점: 빠른 학습, 적은 메모리, 과적합 방지")
print("단점: 제한된 성능 향상")
```

---
# 4. Fine-tuning
---

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Pre-trained 모델 로드
model = models.resnet18(pretrained=True)

# === 전략 1: 후기 레이어만 fine-tune ===
# 초기 레이어 고정
for name, param in model.named_parameters():
    # layer4와 fc만 학습
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

# 분류기 교체
num_features = model.fc.in_features
num_classes = 10
model.fc = nn.Linear(num_features, num_classes)

print("=== Fine-tuning 전략 1: 후기 레이어 ===")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"학습 가능: {trainable:,} ({trainable/total*100:.1f}%)")

# === 전략 2: 전체 fine-tune (다른 learning rate) ===
model2 = models.resnet18(pretrained=True)
model2.fc = nn.Linear(num_features, num_classes)

# Discriminative Learning Rate
# 초기 레이어: 작은 LR, 후기 레이어: 큰 LR
optimizer = torch.optim.Adam([
    {'params': model2.layer1.parameters(), 'lr': 1e-5},
    {'params': model2.layer2.parameters(), 'lr': 1e-5},
    {'params': model2.layer3.parameters(), 'lr': 1e-4},
    {'params': model2.layer4.parameters(), 'lr': 1e-4},
    {'params': model2.fc.parameters(), 'lr': 1e-3}
])

print("\n=== Fine-tuning 전략 2: Discriminative LR ===")
for i, param_group in enumerate(optimizer.param_groups):
    print(f"그룹 {i}: LR = {param_group['lr']}")

# === 전략 3: 점진적 해동 (Gradual Unfreezing) ===
def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    뒤에서부터 점진적으로 레이어 해동
    """
    # 모든 레이어 고정
    for param in model.parameters():
        param.requires_grad = False
    
    # 뒤에서부터 해동
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for i in range(min(num_layers_to_unfreeze, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = True
    
    # 분류기는 항상 학습
    for param in model.fc.parameters():
        param.requires_grad = True

model3 = models.resnet18(pretrained=True)
model3.fc = nn.Linear(num_features, num_classes)

print("\n=== Fine-tuning 전략 3: 점진적 해동 ===")
for stage in range(1, 5):
    unfreeze_layers(model3, stage)
    trainable = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    print(f"Stage {stage}: {trainable:,} 파라미터 학습 가능")

print("\n✓ Fine-tuning 전략 준비 완료")
print("\n권장 순서:")
print("1. Feature Extraction으로 시작")
print("2. 성능 plateau 시 후기 레이어 fine-tune")
print("3. 필요 시 전체 fine-tune")
```

---
# 5. 실전 프로젝트: 개/고양이 분류
---

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# 1. 데이터 전처리
# ImageNet 통계 사용 (Pre-trained 모델과 일치)
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 평균
        std=[0.229, 0.224, 0.225]     # ImageNet 표준편차
    )
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("=== 데이터 전처리 ===")
print("✓ 입력 크기: 224x224")
print("✓ ImageNet 정규화 적용")
print("✓ Data Augmentation: Flip, Crop")

# 2. 모델 구성
def create_transfer_model(num_classes, freeze_backbone=True):
    """
    전이 학습 모델 생성
    """
    model = models.resnet18(pretrained=True)
    
    # Backbone 고정
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # 분류기 교체
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model

model = create_transfer_model(num_classes=2, freeze_backbone=True)

print("\n=== 모델 구성 ===")
print("✓ Base: ResNet18 (ImageNet pre-trained)")
print("✓ Backbone: 고정")
print("✓ Classifier: 2-class (개/고양이)")

# 3. 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print(f"\n=== 학습 설정 ===")
print(f"Device: {device}")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Scheduler: StepLR (step=7, gamma=0.1)")
print(f"Loss: CrossEntropyLoss")
```

```python
# 4. 학습 루프
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    한 epoch 학습
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    검증
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# 전체 학습 함수
def train_transfer_learning(
    model, train_loader, val_loader, 
    criterion, optimizer, scheduler,
    device, num_epochs=25
):
    """
    전이 학습 전체 프로세스
    """
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 검증
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Best model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 출력
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n✓ 학습 완료! Best Val Acc: {best_acc:.4f}")
    return history

print("학습 함수 준비 완료")
print("\n사용법:")
print("history = train_transfer_learning(")
print("    model, train_loader, val_loader,")
print("    criterion, optimizer, scheduler,")
print("    device, num_epochs=25")
print(")")
```

---
# 6. 2단계 학습: Feature Extraction → Fine-tuning
---

```python
import torch
import torch.nn as nn
import torch.optim as optim

def two_stage_training(
    model, train_loader, val_loader, device,
    stage1_epochs=10, stage2_epochs=15
):
    """
    2단계 학습 전략
    Stage 1: Feature Extraction
    Stage 2: Fine-tuning
    """
    
    # === Stage 1: Feature Extraction ===
    print("=== Stage 1: Feature Extraction ===")
    print("Backbone 고정, Classifier만 학습\n")
    
    # Backbone 고정
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 학습
    history_stage1 = train_transfer_learning(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, num_epochs=stage1_epochs
    )
    
    # === Stage 2: Fine-tuning ===
    print("\n=== Stage 2: Fine-tuning ===")
    print("후기 레이어 해동, 전체 학습\n")
    
    # 후기 레이어 해동
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # 새로운 Optimizer (더 작은 LR)
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 학습
    history_stage2 = train_transfer_learning(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, num_epochs=stage2_epochs
    )
    
    # 히스토리 병합
    history = {
        'stage1': history_stage1,
        'stage2': history_stage2
    }
    
    return history

print("2단계 학습 함수 준비 완료")
print("\n장점:")
print("1. Stage 1에서 빠르게 분류기 학습")
print("2. Stage 2에서 세밀한 조정")
print("3. 안정적이고 높은 성능")
```

---
# 7. 추론 및 배포
---

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class ImageClassifier:
    """
    전이 학습 모델 추론 클래스
    """
    def __init__(self, model_path, class_names, device='cpu'):
        self.device = device
        self.class_names = class_names
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_path):
        """모델 로드"""
        import torchvision.models as models
        
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, len(self.class_names))
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        
        return model
    
    def predict(self, image_path):
        """
        이미지 분류
        """
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # 결과
        predicted_idx = probabilities.argmax().item()
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        # 모든 클래스 확률
        all_probs = {
            self.class_names[i]: probabilities[i].item()
            for i in range(len(self.class_names))
        }
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, image_paths):
        """
        배치 추론
        """
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(self.transform(image))
        
        batch = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
        
        results = []
        for i, probs in enumerate(probabilities):
            predicted_idx = probs.argmax().item()
            results.append({
                'image': image_paths[i],
                'class': self.class_names[predicted_idx],
                'confidence': probs[predicted_idx].item()
            })
        
        return results

# 사용 예시
print("=== ImageClassifier 사용법 ===")
print("\n# 초기화")
print("classifier = ImageClassifier(")
print("    model_path='best_model.pth',")
print("    class_names=['cat', 'dog'],")
print("    device='cuda'")
print(")")
print("\n# 단일 예측")
print("result = classifier.predict('image.jpg')")
print("print(result['class'], result['confidence'])")
print("\n# 배치 예측")
print("results = classifier.predict_batch(['img1.jpg', 'img2.jpg'])")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 전이 학습의 원리
- **계층적 특징**: 초기(일반) → 후기(특화)
- **재사용**: ImageNet 지식 활용
- **효율성**: 적은 데이터, 빠른 학습

### 2. 두 가지 전략
**Feature Extraction**:
- Backbone 고정
- 분류기만 학습
- 데이터 적을 때

**Fine-tuning**:
- 일부/전체 해동
- 재학습
- 데이터 충분할 때

### 3. 실전 기법
- **Discriminative LR**: 레이어별 다른 learning rate
- **점진적 해동**: 뒤에서부터 단계적 해동
- **2단계 학습**: Feature Extraction → Fine-tuning

### 4. 데이터 전처리
- **ImageNet 정규화**: Pre-trained 모델과 일치
- **입력 크기**: 224x224 (ResNet 기준)
- **Data Augmentation**: Flip, Crop, Rotation

### 5. 모델 선택
- **ResNet18/34**: 빠르고 가벼움
- **ResNet50+**: 높은 정확도
- **MobileNet**: 모바일/엣지
- **EfficientNet**: 최고 효율

### 실전 체크리스트
```python
1. Pre-trained 모델 선택
2. 데이터 전처리 (ImageNet 정규화)
3. Feature Extraction으로 시작
4. 성능 plateau 시 Fine-tuning
5. Discriminative LR 사용
6. Best model 저장
7. 추론 클래스 구현
```

### 성능 비교
| 방법 | 학습 시간 | 정확도 | 데이터 요구량 |
|------|----------|--------|---------------|
| From Scratch | 수 일 | 낮음 | 매우 많음 |
| Feature Extraction | 수 분 | 중간 | 적음 |
| Fine-tuning | 수 시간 | 높음 | 중간 |

다음 단원에서는 **시계열 예측**을 배워보겠습니다!
