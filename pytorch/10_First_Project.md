# 10. 첫 번째 프로젝트: MNIST 손글씨 인식

이번 단원에서는 지금까지 배운 모든 내용을 종합하여 **완전한 머신러닝 프로젝트**를 구현해보겠습니다.

## 학습 목표
- 전체 머신러닝 파이프라인 구현
- 데이터 로드 및 전처리
- 모델 정의 및 학습
- 성능 평가
- 실전 팁과 디버깅

## 프로젝트 개요

**MNIST**: 손으로 쓴 숫자 이미지 분류 데이터셋
- 70,000개의 28x28 그레이스케일 이미지
- 0~9 숫자 분류 (10 클래스)
- 딥러닝의 "Hello World"

이 프로젝트에서 우리는 이미지 분류를 위한 완전한 파이프라인을 구축합니다.

## 왜 이 단원이 필요한가?

지금까지 이론을 배웠습니다. 이제 **실전**을 경험해야 합니다.

- **전체 파이프라인**: 데이터 → 모델 → 학습 → 평가의 완전한 흐름
- **점진적 접근**: 가장 간단한 모델부터 시작해서 점점 개선
- **이해의 깊이**: 왜 이 방식이 더 좋은지 각 단계에서 확인
- **실전 문제**: 실제로 작동하는 코드 작성
- **디버깅 경험**: 에러 해결하며 배우기
- **자신감**: 완전한 프로젝트를 끝냈다는 성취감

이 단원을 마치면 자신만의 프로젝트를 시작할 수 있습니다!

## 이 단원을 배우기 전에

**이전 단원 (09) 복습: 데이터 라벨링과 Dataset 만들기를 이해하셨나요?**

## 이 단원 다음에는

**축하합니다! 이제 자신만의 프로젝트를 시작할 수 있습니다.**

## Step 1: 데이터 로드 및 탐색

먼저 데이터가 어떻게 생겼는지 확인해봅시다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image 또는 numpy 배열을 텐서로 변환
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균과 표준편차
])

# 데이터셋 다운로드 및 로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"학습 데이터: {len(train_dataset)}개")
print(f"테스트 데이터: {len(test_dataset)}개")
print(f"클래스 수: {len(train_dataset.classes)}")
```

### 데이터 시각화

데이터가 어떻게 생겼는지 확인해봅시다.

```python
# 몇 개 샘플 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for idx in range(10):
    image, label = train_dataset[idx]
    axes[idx // 5, idx % 5].imshow(image.squeeze(), cmap='gray')
    axes[idx // 5, idx % 5].set_title(f'Label: {label}')
    axes[idx // 5, idx % 5].axis('off')

plt.tight_layout()
plt.show()
```

## Step 2: 모델 정의 - 점진적 접근

**중요**: 가장 간단한 모델부터 시작해서 점점 개선하겠습니다!

### 전략

1. **Logistic Regression**: 단일 레이어 (베이스라인)
2. **MLP**: 여러 레이어 (표현력 향상)
3. **CNN**: 컨볼루션 (이미지 특화)

각 단계에서 성능을 비교하며 "왜 더 좋은가?"를 이해합니다.

### 2-1. Logistic Regression (베이스라인)

**가장 간단한 모델**: 입력을 평탄화하고 바로 출력으로 연결

**의미**: 28×28 = 784 픽셀을 각 클래스의 가중치와 내적 → 각 클래스 점수

```python
# Logistic Regression 모델
class LogisticRegression(nn.Module):
    def __init__(self, num_classes=10):
        super(LogisticRegression, self).__init__()
        # 28x28 = 784 픽셀을 10개 클래스로
        self.linear = nn.Linear(28 * 28, num_classes)
    
    def forward(self, x):
        # 이미지를 평탄화 (flatten)
        x = x.view(-1, 28 * 28)
        # 직접 10개 클래스 점수로 변환
        return self.linear(x)

model_lr = LogisticRegression(num_classes=10)
print("=== Logistic Regression (베이스라인) ===")
print(model_lr)
print(f"\n파라미터 수: {sum(p.numel() for p in model_lr.parameters())}")
print("\n의미: 784 픽셀 각각에 가중치를 두고, 클래스 점수를 직접 계산")
```

### 2-2. MLP (Multi-Layer Perceptron)

**개선**: 여러 레이어로 표현력 향상

**의미**: 픽셀 → 은닉층 → 출력. 은닉층이 중간 표현을 학습

```python
# MLP 모델
class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_mlp = SimpleMLP(num_classes=10)
print("=== MLP (다층 신경망) ===")
print(model_mlp)
print(f"\n파라미터 수: {sum(p.numel() for p in model_mlp.parameters())}")
print("\n의미: 은닉층이 중간 패턴을 학습 → 표현력 향상")
```

## Step 3: 학습 준비

손실 함수와 옵티마이저를 설정합니다.

```python
# 손실 함수: Cross Entropy (다중 분류에 적합)
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam (adaptive learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("손실 함수: CrossEntropyLoss")
print("옵티마이저: Adam (lr=0.001)")
```

## Step 4: 학습 루프

이제 모델을 학습시켜봅시다.

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 epoch 학습"""
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

print("학습 함수 정의 완료")
```

### 실제 학습 실행

여러 epoch 동안 학습합니다.

```python
# 학습 파라미터
num_epochs = 5  # 학습할 epoch 수

# 학습 히스토리 저장
train_losses = []
train_accs = []

print(f"학습 시작: {num_epochs} epochs")
print("-" * 50)

for epoch in range(num_epochs):
    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(loss)
    train_accs.append(acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.4f}, Acc: {acc:.2f}%")

print("-" * 50)
print("학습 완료!")
```

## Step 5: 평가

테스트 데이터로 모델을 평가합니다.

```python
def evaluate(model, test_loader, device):
    """모델 평가"""
    model.eval()  # 평가 모드
    correct = 0
    total = 0
    
    with torch.no_grad():  # Gradient 계산 비활성화
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# 테스트 정확도 계산
test_accuracy = evaluate(model, test_loader, device)
print(f"테스트 정확도: {test_accuracy:.2f}%")
```

### 결과 시각화

학습 과정과 예측 결과를 시각화합니다.

```python
# 학습 곡선
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(train_accs)
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()
```

## Step 6: 오류 분석

어떤 숫자를 잘못 예측했는지 확인합니다.

```python
# 잘못 예측한 샘플 찾기
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
count = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # 잘못 예측한 것만
        wrong = (predicted != labels)
        wrong_images = images[wrong]
        wrong_labels = labels[wrong]
        wrong_preds = predicted[wrong]
        
        for i in range(min(10, len(wrong_images))):
            if count >= 10:
                break
            ax = axes[count // 5, count % 5]
            ax.imshow(wrong_images[i].cpu().squeeze(), cmap='gray')
            ax.set_title(f'True: {wrong_labels[i].item()}, Pred: {wrong_preds[i].item()}')
            ax.axis('off')
            count += 1
        
        if count >= 10:
            break

plt.tight_layout()
plt.show()
```

## Step 7: 모델 저장 및 로드

학습한 모델을 저장하고 나중에 다시 불러와 사용할 수 있습니다.

```python
# 모델 저장
# 방법 1: 전체 모델 저장 (권장하지 않음)
# torch.save(model, 'mnist_model_full.pth')

# 방법 2: 상태 사전 저장 (가중치만, 권장)
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_losses[-1],
    'accuracy': train_accs[-1],
}, 'mnist_model_checkpoint.pth')

print("모델 저장 완료!")
```

```python
# 모델 로드
# 새로운 모델 인스턴스 생성
loaded_model = SimpleCNN(num_classes=10).to(device)

# 체크포인트 로드
checkpoint = torch.load('mnist_model_checkpoint.pth')
loaded_model.load_state_dict(checkpoint['model_state_dict'])

print(f"모델 로드 완료!")
print(f"저장된 정확도: {checkpoint['accuracy']:.2f}%")

# 로드된 모델로 테스트
loaded_model.eval()
test_accuracy_loaded = evaluate(loaded_model, test_loader, device)
print(f"로드된 모델 테스트 정확도: {test_accuracy_loaded:.2f}%")
```

## 실전 팁

### 디버깅 팁

1. **Shapes 확인**: 각 레이어의 입출력 크기를 print로 확인
   ```python
   print(x.shape)  # 각 레이어마다 추가
   ```

2. **Loss가 NaN인 경우**:
   - Learning rate가 너무 크거나
   - 입력에 이상치가 있거나
   - 수치 불안정

3. **학습이 안 되는 경우**:
   - Learning rate 조정 (0.0001, 0.001, 0.01 등 시도)
   - 모델이 너무 단순/복잡
   - 데이터 전처리 문제

### 성능 개선 팁

1. **Data Augmentation**: 더 많은 변환 추가
2. **Regularization**: Dropout 비율 조정, Weight decay
3. **모델 크기**: 레이어 수, 채널 수 증가
4. **학습률 스케줄링**: StepLR, CosineAnnealingLR 등

## 추가 도전 문제

### 문제 1: 모델 개선
현재 CNN 모델을 개선하여 정확도를 99% 이상으로 올려보세요.

### 문제 2: 하이퍼파라미터 튜닝
learning rate를 0.0001, 0.001, 0.01로 바꿔가며 학습 곡선을 비교하세요.

### 문제 3: 데이터 증강
더 많은 transforms를 추가하여 성능 변화를 관찰하세요.

### 문제 4: Fashion-MNIST 전환
MNIST 대신 Fashion-MNIST로 같은 코드를 실행해보세요.

### 문제 5: 학습률 스케줄링
StepLR이나 CosineAnnealingLR을 적용해보고 효과를 확인하세요.

## 추가 도전 문제

### 문제 1: 모델 개선
현재 CNN 모델을 개선하여 정확도를 99% 이상으로 올려보세요.

### 문제 2: 하이퍼파라미터 튜닝
learning rate를 0.0001, 0.001, 0.01로 바꿔가며 학습 곡선을 비교하세요.

### 문제 3: 데이터 증강
더 많은 transforms를 추가하여 성능 변화를 관찰하세요.

### 문제 4: Fashion-MNIST 전환
MNIST 대신 Fashion-MNIST로 같은 코드를 실행해보세요.

### 문제 5: 학습률 스케줄링
StepLR이나 CosineAnnealingLR을 적용해보고 효과를 확인하세요.

## 핵심 요약

축하합니다! 여러분의 첫 번째 머신러닝 프로젝트를 완성했습니다.

### 구현한 내용

✅ **데이터 로드**: MNIST 데이터셋 다운로드 및 전처리  
✅ **모델 정의**: CNN 아키텍처 설계  
✅ **학습**: 손실 함수, 옵티마이저, 학습 루프  
✅ **평가**: 테스트 정확도 측정  
✅ **분석**: 오류 패턴 확인

### 배운 점

1. **전체 파이프라인**: 데이터 → 모델 → 학습 → 평가의 완전한 흐름
2. **실전 코드**: 단순 예제가 아닌 실제 동작하는 코드
3. **디버깅**: 문제를 찾고 해결하는 방법

### 다음 단계

이제 다양한 데이터셋과 모델로 프로젝트를 확장할 수 있습니다:
- CIFAR-10, Fashion-MNIST
- ResNet, VGG 등 더 복잡한 모델
- 이미지 분류 외의 다른 태스크 (물체 탐지, 세그멘테이션 등)

**좋은 성과를 거두셨습니다! 🎉**
