# 14. 시각화 및 모니터링

이번 단원에서는 **학습 과정을 시각화하고 모니터링하는 방법**을 배워보겠습니다.

## 학습 목표
- TensorBoard 사용법
- 학습 곡선 분석
- 모델 성능 시각화
- 실시간 모니터링
- 백엔드 개발자를 위한 로깅 전략

## 왜 이 단원이 필요한가?

딥러닝 학습은 **블랙박스**처럼 보입니다:

- **긴 학습 시간**: 수 시간 ~ 수 일
- **복잡한 동작**: 내부에서 무슨 일이 일어나는지 모름
- **디버깅 어려움**: 문제를 빨리 발견해야 함
- **성능 평가**: 다양한 지표 추적 필요

백엔드 개발자라면 **APM, 로그 분석, 대시보드**에 익숙할 것입니다. 이 경험을 딥러닝 모니터링에 적용해봅시다.

---
# 1. TensorBoard 기초
---

## 1.1 TensorBoard란?

**TensorBoard**는 TensorFlow에서 시작했지만 PyTorch에서도 사용 가능한 시각화 도구입니다.

### 주요 기능
1. **Scalars**: Loss, Accuracy 등 스칼라 값 추적
2. **Images**: 입력 이미지, 예측 결과 시각화
3. **Graphs**: 모델 구조 시각화
4. **Histograms**: 가중치, Gradient 분포
5. **Embeddings**: 고차원 데이터 시각화

### 설치
```bash
pip install tensorboard
```

### 실행
```bash
tensorboard --logdir=runs
```

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# TensorBoard Writer 생성
writer = SummaryWriter('runs/experiment_1')

# 1. Scalar 로깅
for epoch in range(100):
    # 가상의 loss 값
    train_loss = 1.0 / (epoch + 1) + np.random.rand() * 0.1
    val_loss = 1.2 / (epoch + 1) + np.random.rand() * 0.1
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    
    # Accuracy
    accuracy = min(0.95, epoch / 100 + np.random.rand() * 0.05)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

# 2. 여러 값을 한 그래프에
writer.add_scalars('Loss/comparison', {
    'train': train_loss,
    'validation': val_loss
}, epoch)

# 3. 이미지 로깅
dummy_img = torch.rand(3, 64, 64)  # [C, H, W]
writer.add_image('sample_image', dummy_img, 0)

# 4. 여러 이미지를 그리드로
dummy_imgs = torch.rand(16, 3, 64, 64)  # [N, C, H, W]
from torchvision.utils import make_grid
img_grid = make_grid(dummy_imgs, nrow=4)
writer.add_image('image_grid', img_grid, 0)

# 5. Histogram (가중치 분포)
model = nn.Linear(10, 5)
for name, param in model.named_parameters():
    writer.add_histogram(name, param, 0)

# 6. 모델 그래프
dummy_input = torch.randn(1, 10)
writer.add_graph(model, dummy_input)

writer.close()

print("TensorBoard 로그 생성 완료!")
print("실행: tensorboard --logdir=runs")
print("브라우저: http://localhost:6006")
```

---
# 2. 학습 곡선 분석
---

## 2.1 학습 곡선이란?

**학습 곡선**: Epoch에 따른 Loss/Accuracy 변화

### 정상적인 학습
```
Loss
  ↑
  │ Train \\\\\\___________
  │ Val    \\\\\\\\________
  │
  └─────────────────────→ Epoch
```

### 오버피팅
```
Loss
  ↑
  │ Train \\\\\\___________
  │ Val    \\\\\\\\↗↗↗↗↗↗↗
  │
  └─────────────────────→ Epoch
```

### 언더피팅
```
Loss
  ↑
  │ Train \\\\\\\\\\\\\\___
  │ Val    \\\\\\\\\\\\\\___
  │        (높은 수준에서 수렴)
  └─────────────────────→ Epoch
```

```python
import matplotlib.pyplot as plt
import numpy as np

# 학습 곡선 시각화
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """
    학습 곡선 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 곡선
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 곡선
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
    axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 예제 데이터
epochs = np.arange(1, 51)

# 정상적인 학습
train_losses = 2.0 / (epochs + 1) + np.random.rand(50) * 0.05
val_losses = 2.2 / (epochs + 1) + np.random.rand(50) * 0.05
train_accs = 1 - train_losses / 2
val_accs = 1 - val_losses / 2

fig = plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("학습 곡선 분석:")
print(f"최종 Train Loss: {train_losses[-1]:.4f}")
print(f"최종 Val Loss: {val_losses[-1]:.4f}")
print(f"Gap: {val_losses[-1] - train_losses[-1]:.4f}")

if val_losses[-1] - train_losses[-1] > 0.2:
    print("⚠️  오버피팅 가능성 - 정규화 강화 고려")
elif train_losses[-1] > 0.5:
    print("⚠️  언더피팅 가능성 - 모델 복잡도 증가 고려")
else:
    print("✓ 정상적인 학습")
```

---
# 3. 실시간 모니터링
---

## 3.1 학습 중 모니터링

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

class TrainingMonitor:
    """
    학습 과정을 실시간으로 모니터링
    """
    def __init__(self, log_dir='runs/experiment'):
        self.writer = SummaryWriter(log_dir)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.best_val_loss = float('inf')
        self.start_time = time.time()
    
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Epoch 종료 시 로깅"""
        # 메트릭 저장
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        
        # TensorBoard 로깅
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        # Best model 체크
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improvement = "✓ Best!"
        else:
            improvement = ""
        
        # 콘솔 출력
        elapsed = time.time() - self.start_time
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s {improvement}")
    
    def log_batch(self, epoch, batch_idx, loss, total_batches):
        """배치 단위 로깅"""
        if batch_idx % 10 == 0:
            global_step = epoch * total_batches + batch_idx
            self.writer.add_scalar('Loss/batch', loss, global_step)
    
    def log_gradients(self, model, epoch):
        """Gradient 분포 로깅"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f'Gradients/{name}', param.grad, epoch
                )
                self.writer.add_histogram(
                    f'Weights/{name}', param.data, epoch
                )
    
    def log_learning_rate(self, optimizer, epoch):
        """Learning rate 로깅"""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'LearningRate/group_{i}', lr, epoch)
    
    def plot_final_curves(self):
        """최종 학습 곡선 시각화"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.metrics['train_loss'], label='Train')
        axes[0].plot(self.metrics['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.metrics['train_acc'], label='Train')
        axes[1].plot(self.metrics['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def close(self):
        """Writer 종료"""
        self.writer.close()

# 사용 예시
monitor = TrainingMonitor('runs/example')

# 가상의 학습 루프
for epoch in range(10):
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.1 / (epoch + 1)
    train_acc = min(0.95, epoch / 10)
    val_acc = min(0.90, epoch / 10)
    
    monitor.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc)

fig = monitor.plot_final_curves()
plt.show()
monitor.close()

print("\n모니터링 완료!")
```

---
# 4. 모델 성능 시각화
---

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Confusion Matrix 시각화
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Confusion Matrix 시각화
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()

# 예측 결과 시각화 (이미지 분류)
def visualize_predictions(images, labels, predictions, class_names, n=16):
    """
    예측 결과 시각화
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(n, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # 정규화 해제 (필요 시)
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label}",
            color=color
        )
    
    plt.tight_layout()
    return fig

# 예측 확률 분포 시각화
def plot_prediction_distribution(probabilities, true_label, class_names):
    """
    예측 확률 분포 시각화
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['green' if i == true_label else 'blue' 
              for i in range(len(class_names))]
    
    plt.bar(class_names, probabilities, color=colors, alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title(f'Prediction Distribution (True: {class_names[true_label]})')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return plt.gcf()

# 예제 데이터
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])
class_names = ['Cat', 'Dog', 'Bird']

# Confusion Matrix
fig = plot_confusion_matrix(y_true, y_pred, class_names)
plt.show()

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# 예측 확률 분포
probabilities = np.array([0.7, 0.2, 0.1])
fig = plot_prediction_distribution(probabilities, 0, class_names)
plt.show()
```

---
# 5. 백엔드 개발자를 위한 로깅 전략
---

```python
import logging
import json
from datetime import datetime
import torch

# 구조화된 로깅
class StructuredLogger:
    """
    JSON 형식의 구조화된 로깅
    """
    def __init__(self, log_file='training.jsonl'):
        self.log_file = log_file
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
    
    def log_event(self, event_type, data):
        """구조화된 이벤트 로깅"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_epoch(self, epoch, metrics):
        """Epoch 메트릭 로깅"""
        self.log_event('epoch_end', {
            'epoch': epoch,
            'metrics': metrics
        })
    
    def log_model_info(self, model):
        """모델 정보 로깅"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        self.log_event('model_info', {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_class': model.__class__.__name__
        })
    
    def log_hyperparameters(self, hparams):
        """하이퍼파라미터 로깅"""
        self.log_event('hyperparameters', hparams)
    
    def log_error(self, error_msg, context=None):
        """에러 로깅"""
        self.log_event('error', {
            'message': error_msg,
            'context': context
        })

# 사용 예시
logger = StructuredLogger('training.jsonl')

# 하이퍼파라미터 로깅
logger.log_hyperparameters({
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'Adam',
    'epochs': 100
})

# Epoch 메트릭 로깅
logger.log_epoch(1, {
    'train_loss': 0.5,
    'val_loss': 0.6,
    'train_acc': 0.85,
    'val_acc': 0.82
})

print("구조화된 로깅 완료!")
print("로그 파일: training.jsonl")

# 로그 분석 예시
def analyze_logs(log_file):
    """로그 파일 분석"""
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['event_type'] == 'epoch_end':
                data = entry['data']
                epochs.append(data['epoch'])
                train_losses.append(data['metrics']['train_loss'])
                val_losses.append(data['metrics']['val_loss'])
    
    return epochs, train_losses, val_losses

print("\n로그 분석 함수 준비 완료")
```

---
# 6. 실전 모니터링 체크리스트
---

## 필수 모니터링 항목

### 1. Loss & Accuracy
- ✓ Train/Validation Loss
- ✓ Train/Validation Accuracy
- ✓ Gap 확인 (오버피팅)

### 2. Learning Rate
- ✓ 현재 learning rate
- ✓ LR scheduler 동작 확인

### 3. Gradient
- ✓ Gradient norm
- ✓ Gradient 분포
- ✓ Vanishing/Exploding 확인

### 4. Weight
- ✓ Weight 분포
- ✓ Weight norm
- ✓ Dead neurons 확인

### 5. 시스템 리소스
- ✓ GPU 메모리 사용량
- ✓ GPU 활용률
- ✓ 학습 속도 (samples/sec)

### 6. 예측 결과
- ✓ Confusion Matrix
- ✓ 샘플 예측 시각화
- ✓ 오분류 사례 분석

```python
import torch
from torch.utils.tensorboard import SummaryWriter

class ComprehensiveMonitor:
    """
    종합 모니터링 시스템
    """
    def __init__(self, log_dir='runs/comprehensive'):
        self.writer = SummaryWriter(log_dir)
    
    def monitor_training_step(
        self, model, optimizer, loss, batch_idx, epoch, total_batches
    ):
        """학습 스텝 모니터링"""
        global_step = epoch * total_batches + batch_idx
        
        # 1. Loss
        self.writer.add_scalar('Loss/batch', loss.item(), global_step)
        
        # 2. Learning Rate
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(
                f'LR/group_{i}', param_group['lr'], global_step
            )
        
        # 3. Gradient (주기적으로)
        if batch_idx % 100 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            self.writer.add_scalar(
                'Gradient/total_norm', total_norm, global_step
            )
    
    def monitor_epoch(
        self, model, epoch, train_metrics, val_metrics
    ):
        """Epoch 종료 시 모니터링"""
        # 1. Metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{key}', value, epoch)
        
        # 2. Weight & Gradient 분포
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(
                    f'Gradients/{name}', param.grad, epoch
                )
    
    def monitor_system(self, epoch):
        """시스템 리소스 모니터링"""
        if torch.cuda.is_available():
            # GPU 메모리
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            
            self.writer.add_scalar('System/GPU_Memory_Allocated_GB', allocated, epoch)
            self.writer.add_scalar('System/GPU_Memory_Reserved_GB', reserved, epoch)
    
    def close(self):
        self.writer.close()

print("=== 종합 모니터링 시스템 ===")
print("\n모니터링 항목:")
print("✓ Loss & Accuracy (Train/Val)")
print("✓ Learning Rate")
print("✓ Gradient Norm & Distribution")
print("✓ Weight Distribution")
print("✓ GPU Memory Usage")
print("\n사용법:")
print("monitor = ComprehensiveMonitor('runs/exp')")
print("monitor.monitor_training_step(...)")
print("monitor.monitor_epoch(...)")
print("monitor.close()")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. TensorBoard
- **기능**: Scalars, Images, Graphs, Histograms
- **사용**: SummaryWriter로 로깅
- **실행**: `tensorboard --logdir=runs`

### 2. 학습 곡선 분석
- **정상**: Train/Val loss 모두 감소
- **오버피팅**: Train 감소, Val 증가
- **언더피팅**: 둘 다 높은 수준에서 수렴

### 3. 실시간 모니터링
- **TrainingMonitor**: Epoch/Batch 단위 로깅
- **메트릭 추적**: Loss, Accuracy, LR, Gradient
- **Best model 저장**: Validation loss 기준

### 4. 성능 시각화
- **Confusion Matrix**: 오분류 패턴 분석
- **예측 결과**: 샘플 시각화
- **확률 분포**: 모델 신뢰도 확인

### 5. 백엔드 로깅 전략
- **구조화된 로깅**: JSON 형식
- **이벤트 기반**: epoch_end, error 등
- **분석 용이**: 프로그래밍 방식 분석

### 필수 모니터링 항목
```python
1. Loss & Accuracy (Train/Val)
2. Learning Rate
3. Gradient Norm
4. Weight Distribution
5. GPU Memory
6. Confusion Matrix
```

### 실전 팁
- TensorBoard로 실시간 확인
- 주기적으로 Gradient/Weight 분포 확인
- 오버피팅 조기 발견
- 구조화된 로그로 자동 분석

다음 단원에서는 **모델 배포**를 배워보겠습니다!
