# 12. PyTorch 디버깅 전략

이번 단원에서는 **PyTorch 모델 개발 시 발생하는 문제를 진단하고 해결하는 방법**을 배워보겠습니다.

## 학습 목표
- NaN/Inf 문제 진단 및 해결
- Shape mismatch 오류 디버깅
- Gradient 문제 확인
- 학습이 안 되는 경우 체크리스트
- 백엔드 개발자 관점의 디버깅 도구

## 왜 이 단원이 필요한가?

딥러닝 모델 개발은 **디버깅이 어렵습니다**:

- **에러 없이 실패**: 학습이 되지 않아도 에러가 나지 않음
- **비결정적**: 같은 코드도 다른 결과
- **복잡한 연산**: 텐서 연산의 shape, device, dtype 문제
- **수치적 불안정**: NaN, Inf 발생

백엔드 개발자라면 **로깅, 모니터링, 프로파일링**에 익숙할 것입니다. 이 경험을 딥러닝 디버깅에 적용해봅시다.

---
# 1. NaN/Inf 문제
---

## 1.1 원인과 해결책

### 주요 원인
1. **Learning rate가 너무 큼** → Gradient clipping, LR 감소
2. **Loss 함수 문제** (`log(0)`) → Numerical stability 추가
3. **Gradient Explosion** → Gradient clipping
4. **데이터 정규화 부족** → Normalization

```python
import torch
import torch.nn as nn

# NaN/Inf 감지 유틸리티
def check_nan_inf(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"❌ {name} contains NaN!")
        return True
    if torch.isinf(tensor).any():
        print(f"❌ {name} contains Inf!")
        return True
    return False

# Gradient Clipping 적용
def train_step_with_clip(model, optimizer, loss, max_norm=1.0):
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient norm 제한
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    optimizer.step()

# Anomaly detection 활성화
torch.autograd.set_detect_anomaly(True)

print("NaN/Inf 감지 도구 준비 완료")
```

---
# 2. Shape Mismatch 디버깅
---

```python
import torch
import torch.nn as nn

# Shape 추적 래퍼
class ShapeDebugger(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._register_hooks()
    
    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                print(f"{name}:")
                if isinstance(input, tuple) and len(input) > 0:
                    if isinstance(input[0], torch.Tensor):
                        print(f"  Input: {input[0].shape}")
                if isinstance(output, torch.Tensor):
                    print(f"  Output: {output.shape}")
            return hook
        
        for name, layer in self.model.named_modules():
            if len(list(layer.children())) == 0:
                layer.register_forward_hook(hook_fn(name))
    
    def forward(self, x):
        return self.model(x)

# 실전 팁
def safe_reshape_example():
    x = torch.randn(32, 3, 28, 28)
    
    # ✓ 명시적 reshape
    x_flat = x.view(x.size(0), -1)
    print(f"Flattened: {x_flat.shape}")
    
    # ✓ keepdim으로 차원 유지
    mean = x.mean(dim=1, keepdim=True)
    print(f"Mean (keepdim): {mean.shape}")

safe_reshape_example()
print("\nShape 디버깅 도구 준비 완료")
```

---
# 3. Gradient 문제
---

```python
import torch
import matplotlib.pyplot as plt

# Gradient 통계 분석
def analyze_gradients(model):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item()
            }
    return grad_stats

# Numerical gradient checking
def numerical_gradient(f, x, eps=1e-5):
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_flat = x.flatten()
        x_flat[i] += eps
        f_plus = f(x.view_as(x))
        x_flat[i] -= 2 * eps
        f_minus = f(x.view_as(x))
        grad.flatten()[i] = (f_plus - f_minus) / (2 * eps)
        x_flat[i] += eps
    return grad

print("Gradient 분석 도구 준비 완료")
```

---
# 4. 학습 체크리스트
---

## 체계적 디버깅 프로세스

### 1단계: 데이터 확인
- ✓ Shape, dtype, device
- ✓ 값의 범위 (min, max, mean, std)
- ✓ 레이블 분포

### 2단계: 모델 확인
- ✓ 파라미터 수
- ✓ Forward pass 테스트
- ✓ 출력 shape

### 3단계: Loss 확인
- ✓ 초기 loss가 합리적인가?
- ✓ Loss가 감소하는가?

### 4단계: Overfitting 테스트
- ✓ 작은 데이터셋으로 overfitting 가능한지

```python
import torch
import torch.nn as nn

class TrainingHealthCheck:
    """학습 상태 자동 진단"""
    
    def __init__(self, model, dataloader, criterion, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
    
    def check_data(self):
        print("\n=== 1. 데이터 확인 ===")
        batch = next(iter(self.dataloader))
        X, y = batch[0], batch[1]
        print(f"✓ Batch shape: {X.shape}")
        print(f"✓ Data range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"✓ Data mean: {X.mean():.4f}, std: {X.std():.4f}")
        if X.std() > 10 or X.std() < 0.1:
            print("⚠️  데이터 정규화를 고려하세요")
        return X, y
    
    def check_model(self, sample_input):
        print("\n=== 2. 모델 확인 ===")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input[:1].to(self.device))
            print(f"✓ Output shape: {output.shape}")
        return output
    
    def check_loss(self, output, target):
        print("\n=== 3. Loss 확인 ===")
        loss = self.criterion(output, target[:1].to(self.device))
        print(f"✓ Initial loss: {loss.item():.4f}")
        
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            num_classes = output.size(-1)
            expected = -torch.log(torch.tensor(1.0 / num_classes))
            print(f"✓ Expected initial loss: ~{expected.item():.4f}")
    
    def run_all_checks(self):
        X, y = self.check_data()
        output = self.check_model(X)
        self.check_loss(output, y)
        print("\n=== 체크 완료 ===")

print("TrainingHealthCheck 클래스 준비 완료")
```

---
# 5. 백엔드 개발자를 위한 도구
---

```python
import logging
import time
import torch

# 구조화된 로깅
def setup_training_logger(log_file='training.log'):
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# 타이머
class Timer:
    def __init__(self, name="Block"):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"{self.name}: {elapsed:.4f}s")

# 메모리 모니터링
def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
    else:
        print("CUDA not available")

# 사용 예시
logger = setup_training_logger()
logger.info("로깅 시스템 준비 완료")

with Timer("Example operation"):
    x = torch.randn(1000, 1000)
    y = x @ x.T

print_memory_stats()
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. NaN/Inf 문제
- **진단**: `torch.isnan()`, `torch.isinf()`, anomaly detection
- **해결**: Gradient clipping, LR 조정, BatchNorm

### 2. Shape Mismatch
- **도구**: Shape debugger, forward hooks
- **팁**: 명시적 reshape, keepdim, einsum

### 3. Gradient 문제
- **Vanishing/Exploding**: 통계 분석, numerical checking
- **해결**: 적절한 초기화, 활성화 함수, BatchNorm

### 4. 체계적 디버깅
1. 데이터 확인
2. 모델 확인
3. Loss 확인
4. Overfitting 테스트

### 5. 백엔드 도구
- 로깅, 타이머, 메모리 모니터링

다음 단원에서는 **메모리 최적화**를 배워보겠습니다!
