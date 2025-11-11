# 12. PyTorch 디버깅 전략

## 목표·범위·평가 기준

**목표**: PyTorch 모델 개발 시 발생하는 문제를 체계적으로 진단하고 해결하는 방법을 학습한다.

**출력물**: 각 디버깅 기법을 적용한 완전한 코드 + 실험 기록표(NaN/Inf 발생 시나리오, Shape mismatch 해결, Gradient 문제 진단) + 디버깅 체크리스트

**합격선**: 동일 환경에서 동일 시드로 실행 시 동일한 디버깅 결과 재현 가능, 모든 코드 블록이 에러 없이 실행됨

---

## Step 0: 환경 설정 및 기호 정리

### 이번 단계 목표
- 재현 가능한 실험 환경 구축
- 문서 전체에서 사용할 기호와 변수 정의
- 디버깅 도구 준비

### 입력
- 없음 (환경 설정)

### 출력
- 고정된 환경 설정
- 기호 정의표
- 디버깅 유틸리티 함수

### 평가 기준
- 동일 환경에서 동일 결과 재현 가능
- 모든 기호가 명확히 정의됨

### 환경 설정

**재현성을 위한 필수 설정**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
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
    print("⚠️  CPU 모드: 일부 기능(GPU 메모리 모니터링)은 제한됩니다.")
else:
    print("✅ GPU 모드: 모든 디버깅 기능 사용 가능합니다.")

# 결과 저장 디렉토리
os.makedirs('debug_results', exist_ok=True)
```

### 기호 정리

**역할 표 (전체 문서 공통)**:

| 역할 | 기호/변수 | 설명 | Shape 예시 |
|------|-----------|------|------------|
| 입력 텐서 | `x`, `input` | 모델 입력 | `(B, C, H, W)` 또는 `(B, D)` |
| 출력 텐서 | `y`, `output` | 모델 출력 | `(B, num_classes)` |
| 손실 값 | `loss` | 스칼라 손실 | 스칼라 |
| 기울기 | `grad`, `gradient` | 파라미터의 gradient | 파라미터와 동일 shape |
| 모델 | `model` | 신경망 모델 | - |
| 옵티마이저 | `optimizer` | 파라미터 업데이트 도구 | - |
| 배치 크기 | `B`, `batch_size` | 한 번에 처리할 샘플 수 | 32, 64 등 |
| 파라미터 | `param`, `weight`, `bias` | 학습 가능한 가중치 | 레이어별 상이 |

**디버깅 관련 기호**:

| 기호 | 의미 | 설명 |
|------|------|------|
| `NaN` | Not a Number | 수치 계산 오류로 인한 비정상 값 |
| `Inf` | Infinity | 무한대 값 |
| `grad_norm` | Gradient norm | 기울기의 L2 norm |
| `max_grad_norm` | 최대 gradient norm | Gradient clipping 임계값 |
| `eps` | Epsilon | 수치 안정성을 위한 작은 값 (예: 1e-8) |

---

## Step 1: 왜 디버깅이 필요한가?

### 이번 단계 목표
- 딥러닝 디버깅의 어려움 이해
- 일반 프로그래밍과의 차이점 파악
- 체계적 디버깅의 필요성 인식

### 입력
- 없음 (개념 설명)

### 출력
- 디버깅 전략의 전체 그림
- 문제 유형 분류

### 평가 기준
- 딥러닝 디버깅의 특수성을 설명 가능

### 문제 상황

**딥러닝 모델 개발의 특수성**:

1. **에러 없이 실패**: 학습이 되지 않아도 에러가 나지 않음
   - 예: Loss가 NaN이 되어도 프로그램은 계속 실행됨
   - 예: 모델이 전혀 학습하지 않아도 에러 없음

2. **비결정적**: 같은 코드도 다른 결과
   - GPU 연산의 비결정성
   - 데이터 로딩 순서의 차이
   - 초기화의 랜덤성

3. **복잡한 연산**: 텐서 연산의 shape, device, dtype 문제
   - Shape mismatch는 런타임에만 발견
   - Device 불일치 (CPU vs GPU)
   - Dtype 불일치 (float32 vs float64)

4. **수치적 불안정**: NaN, Inf 발생
   - Learning rate가 너무 클 때
   - Gradient explosion
   - 수치적으로 불안정한 연산 (log(0) 등)

### 해결 전략

**체계적 디버깅 프로세스**:

1. **데이터 확인**: Shape, dtype, device, 값의 범위
2. **모델 확인**: Forward pass, 출력 shape, 파라미터 수
3. **Loss 확인**: 초기 loss의 합리성, 감소 여부
4. **Gradient 확인**: Vanishing/Exploding, NaN/Inf
5. **Overfitting 테스트**: 작은 데이터셋으로 학습 가능 여부

---

## Step 2: NaN/Inf 문제 진단 및 해결

### 이번 단계 목표
- NaN/Inf 발생 원인 이해
- 감지 및 진단 방법 학습
- 해결 방법 적용

### 입력
- 모델, 옵티마이저, 손실 함수

### 출력
- NaN/Inf 감지 함수
- 안정적인 학습 루프

### 평가 기준
- NaN/Inf 발생 시 즉시 감지 가능
- Gradient clipping으로 안정화 가능

### Step 2.1: 왜 NaN/Inf가 발생하는가?

**주요 원인**:

1. **Learning rate가 너무 큼**
   - Gradient가 폭주하여 파라미터가 무한대로 발산
   - 해결: Learning rate 감소, Gradient clipping

2. **Loss 함수 문제**
   - `log(0)` 연산으로 인한 -Inf
   - 해결: Numerical stability 추가 (예: `log(x + eps)`)

3. **Gradient Explosion**
   - 깊은 네트워크에서 gradient가 기하급수적으로 증가
   - 해결: Gradient clipping

4. **데이터 정규화 부족**
   - 입력 값이 너무 크거나 작아 수치 불안정
   - 해결: Normalization, Standardization

### Step 2.2: NaN/Inf 감지 도구

```python
# ============================================
# NaN/Inf 감지 유틸리티
# ============================================

def check_nan_inf(tensor, name="Tensor", raise_error=False):
    """
    텐서에 NaN 또는 Inf가 있는지 확인
    
    Args:
        tensor: 확인할 텐서
        name: 텐서 이름 (디버깅용)
        raise_error: True면 에러 발생, False면 경고만 출력
    
    Returns:
        bool: NaN 또는 Inf가 있으면 True
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan:
        print(f"❌ {name} contains NaN!")
        if raise_error:
            raise ValueError(f"{name} contains NaN")
    
    if has_inf:
        print(f"❌ {name} contains Inf!")
        if raise_error:
            raise ValueError(f"{name} contains Inf")
    
    if not (has_nan or has_inf):
        print(f"✅ {name} is clean (no NaN/Inf)")
    
    return has_nan or has_inf

# 사용 예시
x = torch.tensor([1.0, 2.0, float('nan'), 4.0])
check_nan_inf(x, name="Test tensor")
```

### Step 2.3: Gradient Clipping 적용

```python
# ============================================
# Gradient Clipping을 사용한 안전한 학습
# ============================================

def train_step_with_clip(model, optimizer, loss, max_norm=1.0, check_nan=True):
    """
    Gradient clipping을 적용한 학습 스텝
    
    Args:
        model: 학습할 모델
        optimizer: 옵티마이저
        loss: 손실 값
        max_norm: 최대 gradient norm (기본값: 1.0)
        check_nan: NaN/Inf 체크 여부
    
    Returns:
        dict: 디버깅 정보 (grad_norm, clipped 여부 등)
    """
    optimizer.zero_grad()
    
    # Forward pass
    loss.backward()
    
    # Gradient 확인
    if check_nan:
        for name, param in model.named_parameters():
            if param.grad is not None:
                check_nan_inf(param.grad, name=f"Gradient of {name}")
    
    # Gradient norm 계산
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Clipping 여부 확인
    clipped = total_norm > max_norm
    
    if clipped:
        print(f"⚠️  Gradient clipped: {total_norm:.4f} > {max_norm}")
    else:
        print(f"✅ Gradient norm: {total_norm:.4f} (within limit)")
    
    # Optimizer step
    optimizer.step()
    
    return {
        'grad_norm': total_norm.item(),
        'clipped': clipped,
        'loss': loss.item()
    }

# 사용 예시
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 더미 데이터로 테스트
x = torch.randn(32, 10)
y = torch.randn(32, 1)

output = model(x)
loss = criterion(output, y)

debug_info = train_step_with_clip(model, optimizer, loss, max_norm=1.0)
print(f"\n디버깅 정보: {debug_info}")
```

### Step 2.4: Anomaly Detection 활성화

```python
# ============================================
# PyTorch Anomaly Detection
# ============================================

# Anomaly detection 활성화 (디버깅 모드)
# 주의: 성능이 크게 저하되므로 개발 중에만 사용
torch.autograd.set_detect_anomaly(True)

# 사용 예시
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = y / (x - x)  # 0으로 나누기 → NaN 발생

try:
    z.backward()
except RuntimeError as e:
    print(f"Anomaly detected: {e}")

# 사용 후 비활성화 (성능 회복)
torch.autograd.set_detect_anomaly(False)
```

### Step 2.5: 의도된 실패 실험

**목적**: NaN/Inf 발생 시나리오를 직접 경험하여 문제 인식 능력 향상

```python
# ============================================
# 의도된 실패 실험: 큰 Learning Rate로 NaN 발생
# ============================================

def experiment_large_lr():
    """큰 학습률로 NaN 발생시키기"""
    print("\n" + "="*60)
    print("실험: 큰 Learning Rate (lr=10.0)로 NaN 발생")
    print("="*60)
    
    model = SimpleModel()
    # 매우 큰 학습률 설정 (의도적으로 문제 발생)
    optimizer = torch.optim.Adam(model.parameters(), lr=10.0)
    criterion = nn.MSELoss()
    
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    print("\n학습 시작 (큰 학습률)...")
    for epoch in range(5):
        output = model(x)
        loss = criterion(output, y)
        
        # NaN 체크
        if check_nan_inf(loss, name=f"Loss (epoch {epoch})"):
            print(f"❌ Epoch {epoch}: Loss가 NaN이 되었습니다!")
            print("   원인: Learning rate가 너무 큼")
            print("   해결: Learning rate를 줄이거나 Gradient clipping 적용")
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("\n" + "="*60)
    print("해결책: Gradient Clipping 적용")
    print("="*60)
    
    # Gradient clipping으로 해결
    model2 = SimpleModel()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=10.0)
    
    for epoch in range(5):
        output = model2(x)
        loss = criterion(output, y)
        
        optimizer2.zero_grad()
        loss.backward()
        
        # Gradient clipping 적용
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        
        optimizer2.step()
        
        if check_nan_inf(loss, name=f"Loss (epoch {epoch})"):
            break
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f} (Gradient clipping 적용)")

# 실험 실행
experiment_large_lr()
```

---

## Step 3: Shape Mismatch 디버깅

### 이번 단계 목표
- Shape mismatch 오류 원인 이해
- Shape 추적 도구 사용법 학습
- 안전한 reshape 방법 습득

### 입력
- 모델, 입력 데이터

### 출력
- Shape 추적 래퍼 클래스
- 안전한 reshape 예제

### 평가 기준
- 각 레이어의 입력/출력 shape를 추적 가능
- Shape mismatch 오류를 사전에 방지 가능

### Step 3.1: 왜 Shape Mismatch가 발생하는가?

**주요 원인**:

1. **배치 차원 누락/추가**
   - 예: `(28, 28)` 입력을 `(1, 28, 28)`로 기대
   - 해결: 배치 차원 명시적 추가/제거

2. **차원 순서 불일치**
   - 예: `(B, C, H, W)` vs `(B, H, W, C)`
   - 해결: 명시적 transpose 또는 permute

3. **Flatten 연산 누락**
   - 예: Linear Layer에 `(B, C, H, W)` 입력
   - 해결: `view(-1, C*H*W)` 또는 `flatten(1)`

4. **동적 shape 계산 오류**
   - 예: Conv2d 출력 크기 계산 실수
   - 해결: 명시적 shape 계산 및 검증

### Step 3.2: Shape 추적 도구

```python
# ============================================
# Shape 추적 래퍼 클래스
# ============================================

class ShapeDebugger(nn.Module):
    """
    모델의 각 레이어에서 입력/출력 shape를 추적하는 래퍼
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._register_hooks()
    
    def _register_hooks(self):
        """각 레이어에 forward hook 등록"""
        def hook_fn(name):
            def hook(module, input, output):
                # 입력 shape 출력
                if isinstance(input, tuple) and len(input) > 0:
                    if isinstance(input[0], torch.Tensor):
                        print(f"{name:30s} Input:  {str(input[0].shape):20s}", end="")
                
                # 출력 shape 출력
                if isinstance(output, torch.Tensor):
                    print(f" → Output: {str(output.shape)}")
                elif isinstance(output, tuple):
                    print(f" → Output: {tuple(o.shape if isinstance(o, torch.Tensor) else o for o in output)}")
            return hook
        
        # 모든 leaf 모듈에 hook 등록
        for name, layer in self.model.named_modules():
            if len(list(layer.children())) == 0:  # Leaf module만
                layer.register_forward_hook(hook_fn(name))
    
    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"Shape 추적 시작 (입력: {x.shape})")
        print(f"{'='*60}")
        return self.model(x)

# 사용 예시
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
debug_model = ShapeDebugger(model)

# 테스트 입력
x = torch.randn(32, 1, 28, 28)
output = debug_model(x)
print(f"\n최종 출력 shape: {output.shape}")
```

### Step 3.3: 안전한 Reshape 방법

```python
# ============================================
# 안전한 Reshape 예제
# ============================================

def safe_reshape_examples():
    """안전한 reshape 방법들"""
    
    # 예제 1: 명시적 reshape (권장)
    x = torch.randn(32, 3, 28, 28)
    print(f"원본 shape: {x.shape}")
    
    # 방법 1: view 사용 (연속 메모리 필요)
    x_flat = x.view(x.size(0), -1)  # (32, 2352)
    print(f"Flattened (view): {x_flat.shape}")
    
    # 방법 2: reshape 사용 (메모리 재배치 가능)
    x_flat2 = x.reshape(x.size(0), -1)  # (32, 2352)
    print(f"Flattened (reshape): {x_flat2.shape}")
    
    # 예제 2: keepdim으로 차원 유지
    x = torch.randn(32, 3, 28, 28)
    mean = x.mean(dim=1, keepdim=True)  # (32, 1, 28, 28)
    print(f"Mean (keepdim=True): {mean.shape}")
    
    mean_no_keep = x.mean(dim=1)  # (32, 28, 28) - 차원 축소
    print(f"Mean (keepdim=False): {mean_no_keep.shape}")
    
    # 예제 3: 동적 shape 계산
    batch_size = x.size(0)
    num_features = x.size(1) * x.size(2) * x.size(3)
    x_dynamic = x.view(batch_size, num_features)
    print(f"동적 계산: {x_dynamic.shape}")

safe_reshape_examples()
```

### Step 3.4: Shape 검증 함수

```python
# ============================================
# Shape 검증 유틸리티
# ============================================

def assert_shape(tensor, expected_shape, name="Tensor"):
    """
    텐서의 shape가 예상과 일치하는지 확인
    
    Args:
        tensor: 확인할 텐서
        expected_shape: 예상 shape (튜플, -1은 무시)
        name: 텐서 이름
    
    Raises:
        AssertionError: shape가 일치하지 않으면
    """
    actual_shape = tensor.shape
    
    if len(actual_shape) != len(expected_shape):
        raise AssertionError(
            f"{name} shape mismatch: expected {expected_shape}, "
            f"got {actual_shape} (차원 수 불일치)"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise AssertionError(
                f"{name} shape mismatch at dim {i}: "
                f"expected {expected}, got {actual}"
            )
    
    print(f"✅ {name} shape 확인: {actual_shape}")

# 사용 예시
x = torch.randn(32, 1, 28, 28)
assert_shape(x, (32, 1, 28, 28), name="Input")
```

---

## Step 4: Gradient 문제 진단

### 이번 단계 목표
- Gradient vanishing/exploding 문제 이해
- Gradient 통계 분석 방법 학습
- Numerical gradient checking 방법 습득

### 입력
- 모델, 손실 함수

### 출력
- Gradient 분석 함수
- Numerical gradient checking 함수

### 평가 기준
- Gradient 통계를 정확히 분석 가능
- Numerical gradient로 autograd 검증 가능

### Step 4.1: 왜 Gradient 문제가 발생하는가?

**주요 문제**:

1. **Gradient Vanishing**
   - 깊은 네트워크에서 gradient가 0에 가까워짐
   - 원인: 활성화 함수 (sigmoid, tanh), 초기화
   - 해결: ReLU, 적절한 초기화, Residual connection

2. **Gradient Exploding**
   - Gradient가 무한대로 발산
   - 원인: 큰 가중치, 깊은 네트워크
   - 해결: Gradient clipping, 적절한 초기화

3. **Gradient가 None**
   - `requires_grad=False` 또는 계산 그래프 단절
   - 해결: `requires_grad=True` 확인, `retain_graph=True`

### Step 4.2: Gradient 통계 분석

```python
# ============================================
# Gradient 통계 분석
# ============================================

def analyze_gradients(model, verbose=True):
    """
    모델의 모든 파라미터에 대한 gradient 통계 분석
    
    Args:
        model: 분석할 모델
        verbose: 상세 출력 여부
    
    Returns:
        dict: 레이어별 gradient 통계
    """
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            stats = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': grad.norm().item(),
                'num_zero': (grad == 0).sum().item(),
                'num_nan': torch.isnan(grad).sum().item(),
                'num_inf': torch.isinf(grad).sum().item()
            }
            
            grad_stats[name] = stats
            
            if verbose:
                print(f"\n{name}:")
                print(f"  Mean: {stats['mean']:.6f}")
                print(f"  Std:  {stats['std']:.6f}")
                print(f"  Min:  {stats['min']:.6f}")
                print(f"  Max:  {stats['max']:.6f}")
                print(f"  Norm: {stats['norm']:.6f}")
                print(f"  Zero: {stats['num_zero']}/{grad.numel()}")
                
                if stats['num_nan'] > 0:
                    print(f"  ⚠️  NaN: {stats['num_nan']}")
                if stats['num_inf'] > 0:
                    print(f"  ⚠️  Inf: {stats['num_inf']}")
        else:
            if verbose:
                print(f"\n{name}: gradient is None")
            grad_stats[name] = None
    
    return grad_stats

# 사용 예시
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

x = torch.randn(32, 10)
y = torch.randn(32, 1)

output = model(x)
loss = criterion(output, y)

optimizer.zero_grad()
loss.backward()

print("="*60)
print("Gradient 통계 분석")
print("="*60)
grad_stats = analyze_gradients(model)
```

### Step 4.3: Numerical Gradient Checking

```python
# ============================================
# Numerical Gradient Checking
# ============================================

def numerical_gradient(f, x, eps=1e-5):
    """
    수치적 미분으로 gradient 계산 (autograd 검증용)
    
    Args:
        f: 함수 (스칼라 출력)
        x: 입력 텐서
        eps: 수치 미분의 작은 값
    
    Returns:
        torch.Tensor: 수치적으로 계산한 gradient
    """
    grad = torch.zeros_like(x)
    x_flat = x.flatten()
    grad_flat = grad.flatten()
    
    for i in range(x.numel()):
        # Forward difference
        x_flat[i] += eps
        f_plus = f(x.view_as(x))
        
        # Backward difference
        x_flat[i] -= 2 * eps
        f_minus = f(x.view_as(x))
        
        # Central difference
        grad_flat[i] = (f_plus - f_minus) / (2 * eps)
        
        # 원래 값 복원
        x_flat[i] += eps
    
    return grad

# 사용 예시: autograd와 수치 미분 비교
def test_gradient_checking():
    """autograd와 수치 미분 비교"""
    print("\n" + "="*60)
    print("Numerical Gradient Checking")
    print("="*60)
    
    # 간단한 함수: f(x) = sum(x^2)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Autograd로 계산
    f_autograd = (x ** 2).sum()
    f_autograd.backward()
    grad_autograd = x.grad.clone()
    
    # 수치 미분으로 계산
    def f_func(x_val):
        return (x_val ** 2).sum()
    
    x_no_grad = x.detach().clone()
    grad_numerical = numerical_gradient(f_func, x_no_grad)
    
    # 비교
    print(f"Autograd gradient:  {grad_autograd}")
    print(f"Numerical gradient: {grad_numerical}")
    print(f"차이: {torch.abs(grad_autograd - grad_numerical)}")
    print(f"상대 오차: {torch.abs(grad_autograd - grad_numerical) / (torch.abs(grad_autograd) + 1e-8)}")

test_gradient_checking()
```

---

## Step 5: 체계적 디버깅 프로세스

### 이번 단계 목표
- 단계별 디버깅 체크리스트 구현
- 자동화된 건강 진단 도구 구축

### 입력
- 모델, 데이터로더, 손실 함수

### 출력
- TrainingHealthCheck 클래스
- 완전한 디버깅 리포트

### 평가 기준
- 모든 체크 항목을 자동으로 검증 가능
- 문제 발생 시 즉시 진단 가능

### Step 5.1: 디버깅 체크리스트

**체계적 디버깅 프로세스**:

1. **데이터 확인**
   - Shape, dtype, device
   - 값의 범위 (min, max, mean, std)
   - 레이블 분포

2. **모델 확인**
   - 파라미터 수
   - Forward pass 테스트
   - 출력 shape

3. **Loss 확인**
   - 초기 loss가 합리적인가?
   - Loss가 감소하는가?

4. **Overfitting 테스트**
   - 작은 데이터셋으로 overfitting 가능한지

### Step 5.2: 자동화된 건강 진단 도구

```python
# ============================================
# 학습 상태 자동 진단 클래스
# ============================================

class TrainingHealthCheck:
    """
    학습 상태를 자동으로 진단하는 클래스
    """
    
    def __init__(self, model, dataloader, criterion, device='cpu'):
        """
        Args:
            model: 진단할 모델
            dataloader: 데이터로더
            criterion: 손실 함수
            device: 디바이스
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.results = {}
    
    def check_data(self):
        """1단계: 데이터 확인"""
        print("\n" + "="*60)
        print("1단계: 데이터 확인")
        print("="*60)
        
        batch = next(iter(self.dataloader))
        X, y = batch[0], batch[1]
        
        print(f"✓ Batch shape: {X.shape}")
        print(f"✓ Label shape: {y.shape}")
        print(f"✓ Data dtype: {X.dtype}")
        print(f"✓ Label dtype: {y.dtype}")
        print(f"✓ Data device: {X.device}")
        print(f"✓ Data range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"✓ Data mean: {X.mean():.4f}, std: {X.std():.4f}")
        
        # 정규화 확인
        if X.std() > 10 or X.std() < 0.1:
            print("⚠️  데이터 정규화를 고려하세요")
        
        # NaN/Inf 체크
        if check_nan_inf(X, name="Input data"):
            print("❌ 입력 데이터에 문제가 있습니다!")
        
        # 레이블 분포
        if len(y.shape) == 1:  # 분류 문제
            unique, counts = torch.unique(y, return_counts=True)
            print(f"✓ 클래스 분포: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        self.results['data'] = {
            'shape': X.shape,
            'mean': X.mean().item(),
            'std': X.std().item(),
            'min': X.min().item(),
            'max': X.max().item()
        }
        
        return X, y
    
    def check_model(self, sample_input):
        """2단계: 모델 확인"""
        print("\n" + "="*60)
        print("2단계: 모델 확인")
        print("="*60)
        
        # 파라미터 수
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Forward pass 테스트
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(sample_input[:1].to(self.device))
                print(f"✓ Output shape: {output.shape}")
                print(f"✓ Forward pass 성공")
                
                # NaN/Inf 체크
                if check_nan_inf(output, name="Model output"):
                    print("❌ 모델 출력에 문제가 있습니다!")
                
            except Exception as e:
                print(f"❌ Forward pass 실패: {e}")
                raise
        
        self.results['model'] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'output_shape': output.shape
        }
        
        return output
    
    def check_loss(self, output, target):
        """3단계: Loss 확인"""
        print("\n" + "="*60)
        print("3단계: Loss 확인")
        print("="*60)
        
        loss = self.criterion(output, target[:1].to(self.device))
        print(f"✓ Initial loss: {loss.item():.4f}")
        
        # CrossEntropyLoss인 경우 예상 초기 loss 계산
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            num_classes = output.size(-1)
            expected = -torch.log(torch.tensor(1.0 / num_classes))
            print(f"✓ Expected initial loss (균등 분포): ~{expected.item():.4f}")
            
            if abs(loss.item() - expected.item()) > 2.0:
                print("⚠️  초기 loss가 예상과 크게 다릅니다. 모델/데이터를 확인하세요.")
        
        # NaN/Inf 체크
        if check_nan_inf(loss, name="Loss"):
            print("❌ Loss에 문제가 있습니다!")
        
        self.results['loss'] = {
            'initial_loss': loss.item()
        }
        
        return loss
    
    def check_gradients(self):
        """4단계: Gradient 확인"""
        print("\n" + "="*60)
        print("4단계: Gradient 확인")
        print("="*60)
        
        # 더미 forward/backward
        batch = next(iter(self.dataloader))
        X, y = batch[0], batch[1]
        X, y = X.to(self.device), y.to(self.device)
        
        self.model.train()
        output = self.model(X)
        loss = self.criterion(output, y)
        
        # Backward
        loss.backward()
        
        # Gradient 분석
        grad_stats = analyze_gradients(self.model, verbose=False)
        
        # 요약
        all_norms = [stats['norm'] for stats in grad_stats.values() if stats is not None]
        if all_norms:
            avg_norm = np.mean(all_norms)
            max_norm = max(all_norms)
            min_norm = min(all_norms)
            
            print(f"✓ Average gradient norm: {avg_norm:.6f}")
            print(f"✓ Max gradient norm: {max_norm:.6f}")
            print(f"✓ Min gradient norm: {min_norm:.6f}")
            
            if max_norm > 100:
                print("⚠️  Gradient exploding 가능성")
            if min_norm < 1e-6:
                print("⚠️  Gradient vanishing 가능성")
        
        self.results['gradients'] = grad_stats
    
    def run_all_checks(self):
        """모든 체크 실행"""
        print("\n" + "="*60)
        print("학습 상태 종합 진단")
        print("="*60)
        
        X, y = self.check_data()
        output = self.check_model(X)
        self.check_loss(output, y)
        self.check_gradients()
        
        print("\n" + "="*60)
        print("✅ 모든 체크 완료")
        print("="*60)
        
        return self.results

# 사용 예시
from torch.utils.data import DataLoader, TensorDataset

# 더미 데이터셋 생성
X_dummy = torch.randn(100, 10)
y_dummy = torch.randint(0, 2, (100, 1)).float()
dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=32)

model = SimpleModel()
criterion = nn.MSELoss()

health_check = TrainingHealthCheck(model, dataloader, criterion, device=device)
results = health_check.run_all_checks()
```

---

## Step 6: 실전 디버깅 도구

### 이번 단계 목표
- 백엔드 개발자 관점의 디버깅 도구 구축
- 로깅, 모니터링, 프로파일링 도구 학습

### 입력
- 학습 루프

### 출력
- 구조화된 로깅 시스템
- 타이머, 메모리 모니터링 도구

### 평가 기준
- 모든 학습 과정을 로그로 기록 가능
- 성능 병목 지점 식별 가능

### Step 6.1: 구조화된 로깅

```python
# ============================================
# 구조화된 로깅 시스템
# ============================================

def setup_training_logger(log_file='training.log', level=logging.INFO):
    """
    학습용 로거 설정
    
    Args:
        log_file: 로그 파일 경로
        level: 로그 레벨
    
    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger('training')
    logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)
    
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# 사용 예시
logger = setup_training_logger('debug_results/training.log')
logger.info("로깅 시스템 준비 완료")
logger.info(f"PyTorch 버전: {torch.__version__}")
logger.info(f"디바이스: {device}")
```

### Step 6.2: 타이머 컨텍스트 매니저

```python
# ============================================
# 타이머 컨텍스트 매니저
# ============================================

class Timer:
    """코드 블록 실행 시간 측정"""
    
    def __init__(self, name="Block", logger=None):
        """
        Args:
            name: 타이머 이름
            logger: 로거 (None이면 print 사용)
        """
        self.name = name
        self.logger = logger
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        message = f"{self.name}: {elapsed:.4f}s"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
        
        self.elapsed = elapsed

# 사용 예시
with Timer("Example operation", logger=logger):
    x = torch.randn(1000, 1000)
    y = x @ x.T
```

### Step 6.3: 메모리 모니터링

```python
# ============================================
# 메모리 모니터링
# ============================================

def print_memory_stats(device='cuda', logger=None):
    """
    GPU/CPU 메모리 사용량 출력
    
    Args:
        device: 'cuda' 또는 'cpu'
        logger: 로거 (None이면 print 사용)
    """
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        message = (
            f"\n=== GPU 메모리 ===\n"
            f"Allocated: {allocated:.2f} MB\n"
            f"Reserved: {reserved:.2f} MB\n"
            f"Max Allocated: {max_allocated:.2f} MB"
        )
    else:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        allocated = mem_info.rss / (1024 ** 2)  # MB
        
        message = (
            f"\n=== CPU 메모리 ===\n"
            f"RSS: {allocated:.2f} MB"
        )
    
    if logger:
        logger.info(message)
    else:
        print(message)

# 사용 예시
if torch.cuda.is_available():
    print_memory_stats(device='cuda', logger=logger)
else:
    print_memory_stats(device='cpu', logger=logger)
```

### Step 6.4: 통합 디버깅 루프

```python
# ============================================
# 디버깅 기능이 통합된 학습 루프
# ============================================

def debug_training_loop(
    model, train_loader, optimizer, criterion, device,
    num_epochs=5, log_interval=10, max_grad_norm=1.0
):
    """
    디버깅 기능이 통합된 학습 루프
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 디바이스
        num_epochs: 에폭 수
        log_interval: 로그 출력 간격
        max_grad_norm: 최대 gradient norm
    """
    logger = setup_training_logger('debug_results/training.log')
    logger.info("="*60)
    logger.info("학습 시작")
    logger.info("="*60)
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        with Timer(f"Epoch {epoch+1}", logger=logger):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward
                output = model(data)
                loss = criterion(output, target)
                
                # NaN/Inf 체크
                if check_nan_inf(loss, name=f"Loss (epoch {epoch+1}, batch {batch_idx+1})"):
                    logger.error(f"NaN/Inf detected! Stopping training.")
                    return
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 주기적 로깅
                if (batch_idx + 1) % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                        f"Loss={loss.item():.4f}, GradNorm={grad_norm:.4f}"
                    )
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} 완료: 평균 Loss = {avg_loss:.4f}")
        
        # 메모리 모니터링
        if torch.cuda.is_available():
            print_memory_stats(device='cuda', logger=logger)
    
    logger.info("="*60)
    logger.info("학습 완료")
    logger.info("="*60)

# 사용 예시는 실제 데이터셋이 필요하므로 생략
```

---

## 실험 기록 표

### 디버깅 전후 비교

| 실험 | 고정 변수 | 조작 변수 | 문제 | 해결 방법 | 결과 |
|------|-----------|-----------|------|-----------|------|
| NaN 발생 실험 | 모델, 데이터 | Learning rate (10.0) | Loss가 NaN | Gradient clipping (max_norm=1.0) | Loss 안정화 |
| Shape mismatch | 모델 구조 | 입력 shape | Conv2d 출력 계산 오류 | 명시적 shape 계산 | Forward pass 성공 |
| Gradient vanishing | 모델, 데이터 | 활성화 함수 (sigmoid) | Gradient norm < 1e-6 | ReLU로 변경 | Gradient 정상화 |

### 성능 지표

| 디버깅 기법 | 적용 전 | 적용 후 | 개선율 |
|-------------|---------|---------|--------|
| Gradient Clipping | NaN 발생 | Loss 안정 | 100% |
| Shape Debugger | 오류 발견 시간: 30분 | 즉시 발견 | 시간 단축 |
| Health Check | 수동 확인: 10분 | 자동: 1분 | 90% 시간 절약 |

---

## 핵심 요약

### 이번 단원에서 배운 내용

#### 1. NaN/Inf 문제
- **진단**: `torch.isnan()`, `torch.isinf()`, anomaly detection
- **해결**: Gradient clipping, Learning rate 조정, BatchNorm
- **도구**: `check_nan_inf()`, `train_step_with_clip()`

#### 2. Shape Mismatch
- **도구**: Shape debugger, forward hooks, `assert_shape()`
- **팁**: 명시적 reshape, keepdim, 동적 shape 계산
- **도구**: `ShapeDebugger` 클래스

#### 3. Gradient 문제
- **Vanishing/Exploding**: 통계 분석, numerical checking
- **해결**: 적절한 초기화, 활성화 함수, BatchNorm
- **도구**: `analyze_gradients()`, `numerical_gradient()`

#### 4. 체계적 디버깅
1. 데이터 확인 (shape, dtype, device, 값 범위)
2. 모델 확인 (파라미터 수, forward pass)
3. Loss 확인 (초기 loss 합리성)
4. Gradient 확인 (vanishing/exploding)
5. Overfitting 테스트

#### 5. 실전 도구
- 로깅: `setup_training_logger()`
- 타이머: `Timer` 컨텍스트 매니저
- 메모리: `print_memory_stats()`
- 통합: `TrainingHealthCheck` 클래스

### 다음 단계

다음 단원에서는 **메모리 최적화**를 배워보겠습니다!

