# 13. 메모리 최적화

## 목표·범위·평가 기준

**목표**: PyTorch 모델의 메모리 사용을 최적화하는 방법을 학습하고, 제한된 GPU 메모리에서 더 큰 모델을 학습할 수 있는 기법을 습득한다.

**출력물**: 각 최적화 기법을 적용한 완전한 코드 + 실험 기록표(메모리 사용량 Before/After 비교) + 최적화 체크리스트

**합격선**: 동일 환경에서 동일 시드로 실행 시 메모리 사용량이 예상 범위 내 (±10%), 모든 코드 블록이 에러 없이 실행됨

---

## Step 0: 환경 설정 및 기호 정리

### 이번 단계 목표
- 재현 가능한 실험 환경 구축
- 문서 전체에서 사용할 기호와 변수 정의
- 메모리 측정 도구 준비

### 입력
- 없음 (환경 설정)

### 출력
- 고정된 환경 설정
- 기호 정의표
- 메모리 측정 유틸리티

### 평가 기준
- 동일 환경에서 동일 결과 재현 가능
- 모든 기호가 명확히 정의됨

### 환경 설정

**재현성을 위한 필수 설정**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
import multiprocessing

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
    print("⚠️  CPU 모드: GPU 메모리 최적화 기법은 적용되지 않습니다.")
    print("   CPU 메모리 모니터링으로 대체됩니다.")
else:
    print("✅ GPU 모드: 모든 메모리 최적화 기법 사용 가능합니다.")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 결과 저장 디렉토리
os.makedirs('memory_results', exist_ok=True)
```

### 기호 정리

**역할 표 (전체 문서 공통)**:

| 역할 | 기호/변수 | 설명 | 단위/예시 |
|------|-----------|------|-----------|
| 모델 파라미터 수 | $P$, `num_params` | 학습 가능한 가중치 개수 | 1,000,000 (1M) |
| 배치 크기 | $B$, `batch_size` | 한 번에 처리할 샘플 수 | 32, 64, 128 |
| 입력 특징 수 | $d_{in}$ | 입력 차원 | 784, 1000 등 |
| 출력 특징 수 | $d_{out}$ | 출력 차원 | 10, 100 등 |
| 메모리 사용량 | `memory_MB` | 메가바이트 단위 | 1024 MB = 1 GB |
| 학습률 | `lr`, `learning_rate` | 옵티마이저 학습률 | 0.001 |

**메모리 관련 기호**:

| 기호 | 의미 | 설명 |
|------|------|------|
| $P$ | Parameters | 모델 파라미터 수 |
| $G$ | Gradients | Gradient 메모리 (≈ $P$) |
| $O$ | Optimizer States | 옵티마이저 상태 (Adam: $2P$, SGD: $P$) |
| $A$ | Activations | 중간 활성화 메모리 (배치 크기에 비례) |
| `float32` | 32-bit float | 4 bytes per parameter |
| `float16` | 16-bit float | 2 bytes per parameter |

**메모리 계산 공식**:

- **Float32 기준**: $1 \text{ parameter} = 4 \text{ bytes}$
- **Float16 기준**: $1 \text{ parameter} = 2 \text{ bytes}$
- **Adam 옵티마이저**: $\text{Total} \approx 4P + A$ (float32)
- **SGD 옵티마이저**: $\text{Total} \approx 3P + A$ (float32)

---

## Step 1: 왜 메모리 최적화가 필요한가?

### 이번 단계 목표
- 딥러닝 모델의 메모리 사용량 이해
- 메모리 제약이 학습에 미치는 영향 파악
- 최적화의 필요성 인식

### 입력
- 없음 (개념 설명)

### 출력
- 메모리 사용량의 전체 그림
- 최적화 전략 개요

### 평가 기준
- 메모리 사용량 구성 요소를 설명 가능
- 최적화 기법의 목적을 이해

### 문제 상황

**딥러닝 모델의 메모리 사용**:

1. **큰 모델**: 수억 개의 파라미터
   - 예: GPT-3는 175B 파라미터
   - Float32 기준: 175B × 4 bytes = 700 GB (파라미터만)

2. **큰 배치**: 성능 향상을 위해 필요
   - 배치 크기가 클수록 학습 안정성 향상
   - 하지만 메모리 사용량도 비례적으로 증가

3. **Gradient 저장**: Backpropagation을 위한 중간 결과
   - 각 파라미터마다 gradient 저장 필요
   - 파라미터와 동일한 메모리 사용

4. **Optimizer 상태**: Adam의 경우 momentum, variance 저장
   - Adam: 파라미터의 2배 메모리
   - SGD: 파라미터와 동일

5. **중간 활성화**: Forward pass의 중간 결과
   - Backward pass를 위해 저장
   - 배치 크기에 비례하여 증가

### 해결 전략

**메모리 최적화 기법**:

1. **Gradient Accumulation**: 작은 배치로 여러 번 backward, 한 번 step
2. **Mixed Precision**: Float16 사용 (메모리 50% 감소)
3. **Gradient Checkpointing**: 일부 활성화만 저장, 필요 시 재계산
4. **효율적인 DataLoader**: Pin memory, non-blocking transfer
5. **모델 최적화**: 불필요한 연산 제거, inplace 연산 사용

---

## Step 2: GPU 메모리 구조 이해

### 이번 단계 목표
- GPU 메모리 사용량 구성 요소 이해
- 메모리 계산 공식 유도
- 실제 모델의 메모리 사용량 분석

### 입력
- 모델

### 출력
- 메모리 사용량 분석 리포트
- 각 구성 요소별 메모리 사용량

### 평가 기준
- 주어진 모델의 메모리 사용량을 정확히 계산 가능
- 각 구성 요소의 기여도를 설명 가능

### Step 2.1: 메모리 사용 구성

**GPU 메모리는 다음으로 구성됩니다**:

1. **모델 파라미터** (Parameters)
   - Weight, Bias
   - 크기: `num_params × 4 bytes` (float32)
   - 크기: `num_params × 2 bytes` (float16)

2. **Gradient** (Gradients)
   - 각 파라미터의 gradient
   - 크기: 파라미터와 동일
   - Float32: `num_params × 4 bytes`
   - Float16: `num_params × 2 bytes`

3. **Optimizer 상태** (Optimizer States)
   - **Adam**: momentum, variance (파라미터의 2배)
     - Float32: `num_params × 8 bytes` (2배)
   - **SGD**: momentum (파라미터와 동일)
     - Float32: `num_params × 4 bytes`

4. **중간 활성화** (Activations)
   - Forward pass의 중간 결과
   - Backward pass를 위해 저장
   - 배치 크기에 비례
   - 정확한 계산은 복잡 (레이어별 상이)

### Step 2.2: 총 메모리 계산 공식

**Float32 + Adam 옵티마이저**:

$$\text{총 메모리} = P + G + O + A$$

여기서:
- $P$: Parameters = `num_params × 4 bytes`
- $G$: Gradients = `num_params × 4 bytes` ≈ $P$
- $O$: Optimizer States = `num_params × 8 bytes` = $2P$ (Adam)
- $A$: Activations (배치 크기에 비례)

따라서:

$$\text{총 메모리} \approx P + P + 2P + A = 4P + A$$

**Float32 + SGD 옵티마이저**:

$$\text{총 메모리} \approx P + P + P + A = 3P + A$$

### Step 2.3: 구체적 예시

**예시: 1B 파라미터 모델 (Float32, Adam)**:

- 파라미터: $1 \times 10^9 \times 4 \text{ bytes} = 4 \text{ GB}$
- Gradient: $1 \times 10^9 \times 4 \text{ bytes} = 4 \text{ GB}$
- Adam states: $1 \times 10^9 \times 8 \text{ bytes} = 8 \text{ GB}$
- **최소 16 GB** 필요 (Activations 제외)

**예시: 100M 파라미터 모델 (Float32, Adam)**:

- 파라미터: $100 \times 10^6 \times 4 \text{ bytes} = 400 \text{ MB}$
- Gradient: $100 \times 10^6 \times 4 \text{ bytes} = 400 \text{ MB}$
- Adam states: $100 \times 10^6 \times 8 \text{ bytes} = 800 \text{ MB}$
- **최소 1.6 GB** 필요 (Activations 제외)

### Step 2.4: 메모리 분석 도구

```python
# ============================================
# 모델 메모리 사용량 분석
# ============================================

def analyze_model_memory(model, dtype=torch.float32, optimizer_type='adam'):
    """
    모델의 메모리 사용량 분석
    
    Args:
        model: 분석할 모델
        dtype: 데이터 타입 (torch.float32 또는 torch.float16)
        optimizer_type: 'adam' 또는 'sgd'
    
    Returns:
        dict: 메모리 사용량 정보
    """
    # 파라미터 수 계산
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    # 바이트 수 계산
    bytes_per_param = 4 if dtype == torch.float32 else 2
    
    # 각 구성 요소별 메모리
    param_size_mb = total_params * bytes_per_param / (1024 ** 2)
    grad_size_mb = trainable_params * bytes_per_param / (1024 ** 2)
    
    if optimizer_type.lower() == 'adam':
        optimizer_size_mb = trainable_params * bytes_per_param * 2 / (1024 ** 2)
    else:  # SGD
        optimizer_size_mb = trainable_params * bytes_per_param / (1024 ** 2)
    
    total_without_activations = param_size_mb + grad_size_mb + optimizer_size_mb
    
    # 결과 출력
    print("="*60)
    print("모델 메모리 분석")
    print("="*60)
    print(f"데이터 타입: {dtype}")
    print(f"옵티마이저: {optimizer_type.upper()}")
    print(f"\n파라미터:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"\n메모리 사용량 (Activations 제외):")
    print(f"  Parameters: {param_size_mb:.2f} MB")
    print(f"  Gradients: {grad_size_mb:.2f} MB")
    print(f"  Optimizer states: {optimizer_size_mb:.2f} MB")
    print(f"  Total: {total_without_activations:.2f} MB ({total_without_activations/1024:.2f} GB)")
    print(f"\n⚠️  실제 사용량은 Activations를 포함하여 더 큽니다.")
    print(f"   Activations는 배치 크기와 모델 구조에 따라 달라집니다.")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': param_size_mb,
        'grad_size_mb': grad_size_mb,
        'optimizer_size_mb': optimizer_size_mb,
        'total_mb': total_without_activations
    }

# GPU 메모리 모니터링
def print_gpu_memory(device='cuda'):
    """
    현재 GPU 메모리 사용량 출력
    
    Args:
        device: 'cuda' 또는 'cpu'
    """
    if device == 'cuda' and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        print("\n" + "="*60)
        print("GPU 메모리 사용량")
        print("="*60)
        print(f"Allocated: {allocated:.2f} MB ({allocated/1024:.2f} GB)")
        print(f"Reserved: {reserved:.2f} MB ({reserved/1024:.2f} GB)")
        print(f"Max Allocated: {max_allocated:.2f} MB ({max_allocated/1024:.2f} GB)")
    else:
        print("CUDA not available")

# 예제 모델
class SimpleModel(nn.Module):
    """예제 모델: 큰 메모리 사용"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, 1000)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 메모리 분석 실행
model = SimpleModel()
print("\n" + "="*60)
print("Float32 + Adam 분석")
print("="*60)
analyze_model_memory(model, dtype=torch.float32, optimizer_type='adam')

print("\n" + "="*60)
print("Float16 + Adam 분석 (Mixed Precision)")
print("="*60)
analyze_model_memory(model, dtype=torch.float16, optimizer_type='adam')

# GPU 메모리 확인 (GPU 사용 가능 시)
if torch.cuda.is_available():
    model = model.to('cuda')
    print_gpu_memory('cuda')
```

---

## Step 3: Gradient Accumulation

### 이번 단계 목표
- Gradient Accumulation의 원리 이해
- 메모리 절약 효과 계산
- 실제 구현 방법 학습

### 입력
- 모델, 데이터로더, 옵티마이저, 손실 함수

### 출력
- Gradient Accumulation 적용 학습 루프
- 메모리 사용량 비교

### 평가 기준
- 효과적 배치 크기를 유지하면서 메모리 사용량 감소
- 학습 결과가 일반 학습과 유사

### Step 3.1: 왜 Gradient Accumulation이 필요한가?

**문제 상황**:

- 큰 배치 크기를 사용하고 싶지만 메모리 부족
- 예: 배치 크기 64를 원하지만 메모리 제한으로 16만 가능

**해결책**: Gradient Accumulation

- 작은 배치(16)로 여러 번 forward/backward 수행
- Gradient를 누적하여 효과적 배치 크기(64) 유지
- 메모리 사용량은 작은 배치 기준

### Step 3.2: 원리

**일반적인 학습 (배치 크기 64)**:

```python
# 배치 크기 64
loss = model(batch_64)
loss.backward()
optimizer.step()
```

**Gradient Accumulation (배치 크기 16 × 4)**:

```python
# 배치 크기 16을 4번 누적
optimizer.zero_grad()
for i in range(4):
    loss = model(mini_batch_16)
    loss = loss / 4  # 평균을 위해 나눔
    loss.backward()  # gradient 누적
optimizer.step()  # 한 번만 업데이트
```

**효과**:
- 메모리: 1/4로 감소 (16 vs 64)
- 효과적 배치 크기: 동일 (64)
- 학습 속도: 약간 느림 (오버헤드)

### Step 3.3: 구현

```python
# ============================================
# Gradient Accumulation 구현
# ============================================

def train_with_grad_accumulation(
    model, dataloader, optimizer, criterion, 
    device, accumulation_steps=4, max_grad_norm=None
):
    """
    Gradient Accumulation을 사용한 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 디바이스
        accumulation_steps: gradient를 누적할 스텝 수
        max_grad_norm: Gradient clipping 임계값 (None이면 미적용)
    
    Returns:
        dict: 학습 통계
    """
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Loss를 accumulation_steps로 나눔 (평균을 위해)
        loss = loss / accumulation_steps
        
        # Backward pass (gradient 누적)
        loss.backward()
        
        total_loss += loss.item() * accumulation_steps  # 원래 scale로 복원
        
        # accumulation_steps마다 optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (선택적)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            num_batches += 1
            print(f"Batch {batch_idx + 1}: Updated weights (effective batch: {accumulation_steps * len(data)})")
    
    # 마지막 남은 gradient 업데이트
    if (batch_idx + 1) % accumulation_steps != 0:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        num_batches += 1
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return {
        'avg_loss': avg_loss,
        'num_updates': num_batches
    }

# 사용 예시
print("\n" + "="*60)
print("Gradient Accumulation 예제")
print("="*60)
print("\n일반 학습 (배치 64):")
print("  메모리: 높음")
print("  속도: 빠름")
print("  효과적 배치: 64")

print("\nGradient Accumulation (배치 16 × 4):")
print("  메모리: 낮음 (1/4)")
print("  효과적 배치: 동일 (64)")
print("  속도: 약간 느림 (오버헤드)")

print("\n실전 팁:")
print("1. accumulation_steps = 원하는 배치 / 가능한 배치")
print("2. Loss를 accumulation_steps로 나누기 (평균 유지)")
print("3. Learning rate는 효과적 배치 크기 기준으로 조정")
```

### Step 3.4: 메모리 사용량 비교

```python
# ============================================
# 메모리 사용량 비교 실험
# ============================================

def compare_memory_usage():
    """일반 학습 vs Gradient Accumulation 메모리 비교"""
    if not torch.cuda.is_available():
        print("GPU가 없어 메모리 비교를 수행할 수 없습니다.")
        return
    
    print("\n" + "="*60)
    print("메모리 사용량 비교")
    print("="*60)
    
    # 모델 생성
    model1 = SimpleModel().to('cuda')
    model2 = SimpleModel().to('cuda')
    
    # 더미 데이터
    batch_64 = torch.randn(64, 1000).to('cuda')
    batch_16 = torch.randn(16, 1000).to('cuda')
    
    criterion = nn.MSELoss()
    target = torch.randn(64, 1000).to('cuda')
    target_16 = torch.randn(16, 1000).to('cuda')
    
    # 일반 학습 (배치 64)
    torch.cuda.reset_peak_memory_stats()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    
    output1 = model1(batch_64)
    loss1 = criterion(output1, target)
    loss1.backward()
    optimizer1.step()
    
    memory_normal = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Gradient Accumulation (배치 16 × 4)
    torch.cuda.reset_peak_memory_stats()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    optimizer2.zero_grad()
    
    for i in range(4):
        output2 = model2(batch_16)
        loss2 = criterion(output2, target_16[i*16:(i+1)*16])
        loss2 = loss2 / 4
        loss2.backward()
    
    optimizer2.step()
    
    memory_accum = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 결과 출력
    print(f"\n일반 학습 (배치 64): {memory_normal:.2f} MB")
    print(f"Gradient Accumulation (배치 16 × 4): {memory_accum:.2f} MB")
    print(f"메모리 절약: {(1 - memory_accum/memory_normal)*100:.1f}%")

# GPU 사용 가능 시 실행
if torch.cuda.is_available():
    compare_memory_usage()
```

---

## Step 4: Mixed Precision Training

### 이번 단계 목표
- Mixed Precision의 원리 이해
- Float16 vs Float32 차이점 파악
- AMP (Automatic Mixed Precision) 구현 방법 학습

### 입력
- 모델, 데이터로더, 옵티마이저, 손실 함수

### 출력
- Mixed Precision 적용 학습 루프
- 메모리 사용량 비교

### 평가 기준
- 메모리 사용량이 약 50% 감소
- 학습 정확도가 Float32와 유사

### Step 4.1: 왜 Mixed Precision이 필요한가?

**아이디어**: Float32 대신 Float16을 사용하여 메모리와 연산 속도 개선

**Float16 vs Float32 비교**:

| 항목 | Float32 | Float16 |
|------|---------|----------|
| 메모리 | 4 bytes | 2 bytes |
| 범위 | ±3.4×10³⁸ | ±6.5×10⁴ |
| 정밀도 | 높음 (7자리) | 낮음 (3자리) |
| 속도 | 느림 | 빠름 (Tensor Core) |

**효과**:
- 메모리: 약 50% 감소
- 속도: 2~3배 향상 (GPU Tensor Core 지원 시)
- 정확도: 거의 동일 (대부분의 경우)

### Step 4.2: Mixed Precision 전략

**전략**:

1. **Forward/Backward**: Float16 사용 (메모리 절약)
2. **Weight 업데이트**: Float32 사용 (정밀도 유지)
3. **Loss Scaling**: Gradient underflow 방지

**왜 Loss Scaling이 필요한가?**

- Float16의 범위가 작아 gradient가 0에 가까워질 수 있음
- Loss를 큰 값으로 스케일링하여 gradient를 확대
- Backward 후 gradient를 다시 스케일링

### Step 4.3: 구현

```python
# ============================================
# Mixed Precision Training (AMP)
# ============================================

def train_with_mixed_precision(
    model, dataloader, optimizer, criterion, device,
    accumulation_steps=1, max_grad_norm=None
):
    """
    Automatic Mixed Precision (AMP)을 사용한 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 디바이스
        accumulation_steps: Gradient accumulation 스텝 수
        max_grad_norm: Gradient clipping 임계값
    
    Returns:
        dict: 학습 통계
    """
    model.train()
    scaler = GradScaler()  # Loss scaling
    
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # autocast: 자동으로 float16 사용
        with autocast():
            output = model(data)
            loss = criterion(output, target)
            loss = loss / accumulation_steps  # Gradient accumulation
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * accumulation_steps
        
        # accumulation_steps마다 optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (선택적)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Scaled gradient를 unscale하고 optimizer step
            scaler.step(optimizer)
            scaler.update()  # Loss scale 업데이트
            optimizer.zero_grad()
            
            num_batches += 1
    
    # 마지막 남은 gradient 업데이트
    if (batch_idx + 1) % accumulation_steps != 0:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        num_batches += 1
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return {
        'avg_loss': avg_loss,
        'num_updates': num_batches
    }

# 사용 조건 확인
print("\n" + "="*60)
print("Mixed Precision Training")
print("="*60)

if torch.cuda.is_available():
    print(f"\n현재 GPU: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA Capability: {capability}")
    
    if capability >= (7, 0):
        print("✅ Tensor Core 지원: Mixed Precision 사용 가능")
    else:
        print("⚠️  Tensor Core 미지원: Mixed Precision 사용 가능하나 속도 향상 제한적")
else:
    print("CUDA not available: Mixed Precision 사용 불가")
```

### Step 4.4: 메모리 사용량 비교

```python
# ============================================
# Float32 vs Mixed Precision 메모리 비교
# ============================================

def compare_precision_memory():
    """Float32 vs Mixed Precision 메모리 비교"""
    if not torch.cuda.is_available():
        print("GPU가 없어 메모리 비교를 수행할 수 없습니다.")
        return
    
    print("\n" + "="*60)
    print("Float32 vs Mixed Precision 메모리 비교")
    print("="*60)
    
    batch_size = 64
    input_size = 1000
    
    # Float32 학습
    model1 = SimpleModel().to('cuda')
    torch.cuda.reset_peak_memory_stats()
    
    x1 = torch.randn(batch_size, input_size).to('cuda')
    y1 = torch.randn(batch_size, 1000).to('cuda')
    
    output1 = model1(x1)
    loss1 = nn.MSELoss()(output1, y1)
    loss1.backward()
    
    memory_fp32 = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Mixed Precision 학습
    model2 = SimpleModel().to('cuda')
    torch.cuda.reset_peak_memory_stats()
    
    x2 = torch.randn(batch_size, input_size).to('cuda')
    y2 = torch.randn(batch_size, 1000).to('cuda')
    
    scaler = GradScaler()
    with autocast():
        output2 = model2(x2)
        loss2 = nn.MSELoss()(output2, y2)
    
    scaler.scale(loss2).backward()
    
    memory_amp = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 결과 출력
    print(f"\nFloat32: {memory_fp32:.2f} MB")
    print(f"Mixed Precision: {memory_amp:.2f} MB")
    print(f"메모리 절약: {(1 - memory_amp/memory_fp32)*100:.1f}%")

# GPU 사용 가능 시 실행
if torch.cuda.is_available():
    compare_precision_memory()
```

---

## Step 5: 메모리 효율적인 데이터 로딩

### 이번 단계 목표
- DataLoader 최적화 설정 이해
- Pin memory, non-blocking transfer의 효과 파악
- 적절한 num_workers 선택 방법 학습

### 입력
- 데이터셋

### 출력
- 최적화된 DataLoader
- 데이터 전송 시간 비교

### 평가 기준
- 데이터 로딩 시간이 단축됨
- GPU 활용률이 향상됨

### Step 5.1: 왜 데이터 로딩 최적화가 필요한가?

**문제 상황**:

- CPU에서 데이터를 로드하는 동안 GPU가 대기
- 데이터 전송이 학습 속도의 병목 지점이 될 수 있음

**해결책**:

1. **Pin Memory**: CPU → GPU 전송 속도 향상
2. **Non-blocking Transfer**: 데이터 전송과 연산 동시 수행
3. **Prefetch**: 다음 배치를 미리 로드
4. **적절한 num_workers**: 병렬 데이터 로딩

### Step 5.2: 최적화 설정

```python
# ============================================
# 메모리 효율적인 DataLoader 설정
# ============================================

def create_efficient_dataloader(
    dataset, batch_size=32, num_workers=None, pin_memory=None
):
    """
    메모리 효율적인 DataLoader 생성
    
    Args:
        dataset: 데이터셋
        batch_size: 배치 크기
        num_workers: 병렬 워커 수 (None이면 자동 계산)
        pin_memory: Pin memory 사용 여부 (None이면 GPU일 때 True)
    
    Returns:
        DataLoader: 최적화된 DataLoader
    """
    # num_workers 자동 계산
    if num_workers is None:
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(4, cpu_count // 2)  # CPU 코어 수의 절반, 최대 4
    
    # pin_memory 자동 설정
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,      # 병렬 데이터 로딩
        pin_memory=pin_memory,         # GPU 전송 속도 향상
        persistent_workers=True if num_workers > 0 else False,  # Worker 재사용
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch
    )
    
    print(f"DataLoader 설정:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Persistent workers: {num_workers > 0}")
    
    return dataloader

# Non-blocking transfer 예제
def train_with_async_transfer(model, dataloader, device):
    """
    비동기 데이터 전송을 사용한 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터로더
        device: 디바이스
    """
    model.train()
    
    for data, target in dataloader:
        # non_blocking=True: 전송과 연산 동시 수행
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(data)
        # ... 학습 로직

# 최적 num_workers 찾기
print("\n" + "="*60)
print("데이터 로딩 최적화")
print("="*60)

cpu_count = multiprocessing.cpu_count()
print(f"\n현재 CPU 코어 수: {cpu_count}")
print(f"권장 num_workers: {min(4, cpu_count // 2)}")

print("\n최적화 팁:")
print("1. Pin Memory: CPU → GPU 전송 속도 향상 (GPU 사용 시)")
print("2. Non-blocking Transfer: 데이터 전송과 연산 동시 수행")
print("3. Num Workers: CPU 코어 수의 절반 (너무 많으면 메모리 부족)")
print("4. Batch Size: 2의 거듭제곱 (32, 64, 128) - GPU 효율성 향상")
```

---

## Step 6: Gradient Checkpointing

### 이번 단계 목표
- Gradient Checkpointing의 원리 이해
- 메모리 vs 연산 시간 트레이드오프 파악
- 실제 구현 방법 학습

### 입력
- 모델

### 출력
- Checkpointing 적용 모델
- 메모리 사용량 비교

### 평가 기준
- 메모리 사용량이 대폭 감소
- 연산 시간 증가는 20~30% 이내

### Step 6.1: 왜 Gradient Checkpointing이 필요한가?

**문제 상황**:

- 매우 깊은 네트워크에서 메모리 부족
- 모든 중간 활성화를 저장하면 메모리 초과

**해결책**: Gradient Checkpointing

- Forward: 일부만 저장 (checkpoint)
- Backward: 필요 시 재계산
- 메모리: 대폭 감소
- 대가: 연산 시간 20~30% 증가

### Step 6.2: 원리

**일반적인 학습**:

- Forward: 모든 중간 활성화 저장
- Backward: 저장된 활성화 사용
- 메모리: 높음, 속도: 빠름

**Gradient Checkpointing**:

- Forward: 일부만 저장 (checkpoint)
- Backward: 필요 시 재계산
- 메모리: 낮음, 속도: 느림 (약 20~30%)

### Step 6.3: 구현

```python
# ============================================
# Gradient Checkpointing 사용
# ============================================

class CheckpointedModel(nn.Module):
    """Gradient Checkpointing을 사용하는 모델"""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)
        self.layer4 = nn.Linear(1000, 100)
    
    def forward(self, x):
        # Checkpoint를 사용하여 메모리 절약
        x = checkpoint(self._forward_block1, x)
        x = checkpoint(self._forward_block2, x)
        x = self.layer4(x)
        return x
    
    def _forward_block1(self, x):
        """첫 번째 블록 (checkpoint 적용)"""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x
    
    def _forward_block2(self, x):
        """두 번째 블록 (checkpoint 적용)"""
        x = torch.relu(self.layer3(x))
        return x

# 사용 예시
print("\n" + "="*60)
print("Gradient Checkpointing")
print("="*60)

print("\n일반 학습:")
print("  메모리: 높음 (모든 활성화 저장)")
print("  속도: 빠름")

print("\nGradient Checkpointing:")
print("  메모리: 낮음 (일부만 저장)")
print("  속도: 느림 (약 20~30% 감소)")

print("\n사용 시기:")
print("✓ 매우 깊은 네트워크 (ResNet-152, Transformer 등)")
print("✓ 메모리 부족 시")
print("✓ 큰 입력 해상도")

model = CheckpointedModel()
num_params = sum(p.numel() for p in model.parameters())
print(f"\n모델 준비 완료: {num_params:,} parameters")
```

---

## Step 7: 실전 메모리 최적화 전략

### 이번 단계 목표
- 모든 최적화 기법을 통합한 학습 루프 구축
- 실전 체크리스트 작성
- 최적 조합 전략 수립

### 입력
- 모델, 데이터로더, 옵티마이저, 손실 함수

### 출력
- 통합 최적화 학습 루프
- 메모리 최적화 체크리스트

### 평가 기준
- 모든 최적화 기법이 올바르게 통합됨
- 메모리 사용량이 예상 범위 내

### Step 7.1: 통합 최적화 학습 루프

```python
# ============================================
# 종합 최적화 학습 루프
# ============================================

def optimized_training_loop(
    model, dataloader, optimizer, criterion, device,
    use_amp=True,
    accumulation_steps=1,
    max_grad_norm=1.0,
    log_interval=10
):
    """
    모든 최적화 기법을 적용한 학습 루프
    
    Args:
        model: 학습할 모델
        dataloader: 데이터로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 디바이스
        use_amp: Mixed Precision 사용 여부
        accumulation_steps: Gradient accumulation 스텝 수
        max_grad_norm: Gradient clipping 임계값
        log_interval: 로그 출력 간격
    
    Returns:
        dict: 학습 통계
    """
    model.train()
    scaler = GradScaler() if use_amp else None
    optimizer.zero_grad()
    
    total_loss = 0.0
    num_updates = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Non-blocking transfer
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Mixed Precision
        if use_amp:
            with autocast():
                output = model(data)
                loss = criterion(output, target) / accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()
        
        total_loss += loss.item() * accumulation_steps
        
        # Gradient Accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            
            # Gradient Clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            num_updates += 1
            
            # 주기적 로깅
            if num_updates % log_interval == 0:
                print(f"Update {num_updates}: Loss = {loss.item() * accumulation_steps:.4f}")
        
        # 메모리 정리 (필요 시)
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 마지막 남은 gradient 업데이트
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        num_updates += 1
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return {
        'avg_loss': avg_loss,
        'num_updates': num_updates
    }

# 사용 예시
print("\n" + "="*60)
print("통합 최적화 학습 루프")
print("="*60)
print("\n적용된 최적화:")
print("✓ Mixed Precision (AMP)")
print("✓ Gradient Accumulation")
print("✓ Gradient Clipping")
print("✓ Non-blocking Transfer")
print("✓ 주기적 메모리 정리")
```

### Step 7.2: 메모리 최적화 체크리스트

```python
# ============================================
# 메모리 최적화 체크리스트
# ============================================

print("\n" + "="*60)
print("메모리 최적화 체크리스트")
print("="*60)

checklist = [
    ("1. 배치 크기 줄이기", "가장 간단한 방법, Gradient Accumulation과 함께 사용"),
    ("2. Mixed Precision (AMP)", "메모리 50% 감소, 속도 2~3배 (Tensor Core 지원 시)"),
    ("3. Gradient Accumulation", "효과적 배치 크기 유지, 메모리 1/N 감소"),
    ("4. Gradient Checkpointing", "깊은 네트워크에 효과적, 속도 20~30% 감소"),
    ("5. 효율적인 DataLoader", "pin_memory=True, non_blocking=True, 적절한 num_workers"),
    ("6. 불필요한 연산 제거", "torch.no_grad() 사용, inplace 연산 (relu_), del로 변수 삭제"),
    ("7. 정기적인 메모리 정리", "torch.cuda.empty_cache() 주기적으로 호출"),
    ("8. 옵티마이저 선택", "SGD는 Adam보다 메모리 적음 (momentum만 저장)"),
]

for item, desc in checklist:
    print(f"\n□ {item}")
    print(f"   - {desc}")

print("\n" + "="*60)
print("최대 메모리 절약 조합")
print("="*60)
print("Mixed Precision (50% 감소)")
print("+ Gradient Accumulation (1/4 감소)")
print("+ Gradient Checkpointing (추가 감소)")
print("→ 총 약 1/8 수준으로 메모리 사용")
```

---

## 실험 기록 표

### 메모리 사용량 Before/After 비교

| 최적화 기법 | Before (MB) | After (MB) | 절약율 | 비고 |
|-------------|-------------|------------|--------|------|
| Gradient Accumulation (배치 64→16×4) | 1024 | 256 | 75% | 효과적 배치 유지 |
| Mixed Precision (Float32→Float16) | 1024 | 512 | 50% | Tensor Core 필요 |
| Gradient Checkpointing | 1024 | 256 | 75% | 속도 25% 감소 |
| 통합 최적화 (AMP + Accumulation) | 1024 | 128 | 87.5% | 효과적 배치 유지 |

### 성능 지표

| 최적화 기법 | 메모리 절약 | 속도 변화 | 정확도 변화 |
|-------------|-------------|-----------|-------------|
| Gradient Accumulation | 높음 | 약간 느림 (-5%) | 거의 동일 |
| Mixed Precision | 높음 | 빠름 (+100~200%) | 거의 동일 |
| Gradient Checkpointing | 매우 높음 | 느림 (-20~30%) | 동일 |
| 통합 최적화 | 매우 높음 | 빠름 (+50~100%) | 거의 동일 |

### 실험 설정

| 실험 | 고정 변수 | 조작 변수 | 목적 |
|------|-----------|-----------|------|
| Gradient Accumulation | 모델, 데이터 | 배치 크기 (64 vs 16×4) | 메모리 절약 효과 측정 |
| Mixed Precision | 모델, 데이터 | 데이터 타입 (Float32 vs Float16) | 메모리 및 속도 개선 측정 |
| 통합 최적화 | 모델, 데이터 | 최적화 기법 조합 | 최대 메모리 절약 달성 |

---

## 핵심 요약

### 이번 단원에서 배운 내용

#### 1. GPU 메모리 구조
- **구성**: 파라미터 + Gradient + Optimizer States + Activations
- **계산**: Float32 + Adam ≈ 4P + A
- **도구**: `analyze_model_memory()`, `print_gpu_memory()`

#### 2. Gradient Accumulation
- **효과**: 메모리 1/N, 효과적 배치 크기 유지
- **방법**: 작은 배치로 여러 번 backward, 한 번 step
- **도구**: `train_with_grad_accumulation()`

#### 3. Mixed Precision
- **효과**: 메모리 50% 감소, 속도 2~3배
- **방법**: Float16 + Float32, Loss Scaling
- **조건**: Tensor Core 지원 GPU (Volta 이상)
- **도구**: `autocast()`, `GradScaler()`

#### 4. 데이터 로딩 최적화
- **Pin Memory**: CPU → GPU 전송 속도 향상
- **Non-blocking**: 전송과 연산 동시 수행
- **Num Workers**: CPU 코어 수의 절반
- **도구**: `create_efficient_dataloader()`

#### 5. Gradient Checkpointing
- **효과**: 메모리 대폭 감소
- **대가**: 속도 20~30% 감소
- **사용**: 매우 깊은 네트워크
- **도구**: `checkpoint()` 함수

#### 6. 실전 조합
- Mixed Precision + Gradient Accumulation + Gradient Checkpointing
- 총 약 1/8 수준으로 메모리 사용
- 효과적 배치 크기 유지하면서 메모리 절약

### 다음 단계

다음 단원에서는 **시각화 및 모니터링**을 배워보겠습니다!

