# 13. 메모리 최적화

이번 단원에서는 **PyTorch 모델의 메모리 사용을 최적화하는 방법**을 배워보겠습니다.

## 학습 목표
- GPU 메모리 구조 이해
- Gradient Accumulation
- Mixed Precision Training
- 메모리 효율적인 데이터 로딩
- Checkpoint와 메모리 트레이드오프

## 왜 이 단원이 필요한가?

딥러닝 모델은 **메모리를 많이 사용**합니다:

- **큰 모델**: 수억 개의 파라미터
- **큰 배치**: 성능 향상을 위해 필요
- **Gradient 저장**: Backpropagation을 위한 중간 결과
- **제한된 GPU 메모리**: 보통 8~24GB

백엔드 개발자라면 **메모리 관리, 캐싱, 최적화**에 익숙할 것입니다. 이 경험을 딥러닝에 적용해봅시다.

---
# 1. GPU 메모리 구조
---

## 1.1 메모리 사용 구성

GPU 메모리는 다음으로 구성됩니다:

1. **모델 파라미터** (Parameters)
   - Weight, Bias
   - 크기: `num_params * 4 bytes` (float32)

2. **Gradient** (Gradients)
   - 각 파라미터의 gradient
   - 크기: 파라미터와 동일

3. **Optimizer 상태** (Optimizer States)
   - Adam: momentum, variance (파라미터의 2배)
   - SGD: momentum (파라미터와 동일)

4. **중간 활성화** (Activations)
   - Forward pass의 중간 결과
   - Backward pass를 위해 저장
   - 배치 크기에 비례

### 총 메모리 계산

```
총 메모리 ≈ 파라미터 + Gradient + Optimizer States + Activations
         ≈ P + P + 2P + Activations  (Adam 사용 시)
         ≈ 4P + Activations
```

예: 1B 파라미터 모델 (float32)
- 파라미터: 4GB
- Gradient: 4GB
- Adam states: 8GB
- **최소 16GB** 필요 (Activations 제외)

```python
import torch
import torch.nn as nn

# 메모리 사용량 분석
def analyze_model_memory(model):
    """모델의 메모리 사용량 분석"""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    # Float32 기준 (4 bytes)
    param_size_mb = total_params * 4 / (1024 ** 2)
    
    print(f"=== 모델 메모리 분석 ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nMemory (float32):")
    print(f"  Parameters: {param_size_mb:.2f} MB")
    print(f"  Gradients: {param_size_mb:.2f} MB")
    print(f"  Adam states: {param_size_mb * 2:.2f} MB")
    print(f"  Total (without activations): {param_size_mb * 4:.2f} MB")
    
    return param_size_mb

# GPU 메모리 모니터링
def print_gpu_memory():
    """현재 GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"\n=== GPU 메모리 ===")
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")
        print(f"Max Allocated: {max_allocated:.2f} MB")
    else:
        print("CUDA not available")

# 예제 모델
class SimpleModel(nn.Module):
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

model = SimpleModel()
analyze_model_memory(model)
print_gpu_memory()
```

---
# 2. Gradient Accumulation
---

## 2.1 개념

**문제**: 큰 배치 크기를 사용하고 싶지만 메모리 부족

**해결**: 작은 배치로 여러 번 forward/backward를 수행하고 gradient를 누적

### 원리

```python
# 일반적인 학습 (배치 크기 64)
loss = model(batch_64)
loss.backward()
optimizer.step()

# Gradient Accumulation (배치 크기 16 × 4)
for mini_batch in split_into_4(batch_64):
    loss = model(mini_batch_16)
    loss.backward()  # gradient 누적
optimizer.step()  # 한 번만 업데이트
```

### 효과
- 메모리: 1/4로 감소
- 효과적 배치 크기: 동일 (64)
- 학습 속도: 약간 느림 (오버헤드)

```python
import torch
import torch.nn as nn

# Gradient Accumulation 구현
def train_with_grad_accumulation(
    model, dataloader, optimizer, criterion, 
    device, accumulation_steps=4
):
    """
    Gradient Accumulation을 사용한 학습
    
    Args:
        accumulation_steps: gradient를 누적할 스텝 수
    """
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Loss를 accumulation_steps로 나눔 (평균을 위해)
        loss = loss / accumulation_steps
        
        # Backward pass (gradient 누적)
        loss.backward()
        
        # accumulation_steps마다 optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Batch {batch_idx + 1}: Updated weights")
    
    # 마지막 남은 gradient 업데이트
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

# 비교: 일반 학습 vs Gradient Accumulation
print("=== Gradient Accumulation ===")
print("일반 학습 (배치 64):")
print("  메모리: 높음")
print("  속도: 빠름")
print("\nGradient Accumulation (배치 16 × 4):")
print("  메모리: 낮음 (1/4)")
print("  효과적 배치: 동일 (64)")
print("  속도: 약간 느림")

# 실전 팁
print("\n실전 팁:")
print("1. accumulation_steps = 원하는 배치 / 가능한 배치")
print("2. Loss를 accumulation_steps로 나누기")
print("3. Learning rate는 효과적 배치 크기 기준으로 조정")
```

---
# 3. Mixed Precision Training
---

## 3.1 개념

**아이디어**: Float32 대신 Float16을 사용하여 메모리와 연산 속도 개선

### Float16 vs Float32

| 항목 | Float32 | Float16 |
|------|---------|----------|
| 메모리 | 4 bytes | 2 bytes |
| 범위 | ±3.4×10³⁸ | ±6.5×10⁴ |
| 정밀도 | 높음 | 낮음 |
| 속도 | 느림 | 빠름 (Tensor Core) |

### Mixed Precision의 전략

1. **Forward/Backward**: Float16
2. **Weight 업데이트**: Float32
3. **Loss Scaling**: Gradient underflow 방지

### 효과
- 메모리: 약 50% 감소
- 속도: 2~3배 향상 (GPU 의존)
- 정확도: 거의 동일

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Mixed Precision Training
def train_with_mixed_precision(
    model, dataloader, optimizer, criterion, device
):
    """
    Automatic Mixed Precision (AMP)을 사용한 학습
    """
    model.train()
    scaler = GradScaler()  # Loss scaling
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # autocast: 자동으로 float16 사용
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        # Scaled gradient를 unscale하고 optimizer step
        scaler.step(optimizer)
        scaler.update()

# 비교 예제
print("=== Mixed Precision Training ===")
print("\n일반 학습 (Float32):")
print("  메모리: 높음")
print("  속도: 느림")
print("  정밀도: 높음")

print("\nMixed Precision (Float16 + Float32):")
print("  메모리: 약 50% 감소")
print("  속도: 2~3배 향상")
print("  정밀도: 거의 동일")

# 사용 조건
print("\n사용 조건:")
print("✓ GPU: Volta (V100) 이상 (Tensor Core 지원)")
print("✓ PyTorch 1.6+")
print("✓ CUDA 지원")

if torch.cuda.is_available():
    print(f"\n현재 GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed Precision 지원: {torch.cuda.get_device_capability(0) >= (7, 0)}")
else:
    print("\nCUDA not available")
```

---
# 4. 메모리 효율적인 데이터 로딩
---

## 4.1 전략

### 1. Pin Memory
- CPU → GPU 전송 속도 향상
- 약간의 CPU 메모리 사용

### 2. Non-blocking Transfer
- 데이터 전송과 연산 동시 수행

### 3. Prefetch
- 다음 배치를 미리 로드

### 4. 적절한 num_workers
- CPU 코어 수에 맞춰 조정

```python
import torch
from torch.utils.data import DataLoader, Dataset

# 메모리 효율적인 DataLoader 설정
def create_efficient_dataloader(
    dataset, batch_size=32, num_workers=4
):
    """
    메모리 효율적인 DataLoader 생성
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 병렬 데이터 로딩
        pin_memory=True,          # GPU 전송 속도 향상
        persistent_workers=True   # Worker 재사용
    )
    return dataloader

# Non-blocking transfer
def train_with_async_transfer(model, dataloader, device):
    """
    비동기 데이터 전송을 사용한 학습
    """
    model.train()
    
    for data, target in dataloader:
        # non_blocking=True: 전송과 연산 동시 수행
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(data)
        # ...

# 메모리 절약 팁
print("=== 데이터 로딩 최적화 ===")
print("\n1. Pin Memory")
print("   - CPU → GPU 전송 속도 향상")
print("   - pin_memory=True")

print("\n2. Non-blocking Transfer")
print("   - 데이터 전송과 연산 동시 수행")
print("   - .to(device, non_blocking=True)")

print("\n3. Num Workers")
print("   - 권장: CPU 코어 수의 절반")
print("   - 너무 많으면 메모리 부족")

print("\n4. Batch Size")
print("   - 2의 거듭제곱 (32, 64, 128)")
print("   - GPU 메모리에 맞춰 조정")

# 최적 num_workers 찾기
import multiprocessing
cpu_count = multiprocessing.cpu_count()
print(f"\n현재 CPU 코어 수: {cpu_count}")
print(f"권장 num_workers: {cpu_count // 2}")
```

---
# 5. Gradient Checkpointing
---

## 5.1 개념

**트레이드오프**: 메모리 vs 연산 시간

### 일반적인 학습
- Forward: 모든 중간 활성화 저장
- Backward: 저장된 활성화 사용
- 메모리: 높음, 속도: 빠름

### Gradient Checkpointing
- Forward: 일부만 저장 (checkpoint)
- Backward: 필요 시 재계산
- 메모리: 낮음, 속도: 느림 (약 20~30%)

### 사용 시기
- 매우 깊은 네트워크
- 메모리가 부족할 때
- 큰 입력 해상도

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Gradient Checkpointing 사용
class CheckpointedModel(nn.Module):
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
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x
    
    def _forward_block2(self, x):
        x = torch.relu(self.layer3(x))
        return x

print("=== Gradient Checkpointing ===")
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
print(f"\n모델 준비 완료: {sum(p.numel() for p in model.parameters()):,} parameters")
```

---
# 6. 실전 메모리 최적화 전략
---

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# 종합 최적화 학습 루프
def optimized_training_loop(
    model, dataloader, optimizer, criterion, device,
    use_amp=True,
    accumulation_steps=1,
    max_grad_norm=1.0
):
    """
    모든 최적화 기법을 적용한 학습 루프
    """
    model.train()
    scaler = GradScaler() if use_amp else None
    optimizer.zero_grad()
    
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
        
        # Gradient Accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # 메모리 정리 (필요 시)
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

# 메모리 최적화 체크리스트
print("=== 메모리 최적화 체크리스트 ===")
print("\n□ 1. 배치 크기 줄이기")
print("   - 가장 간단한 방법")
print("   - Gradient Accumulation과 함께 사용")

print("\n□ 2. Mixed Precision (AMP)")
print("   - 메모리 50% 감소, 속도 2~3배")
print("   - GPU Tensor Core 지원 필요")

print("\n□ 3. Gradient Accumulation")
print("   - 효과적 배치 크기 유지")
print("   - 메모리 1/N 감소")

print("\n□ 4. Gradient Checkpointing")
print("   - 깊은 네트워크에 효과적")
print("   - 속도 20~30% 감소")

print("\n□ 5. 효율적인 DataLoader")
print("   - pin_memory=True")
print("   - non_blocking=True")
print("   - 적절한 num_workers")

print("\n□ 6. 불필요한 연산 제거")
print("   - torch.no_grad() 사용")
print("   - inplace 연산 (relu_)")
print("   - del로 변수 삭제")

print("\n□ 7. 정기적인 메모리 정리")
print("   - torch.cuda.empty_cache()")
print("   - 주기적으로 호출")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. GPU 메모리 구조
- **구성**: 파라미터 + Gradient + Optimizer States + Activations
- **계산**: 약 4P (Adam 사용 시)

### 2. Gradient Accumulation
- **효과**: 메모리 1/N, 효과적 배치 크기 유지
- **방법**: 작은 배치로 여러 번 backward, 한 번 step

### 3. Mixed Precision
- **효과**: 메모리 50% 감소, 속도 2~3배
- **방법**: Float16 + Float32, Loss Scaling
- **조건**: Tensor Core 지원 GPU

### 4. 데이터 로딩 최적화
- **Pin Memory**: CPU → GPU 전송 속도 향상
- **Non-blocking**: 전송과 연산 동시 수행
- **Num Workers**: CPU 코어 수의 절반

### 5. Gradient Checkpointing
- **효과**: 메모리 대폭 감소
- **대가**: 속도 20~30% 감소
- **사용**: 매우 깊은 네트워크

### 실전 조합
```python
# 최대 메모리 절약
- Mixed Precision (50% 감소)
- Gradient Accumulation (1/4 감소)
- Gradient Checkpointing (추가 감소)
→ 총 약 1/8 수준으로 메모리 사용
```

다음 단원에서는 **시각화 및 모니터링**을 배워보겠습니다!
