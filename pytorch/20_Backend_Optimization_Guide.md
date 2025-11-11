# 20. 백엔드 개발자 관점 최적화 가이드

이번 단원에서는 **백엔드 시스템에 딥러닝 모델을 통합하고 운영할 때 필요한 최적화 기법**을 심도 있게 다룹니다.

## 학습 목표
- GPU 메모리 심층 분석 (단편화, 프로파일링)
- PyTorch Profiler를 활용한 병목 지점 분석
- 동적 배치 처리 (Dynamic Batching) 구현
- 추론 서버 활용 (TorchServe, Triton)
- CPU/GPU 최적 활용 전략

## 왜 이 단원이 필요한가?

모델을 API로 만드는 것과 **안정적으로 운영**하는 것은 다릅니다. 백엔드 개발자는 **지연 시간(Latency), 처리량(Throughput), 안정성(Stability), 비용(Cost)**을 모두 고려해야 합니다.

이 단원은 기존 백엔드 시스템 지식(큐, 캐싱, 비동기 처리, 리소스 관리)을 딥러닝 모델 운영에 접목하는 방법을 제시합니다.

---
# 1. GPU 메모리 심층 분석
---

```python
import torch

def detailed_memory_report(device='cuda'):
    """
    PyTorch의 상세 메모리 리포트 출력
    """ 
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
        
    print(f"=== Detailed Memory Report for {device} ===")
    
    # 현재 메모리 사용량
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB (from CUDA allocator)")
    
    # 최대 사용량
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"Peak Allocated: {max_allocated:.2f} MB")
    print(f"Peak Reserved: {max_reserved:.2f} MB")
    
    # 상세 통계 (PyTorch 1.10+)
    print("\n--- Memory Summary ---")
    try:
        print(torch.cuda.memory_summary(device))
    except Exception as e:
        print(f"Could not get memory summary: {e}")

# 메모리 단편화 시뮬레이션
def simulate_fragmentation():
    """
    메모리 단편화 시뮬레이션
    """
    if not torch.cuda.is_available(): return
    
    # 1. 작은 텐서들을 많이 할당
    tensors = [torch.randn(1024, 1024, device='cuda') for _ in range(100)]
    
    # 2. 일부를 해제 (홀수 인덱스)
    for i in range(0, 100, 2):
        tensors[i] = None
    
    # 이제 메모리에는 빈 공간이 생김
    print("After fragmentation:")
    detailed_memory_report()
    
    # 3. 큰 텐서 할당 시도
    try:
        large_tensor = torch.randn(20480, 20480, device='cuda')
    except RuntimeError as e:
        print(f"\nFailed to allocate large tensor: {e}")
        print("Reason: Memory is fragmented.")
    
    # 4. 캐시 비우기
    torch.cuda.empty_cache()
    print("\nAfter empty_cache():")
    detailed_memory_report()
    
    # 5. 재시도
    try:
        large_tensor = torch.randn(20480, 20480, device='cuda')
        print("\nSuccessfully allocated large tensor after clearing cache.")
    except RuntimeError as e:
        print(f"\nStill failed: {e}")

# 실행
simulate_fragmentation()
```

---
# 2. PyTorch Profiler 활용
---

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# 프로파일링할 모델
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc = nn.Linear(32 * 220 * 220, 10) # Simplified
    
    def forward(self, x):
        # record_function으로 특정 블록 추적
        with record_function("conv_block"):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
        
        with record_function("flatten_and_fc"):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        
        return x

model = MyModel().cuda()
inputs = torch.randn(16, 3, 224, 224).cuda()

# 프로파일러 실행
with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(10):
        model(inputs)
        prof.step() # 다음 스텝으로

print("=== Profiler Report ===")
print("Trace saved to ./log/profiler")
print("Run: tensorboard --logdir=./log/profiler")

# 콘솔 출력
print("\n--- Operator View (Top 5 by CUDA time) ---")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

print("\n--- Memory View ---")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
```

---
# 3. 동적 배치 처리 (Dynamic Batching)
---

```python
import time
import queue
import threading
import torch

class DynamicBatcher:
    """
    백엔드 서버를 위한 동적 배치 처리기
    """
    def __init__(self, model, max_batch_size=32, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = queue.Queue()
        
        # 백그라운드 스레드 시작
        self.running = True
        self.thread = threading.Thread(target=self._batch_processor)
        self.thread.start()
        
    def predict(self, data):
        """
        API 요청 처리 (동기)
        """
        # 요청 큐에 추가
        response_queue = queue.Queue()
        self.request_queue.put((data, response_queue))
        
        # 결과 대기
        return response_queue.get()
        
    def _batch_processor(self):
        """
        백그라운드에서 배치 처리
        """
        while self.running:
            batch = []
            response_queues = []
            start_time = time.time()
            
            # max_batch_size 또는 max_wait_time까지 요청 수집
            while (
                len(batch) < self.max_batch_size and
                (time.time() - start_time) < self.max_wait_time
            ):
                try:
                    # 큐에서 요청 가져오기 (non-blocking)
                    data, res_q = self.request_queue.get(block=False)
                    batch.append(data)
                    response_queues.append(res_q)
                except queue.Empty:
                    # 큐가 비어있으면 잠시 대기
                    time.sleep(0.001)
            
            # 처리할 배치가 없으면 계속
            if not batch:
                continue
            
            # 모델 추론
            batch_tensor = torch.stack(batch)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # 각 요청에 결과 반환
            for i, res_q in enumerate(response_queues):
                res_q.put(outputs[i])
    
    def stop(self):
        self.running = False
        self.thread.join()

# 사용 예시
model = nn.Linear(10, 5) # 더미 모델
batcher = DynamicBatcher(model, max_batch_size=8, max_wait_time=0.05)

def worker(data_id):
    """API 요청 시뮬레이션"""
    data = torch.randn(10)
    result = batcher.predict(data)
    print(f"Request {data_id} got result: {result.shape}")

# 여러 스레드에서 동시에 요청
client_threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
for t in client_threads:
    t.start()
for t in client_threads:
    t.join()

batcher.stop()
print("\nDynamic batching finished.")
```

---
# 4. 추론 서버 활용
---

## 4.1 TorchServe vs Triton

| 기능 | TorchServe (PyTorch) | Triton (NVIDIA) |
|---|---|---|
| 프레임워크 | PyTorch 전용 | PyTorch, TF, ONNX, TensorRT |
| 성능 | 좋음 | 매우 좋음 (GPU 최적화) |
| 기능 | 모델 버전, 배치, 로깅 | 동적 배치, 모델 앙상블, 멀티 GPU |
| 사용성 | 간단 | 복잡 |
| 추천 | PyTorch만 사용, 빠른 시작 | 최고 성능, 여러 프레임워크 |

## 4.2 TorchServe 사용법
```bash
# 1. 설치
pip install torchserve torch-model-archiver

# 2. 모델 아카이브 생성 (.mar)
torch-model-archiver --model-name my_model \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pth \
  --handler image_classifier

# 3. 서버 시작
torchserve --start --model-store model_store --models my_model.mar

# 4. 추론 요청
curl http://127.0.0.1:8080/predictions/my_model -T image.jpg
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. GPU 메모리
- **단편화**: 작은 메모리 조각 문제
- **`empty_cache()`**: 캐시된 메모리 해제 (단편화 해결 X)
- **`memory_summary`**: 상세 메모리 사용량 분석

### 2. PyTorch Profiler
- **병목 분석**: CPU/CUDA 연산 시간, 메모리 할당
- **시각화**: TensorBoard로 Trace 확인
- **활용**: 느린 연산자 식별, 메모리 누수 추적

### 3. 동적 배치 처리
- **개념**: 요청을 큐에 모아 배치로 처리
- **효과**: GPU 활용도 극대화, 처리량 향상
- **구현**: 백그라운드 스레드, 큐, 타이머

### 4. 추론 서버
- **TorchServe**: PyTorch 전용, 간단
- **Triton**: NVIDIA, 최고 성능, 다중 프레임워크
- **장점**: 모델 버전 관리, 자동 배치, 모니터링

### 백엔드 최적화 체크리스트
```python
□ 1. 모델 최적화 (TorchScript, ONNX)
□ 2. 정밀도 최적화 (FP16/INT8)
□ 3. 프로파일러로 병목 지점 확인
□ 4. 동적 배치 구현 (처리량 중심)
□ 5. 추론 서버 도입 (운영 효율성)
□ 6. CPU/GPU 리소스 분배
□ 7. 로깅 및 모니터링 (Latency, Throughput)
□ 8. A/B 테스트 및 점진적 배포
```
