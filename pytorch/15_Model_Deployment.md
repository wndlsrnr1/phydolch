# 15. 모델 배포 기본

이번 단원에서는 **학습된 PyTorch 모델을 실제 서비스에 배포하는 방법**을 배워보겠습니다.

## 학습 목표
- 모델 저장 및 로드
- TorchScript로 모델 최적화
- ONNX 형식 변환
- FastAPI로 REST API 구축
- 백엔드 개발자를 위한 배포 전략

## 왜 이 단원이 필요한가?

모델을 학습하는 것과 배포하는 것은 **완전히 다른 문제**입니다:

- **학습**: 정확도 최우선, 시간 여유
- **배포**: 속도, 안정성, 확장성 중요

백엔드 개발자라면 **API 설계, 성능 최적화, 배포 파이프라인**에 익숙할 것입니다. 이 경험을 딥러닝 모델 배포에 적용해봅시다.

---
# 1. 모델 저장 및 로드
---

## 1.1 두 가지 방법

### 방법 1: State Dict 저장 (권장)
- **저장**: 모델의 파라미터만 저장
- **장점**: 유연성, 모델 구조 변경 가능
- **단점**: 모델 클래스 정의 필요

### 방법 2: 전체 모델 저장
- **저장**: 모델 구조 + 파라미터
- **장점**: 간단함
- **단점**: 유연성 낮음, pickle 의존

```python
import torch
import torch.nn as nn

# 예제 모델
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 생성 및 학습 (가상)
model = SimpleModel(10, 50, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 방법 1: State Dict 저장 (권장) ===
# 저장
checkpoint = {
    'epoch': 100,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.5,
}
torch.save(checkpoint, 'model_checkpoint.pth')
print("✓ Checkpoint 저장 완료: model_checkpoint.pth")

# 로드
model_loaded = SimpleModel(10, 50, 5)  # 모델 구조 정의 필요
optimizer_loaded = torch.optim.Adam(model_loaded.parameters())

checkpoint = torch.load('model_checkpoint.pth')
model_loaded.load_state_dict(checkpoint['model_state_dict'])
optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model_loaded.eval()  # 추론 모드
print(f"✓ Checkpoint 로드 완료: epoch={epoch}, loss={loss}")

# === 방법 2: 전체 모델 저장 ===
torch.save(model, 'model_full.pth')
print("\n✓ 전체 모델 저장 완료: model_full.pth")

model_loaded_full = torch.load('model_full.pth')
model_loaded_full.eval()
print("✓ 전체 모델 로드 완료")

# === 추론용 모델 저장 (가장 간단) ===
torch.save(model.state_dict(), 'model_weights.pth')
print("\n✓ 가중치만 저장: model_weights.pth (추론용)")
```

---
# 2. TorchScript
---

## 2.1 TorchScript란?

**TorchScript**: PyTorch 모델을 최적화된 중간 표현으로 변환

### 장점
1. **Python 독립**: Python 없이 실행 가능 (C++ 등)
2. **최적화**: 연산 융합, 불필요한 연산 제거
3. **배포 용이**: 모바일, 임베디드 환경
4. **성능 향상**: JIT 컴파일

### 두 가지 방법
1. **Tracing**: 예제 입력으로 실행 추적
2. **Scripting**: 코드를 직접 변환

```python
import torch
import torch.nn as nn

# 예제 모델
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
model.eval()

# === 방법 1: Tracing (추천) ===
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)

# 저장
traced_model.save('model_traced.pt')
print("✓ Traced 모델 저장: model_traced.pt")

# 로드 및 추론
loaded_traced = torch.jit.load('model_traced.pt')
output = loaded_traced(example_input)
print(f"✓ Traced 모델 추론: {output.shape}")

# === 방법 2: Scripting ===
# 제어 흐름(if, for)이 있는 모델에 적합
class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        if x.sum() > 0:
            return self.fc(x)
        else:
            return torch.zeros(x.size(0), 5)

conditional_model = ConditionalModel()
scripted_model = torch.jit.script(conditional_model)

scripted_model.save('model_scripted.pt')
print("\n✓ Scripted 모델 저장: model_scripted.pt")

# === 성능 비교 ===
import time

# 일반 모델
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(1000):
        _ = model(example_input)
    normal_time = time.time() - start

# TorchScript 모델
with torch.no_grad():
    start = time.time()
    for _ in range(1000):
        _ = traced_model(example_input)
    traced_time = time.time() - start

print(f"\n=== 성능 비교 (1000회) ===")
print(f"일반 모델: {normal_time:.4f}s")
print(f"TorchScript: {traced_time:.4f}s")
print(f"속도 향상: {normal_time / traced_time:.2f}x")
```

---
# 3. ONNX 변환
---

## 3.1 ONNX란?

**ONNX (Open Neural Network Exchange)**: 딥러닝 모델의 표준 형식

### 장점
1. **프레임워크 독립**: PyTorch → TensorFlow, TensorRT 등
2. **최적화**: ONNX Runtime으로 빠른 추론
3. **배포 용이**: 다양한 플랫폼 지원
4. **산업 표준**: 널리 사용됨

### 사용 시나리오
- 프로덕션 배포 (ONNX Runtime)
- 모바일/엣지 디바이스
- 다른 프레임워크로 이식

```python
import torch
import torch.nn as nn

# 예제 모델
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
model.eval()

# ONNX로 변환
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,                      # 모델
    dummy_input,                # 예제 입력
    'model.onnx',               # 출력 파일
    export_params=True,         # 파라미터 포함
    opset_version=11,           # ONNX 버전
    do_constant_folding=True,   # 최적화
    input_names=['input'],      # 입력 이름
    output_names=['output'],    # 출력 이름
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✓ ONNX 변환 완료: model.onnx")

# ONNX Runtime으로 추론 (설치 필요: pip install onnxruntime)
try:
    import onnxruntime as ort
    import numpy as np
    
    # ONNX 모델 로드
    ort_session = ort.InferenceSession('model.onnx')
    
    # 추론
    input_data = np.random.randn(1, 10).astype(np.float32)
    outputs = ort_session.run(
        None,
        {'input': input_data}
    )
    
    print(f"✓ ONNX Runtime 추론: {outputs[0].shape}")
    
    # 성능 비교
    import time
    
    # PyTorch
    with torch.no_grad():
        start = time.time()
        for _ in range(1000):
            _ = model(torch.from_numpy(input_data))
        pytorch_time = time.time() - start
    
    # ONNX Runtime
    start = time.time()
    for _ in range(1000):
        _ = ort_session.run(None, {'input': input_data})
    onnx_time = time.time() - start
    
    print(f"\n=== 성능 비교 (1000회) ===")
    print(f"PyTorch: {pytorch_time:.4f}s")
    print(f"ONNX Runtime: {onnx_time:.4f}s")
    print(f"속도 향상: {pytorch_time / onnx_time:.2f}x")
    
except ImportError:
    print("\n⚠️  ONNX Runtime이 설치되지 않음")
    print("설치: pip install onnxruntime")
```

---
# 4. FastAPI로 REST API 구축
---

## 4.1 FastAPI란?

**FastAPI**: 현대적이고 빠른 Python 웹 프레임워크

### 장점
1. **빠름**: Starlette + Pydantic 기반
2. **타입 힌트**: 자동 검증 및 문서화
3. **비동기**: async/await 지원
4. **자동 문서**: Swagger UI 자동 생성

### 설치
```bash
pip install fastapi uvicorn python-multipart
```

```python
# FastAPI 서버 코드 (app.py)
fastapi_code = '''
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from typing import List

app = FastAPI(title="PyTorch Model API")

# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 로드
model = SimpleModel()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# 요청/응답 스키마
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: List[float]
    class_id: int
    confidence: float

# Health check
@app.get("/")
async def root():
    return {"status": "ok", "model": "SimpleModel"}

# 예측 엔드포인트
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # 입력 검증
    if len(request.features) != 10:
        raise ValueError("Expected 10 features")
    
    # 추론
    with torch.no_grad():
        input_tensor = torch.tensor([request.features], dtype=torch.float32)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        class_id = probabilities.argmax().item()
        confidence = probabilities[0, class_id].item()
    
    return PredictionResponse(
        prediction=probabilities[0].tolist(),
        class_id=class_id,
        confidence=confidence
    )

# 배치 예측
@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    results = []
    
    with torch.no_grad():
        for req in requests:
            input_tensor = torch.tensor([req.features], dtype=torch.float32)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            results.append({
                "prediction": probabilities[0].tolist(),
                "class_id": probabilities.argmax().item(),
                "confidence": probabilities[0, probabilities.argmax()].item()
            })
    
    return {"results": results}

# 실행: uvicorn app:app --reload
# 문서: http://localhost:8000/docs
'''

# 파일로 저장
with open('app.py', 'w') as f:
    f.write(fastapi_code)

print("✓ FastAPI 서버 코드 생성: app.py")
print("\n실행 방법:")
print("  uvicorn app:app --reload")
print("\n문서 확인:")
print("  http://localhost:8000/docs")
print("\n테스트:")
print("  curl -X POST http://localhost:8000/predict \\")
print("    -H 'Content-Type: application/json' \\")
print("    -d '{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'")
```

---
# 5. 백엔드 개발자를 위한 배포 전략
---

## 5.1 배포 체크리스트

### 1. 모델 최적화
- ✓ TorchScript 또는 ONNX 변환
- ✓ Quantization (INT8)
- ✓ Pruning (가지치기)

### 2. API 설계
- ✓ RESTful API (FastAPI)
- ✓ 입력 검증 (Pydantic)
- ✓ 에러 처리
- ✓ Rate limiting

### 3. 성능 최적화
- ✓ Batch inference
- ✓ 비동기 처리
- ✓ 캐싱
- ✓ GPU 활용

### 4. 모니터링
- ✓ 응답 시간
- ✓ 처리량 (requests/sec)
- ✓ 에러율
- ✓ 리소스 사용량

### 5. 인프라
- ✓ Docker 컨테이너화
- ✓ Kubernetes 오케스트레이션
- ✓ Load balancing
- ✓ Auto-scaling

```python
# Dockerfile 예제
dockerfile = '''
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY app.py .
COPY model_weights.pth .

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

with open('Dockerfile', 'w') as f:
    f.write(dockerfile)

print("✓ Dockerfile 생성 완료")

# requirements.txt
requirements = '''
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
numpy==1.24.3
pydantic==2.5.0
python-multipart==0.0.6
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("✓ requirements.txt 생성 완료")

# Docker 명령어
print("\n=== Docker 빌드 및 실행 ===")
print("빌드: docker build -t pytorch-api .")
print("실행: docker run -p 8000:8000 pytorch-api")
print("테스트: curl http://localhost:8000")

# 성능 모니터링 코드
monitoring_code = '''
from fastapi import FastAPI, Request
import time
import logging

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 미들웨어: 응답 시간 측정
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"path={request.url.path} "
        f"method={request.method} "
        f"status={response.status_code} "
        f"duration={process_time:.4f}s"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response
'''

print("\n=== 모니터링 코드 ===")
print("응답 시간, 상태 코드, 경로 등을 로깅")
```

---
# 6. 배치 추론 최적화
---

```python
import torch
import torch.nn as nn
import time
import numpy as np

# 예제 모델
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
model.eval()

# 테스트 데이터
num_samples = 1000
test_data = [torch.randn(1, 10) for _ in range(num_samples)]

# === 방법 1: 개별 추론 ===
with torch.no_grad():
    start = time.time()
    results_individual = []
    for data in test_data:
        output = model(data)
        results_individual.append(output)
    individual_time = time.time() - start

print(f"=== 개별 추론 ===")
print(f"시간: {individual_time:.4f}s")
print(f"처리량: {num_samples / individual_time:.2f} samples/sec")

# === 방법 2: 배치 추론 ===
batch_size = 32

with torch.no_grad():
    start = time.time()
    results_batch = []
    
    for i in range(0, num_samples, batch_size):
        batch = torch.cat(test_data[i:i+batch_size], dim=0)
        output = model(batch)
        results_batch.append(output)
    
    batch_time = time.time() - start

print(f"\n=== 배치 추론 (batch_size={batch_size}) ===")
print(f"시간: {batch_time:.4f}s")
print(f"처리량: {num_samples / batch_time:.2f} samples/sec")
print(f"속도 향상: {individual_time / batch_time:.2f}x")

# === 동적 배치 처리 ===
class DynamicBatcher:
    """
    요청을 모아서 배치로 처리
    """
    def __init__(self, model, max_batch_size=32, max_wait_time=0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = []
    
    def predict(self, input_data):
        self.queue.append(input_data)
        
        # 배치가 찼거나 대기 시간 초과
        if len(self.queue) >= self.max_batch_size:
            return self._process_batch()
        
        # 실제로는 타이머 사용
        return None
    
    def _process_batch(self):
        if not self.queue:
            return []
        
        batch = torch.cat(self.queue, dim=0)
        with torch.no_grad():
            outputs = self.model(batch)
        
        results = [outputs[i:i+1] for i in range(len(outputs))]
        self.queue = []
        return results

print("\n✓ 동적 배치 처리 클래스 준비 완료")
print("실전에서는 asyncio와 함께 사용")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 모델 저장/로드
- **State Dict**: 파라미터만 저장 (권장)
- **전체 모델**: 구조 + 파라미터
- **Checkpoint**: Epoch, optimizer 포함

### 2. TorchScript
- **Tracing**: 예제 입력으로 추적
- **Scripting**: 제어 흐름 지원
- **효과**: Python 독립, 최적화, 성능 향상

### 3. ONNX
- **표준 형식**: 프레임워크 독립
- **ONNX Runtime**: 빠른 추론
- **배포**: 다양한 플랫폼 지원

### 4. FastAPI
- **REST API**: 빠르고 현대적
- **타입 힌트**: 자동 검증
- **문서화**: Swagger UI 자동 생성

### 5. 배포 전략
- **최적화**: TorchScript, ONNX, Quantization
- **API**: RESTful, 입력 검증, 에러 처리
- **성능**: 배치 추론, 비동기, 캐싱
- **인프라**: Docker, Kubernetes, 모니터링

### 배치 추론
- **개별 추론**: 간단하지만 느림
- **배치 추론**: 10~100배 빠름
- **동적 배치**: 요청을 모아서 처리

### 실전 체크리스트
```python
1. 모델 최적화 (TorchScript/ONNX)
2. API 구축 (FastAPI)
3. Docker 컨테이너화
4. 배치 추론 구현
5. 모니터링 설정
6. 로드 테스트
7. 프로덕션 배포
```

### 다음 단계
- 실제 프로젝트 구현
- 전이 학습
- 시계열 예측
- 텍스트 분류
- 빅데이터 처리

축하합니다! 이제 PyTorch 모델을 배포할 준비가 되었습니다! 🚀
