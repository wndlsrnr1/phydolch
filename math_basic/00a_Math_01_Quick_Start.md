# 00. Quick Start: 30분 PyTorch 전체 흐름 체험

이 단원은 **전체 학습 흐름**을 빠르게 체험해보는 가이드입니다.

## 목표

**30분 안에** PyTorch의 핵심 개념을 전체적으로 경험합니다:
- 텐서 (Tensor) 기본
- 간단한 신경망 (Neural Network) 구성
- 데이터 로드
- 학습 루프
- 예측 (Prediction)

**중요**: 여기서는 "왜"보다 "무엇을 하는가"를 빠르게 느끼는 것이 목표입니다.  
각 개념의 **왜/무엇/어떻게**는 이후 단원에서 자세히 다룹니다.

## 1단계: 텐서 (Tensor) 기본 이해 (5분)

먼저 PyTorch의 기본 데이터 구조인 텐서를 만들어봅시다.

```python
import torch

# 텐서 생성
x = torch.tensor([1.0, 2.0, 3.0])
print(f"x: {x}")
print(f"x의 shape: {x.shape}")

# 행렬 생성
W = torch.tensor([[1.0, 2.0], 
                  [3.0, 4.0]])
print(f"\nW:\n{W}")
print(f"W의 shape: {W.shape}")

# 행렬곱 (Matrix Multiplication) (이것이 신경망의 핵심!)
y = W @ x
print(f"\ny = W @ x: {y}")
```

## 2단계: 간단한 신경망 (Neural Network) 구성 (5분)

이제 실제 신경망을 만들어봅시다.

```python
import torch.nn as nn

# 간단한 선형 레이어 (Linear Layer)
model = nn.Linear(3, 1)  # 입력 3차원 → 출력 1차원
print(f"모델: {model}")

# 예측 (Prediction) (순전파, Forward Pass)
x_sample = torch.tensor([[1.0, 2.0, 3.0]])
prediction = model(x_sample)
print(f"\n입력: {x_sample}")
print(f"예측값: {prediction}")

# 파라미터 확인
print(f"\n가중치 (Weight): {model.weight}")
print(f"편향 (Bias): {model.bias}")
```

## 3단계: 실제 데이터 로드 (5분)

이제 실제 데이터를 로드해봅시다. 간단한 예시로 더미 데이터를 사용합니다.

```python
# 간단한 회귀 데이터 생성
torch.manual_seed(42)

# 입력 데이터
X_train = torch.randn(100, 3)  # 100개 샘플, 3차원 특징

# 목표 값 (간단한 선형 관계)
y_train = X_train.sum(dim=1, keepdim=True)  # 각 샘플의 합

print(f"학습 데이터 크기: {X_train.shape}")
print(f"정답 크기: {y_train.shape}")
print(f"\n첫 5개 샘플:\n{X_train[:5]}")
```

## 4단계: 학습 루프 (10분)

가장 중요한 단계입니다. **학습**은 모델이 점점 정답에 가까워지도록 파라미터를 조정하는 것입니다. (최적화, Optimization)

```python
# 모델, 손실함수 (Loss Function), 옵티마이저 (Optimizer) 준비
model = nn.Linear(3, 1)
criterion = nn.MSELoss()  # 평균 제곱 오차 (Mean Squared Error, MSE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 시작!
print("학습 시작...\n")

for epoch in range(20):
    # Forward: 예측 (순전파)
    predictions = model(X_train)
    
    # Loss 계산: 예측과 정답의 차이
    loss = criterion(predictions, y_train)
    
    # Backward: 기울기 (Gradient) 계산 (역전파)
    optimizer.zero_grad()  # 이전 gradient 초기화
    loss.backward()         # gradient 계산
    
    # Update: 파라미터 업데이트
    optimizer.step()        # 가중치 조정
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

print("\n학습 완료!")
```

## 5단계: 결과 확인 (5분)

학습된 모델이 잘 작동하는지 확인해봅시다.

```python
# 새 샘플로 예측 (Prediction)
X_test = torch.randn(5, 3)
y_test = X_test.sum(dim=1, keepdim=True)

with torch.no_grad():  # gradient 계산 비활성화
    predictions = model(X_test)

print("예측 결과 비교:")
for i in range(5):
    print(f"입력: {X_test[i].tolist()}")
    print(f"  정답: {y_test[i].item():.2f}")
    print(f"  예측: {predictions[i].item():.2f}")
    print()
```

## 축하합니다! 🎉

30분 안에 전체 흐름을 체험했습니다:
- ✅ 텐서 (Tensor)와 행렬곱 (Matrix Multiplication)
- ✅ 신경망 (Neural Network) 구성
- ✅ 데이터 로드
- ✅ 학습 루프 (순전파 → 손실 → 역전파 → 업데이트)
- ✅ 예측 (Prediction)

### 다음 단계

이제 **왜 이렇게 동작하는가**를 자세히 배워봅시다:
1. **00_Math_Foundations**: 벡터와 행렬의 기하학적 의미
2. **01_Tensor_Basics**: 텐서 자세히 다루기
3. **03_Autograd**: backward()가 어떻게 기울기 (Gradient)를 계산하는가
4. **10_First_Project**: 완전한 프로젝트 경험

**전체 로드맵**: 00_Roadmap.ipynb 참고
