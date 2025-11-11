# FAQ: 자주 묻는 질문과 오해하기 쉬운 개념

이 문서는 학습 중 자주 발생하는 질문과 오해를 정리했습니다.

## Q1: 행렬곱 A @ B에서 왜 B가 전치(transpose)되지 않나요?

### 오해

많은 초보자가 `A @ B`를 할 때 B가 자동으로 전치된다고 생각합니다.

### 올바른 이해

**PyTorch의 행렬곱은 수학적 표기와 동일합니다!**

```python
A = torch.tensor([[1, 2], [3, 4]])  # 2x2
B = torch.tensor([[5, 6], [7, 8]])  # 2x2
C = A @ B  # 수학적으로 C_{ij} = sum_k A_{ik} * B_{kj}
```

**nn.Linear에서는**:
```python
# output = input @ weight.T + bias
# weight가 전치됩니다!
```

왜? 수학적 관습에서 행벡터를 기본으로 하기 때문입니다.

### 결론

- **torch.matmul / @**: 수학적 행렬곱
- **nn.Linear**: 내부적으로 weight.T를 사용하여 편의성 제공

## Q2: require_grad=True를 모든 텐서에 설정해야 하나요?

### 오해

모든 텐서에 `require_grad=True`를 설정해야 한다고 생각합니다.

### 올바른 이해

**데이터는 gradient 불필요, 파라미터만 필요합니다!**

```python
# ❌ 잘못된 예
x_train = torch.tensor([...], requires_grad=True)  # 불필요!
y_train = torch.tensor([...], requires_grad=True)  # 불필요!

# ✅ 올바른 예
x_train = torch.tensor([...], requires_grad=False)  # False가 기본값
y_train = torch.tensor([...], requires_grad=False)

# 모델 파라미터는 자동으로 requires_grad=True
model = nn.Linear(10, 1)
```

**규칙**:
- 입력 데이터(x, y): gradient 불필요 → False (기본값)
- 모델 파라미터(weight, bias): gradient 필요 → 자동으로 True

### 왜?

입력 데이터는 고정되어 있고, 학습해야 하는 것은 **모델의 가중치**입니다!

## Q3: backward() 두 번 호출하면 에러가 나는 이유는?

### 문제

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()  # 첫 번째 OK
y.backward()  # ❌ RuntimeError!
```

### 이유

**계산 그래프가 역전파 후 소멸되기 때문입니다.**

- 첫 번째 backward(): gradient 계산 후 그래프 삭제
- 두 번째 backward(): 그래프가 이미 없어서 에러

### 해결

```python
# 방법 1: retain_graph=True
y.backward(retain_graph=True)
y.backward()  # OK

# 방법 2: gradient 초기화 후 다시 계산
x.grad.zero_()
y.backward()  # OK
```

### 언제 사용?

**거의 사용하지 않습니다!**  
한 샘플에 대해 한 번만 backward()를 호출하면 됩니다.

## Q4: 왜 optimizer.zero_grad()를 호출해야 하나요?

### 문제

```python
# Learning rate가 이상하게 커짐
for batch in dataloader:
    loss.backward()  # gradient 계산
    optimizer.step()  # 업데이트
```

### 이유

**PyTorch는 gradient를 누적합니다!**

- backward()를 호출하면 `param.grad`에 **덧셈**
- 이전 gradient가 그대로 남아있음
- accumulator처럼 계속 더해짐

### 해결

```python
for batch in dataloader:
    optimizer.zero_grad()  # gradient 초기화!
    loss.backward()
    optimizer.step()
```

**zero_grad()가 없다면**:  
$$
w_{new} = w_{old} - \eta \cdot (\nabla L_1 + \nabla L_2 + \nabla L_3 + ...)
$$

**zero_grad()가 있다면**:  
$$
w_{new} = w_{old} - \eta \cdot \nabla L_3
$$

### 비유

계산기를 초기화하지 않고 더하기만 계속 누르면 합이 계속 커지는 것과 같습니다!

## Q5: reshape vs view vs flatten 차이점은?

### 간단히

- **reshape**: 가능하면 view처럼, 불가능하면 복사
- **view**: 메모리 공유 (빠름, 제약 많음)
- **flatten**: 지정한 차원부터 끝까지 평탄화

### 예시

```python
x = torch.randn(2, 3, 4)

# 모두 2x12로 변환
x_reshape = x.reshape(2, 12)     # 안전
x_view = x.view(2, 12)            # 빠름, 제약 있음
x_flat = x.flatten(1)             # 1차원부터 평탄화 (2x12)
```

**가이드**:
- reshape: 대부분의 경우 사용
- view: 성능 중요하고 contiguity 보장될 때
- flatten: 간단히 평탄화

## Q6: loss가 NaN이 나오는 이유는?

### 가능한 원인

1. **Learning rate가 너무 큼**
   - 해결: 0.001, 0.0001로 줄이기

2. **입력에 잘못된 값 (inf, nan)**
   - 해결: 데이터 검증

3. **수치 불안정 (log(0), sqrt(-1) 등)**
   - 해결: epsilon 추가

### 디버깅

```python
# 1. Loss 확인
if torch.isnan(loss):
    print("Loss is NaN!")
    
# 2. Gradient 확인
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}!")

# 3. Input 확인
if torch.isnan(x).any():
    print("NaN in input!")
```

### 일반적 해결법

```python
# Learning rate 감소
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Q7: nn.Module을 상속하는 이유는?

### 이유

**PyTorch의 모든 기능을 사용하려면** nn.Module이 필요합니다:

1. **자동 gradient 추적**: requires_grad 자동 관리
2. **parameter 확인**: `model.parameters()` 
3. **디바이스 이동**: `model.cuda()`, `model.to(device)`
4. **train/eval 모드**: `model.train()`, `model.eval()`
5. **모델 저장/로드**: `torch.save(model.state_dict())`

### 예시

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 필수!
        self.layer = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.layer(x)

model = MyModel()
print(list(model.parameters()))  # super().__init__() 필요!
```

**super().__init__()를 빼면**: parameter 추적 불가!

## Q8: 왜 모델을 .eval() 모드로 설정하나요?

### 이유

**Dropout, BatchNorm 등이 평가 시 다르게 동작하기 때문입니다.**

### 차이점

```python
# 학습 시
model.train()  # Dropout 활성화, BatchNorm 업데이트
for batch in train_loader:
    output = model(x)
    
# 평가 시
model.eval()  # Dropout 비활성화, BatchNorm 고정
with torch.no_grad():
    for batch in test_loader:
        output = model(x)
```

### 왜 중요?

- **Dropout**: 학습 시는 랜덤하게 뉴런 끔, 평가 시는 모두 사용
- **BatchNorm**: 학습 시는 배치 통계 사용, 평가 시는 이동평균 사용

**실수하면**: 학습 정확도와 테스트 정확도 차이가 커짐!

### 규칙

```python
# 학습 루프
model.train()
for ...:
    ...

# 평가
model.eval()
with torch.no_grad():
    ...
```

## Q9: torch.no_grad()를 왜 사용하나요?

### 이유

**평가 시 gradient 불필요 → 메모리 절약 + 속도 향상**

### 비교

```python
# ❌ 비효율적
for x in test_data:
    y = model(x)  # gradient 계산됨

# ✅ 효율적
with torch.no_grad():
    for x in test_data:
        y = model(x)  # gradient 계산 안 됨
```

### 효과

- **메모리**: 30-50% 절약
- **속도**: 20-30% 향상
- **안전성**: 실수로 update 방지

### 언제 사용?

**모든 평가/추론 시 사용!**
- Test set 평가
- 예측 생성
- Validation

**학습 시에는 사용하지 않음!**

## Q10: "벡터와 행렬이 뭐가 다른가요?"

### 간단히

- **벡터**: 숫자들의 1차원 배열 (예: [1, 2, 3])
- **행렬**: 벡터들의 2차원 배열 (예: [[1,2], [3,4]])
- **텐서**: 벡터/행렬의 일반화 (N차원)

### PyTorch에서

```python
# 벡터 (1차원 텐서)
v = torch.tensor([1.0, 2.0, 3.0])      # shape: (3,)

# 행렬 (2차원 텐서)
M = torch.tensor([[1, 2], [3, 4]])    # shape: (2, 2)

# 3차원 텐서
T = torch.randn(10, 32, 32)            # shape: (10, 32, 32)
```

### 딥러닝에서

- **입력 데이터**: 벡터 (예: [픽셀1, 픽셀2, ...])
- **가중치**: 행렬 (예: [[w11, w12], [w21, w22]])
- **배치 처리**: 텐서 (예: (64, 784) = 64개 샘플 각 784차원)

**결론**: 모든 것은 텐서이고, 차원(shape)만 다릅니다!
