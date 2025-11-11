# 17. 시계열 예측 (Time Series Prediction)

이번 단원에서는 **시계열 데이터를 다루고 미래를 예측하는 방법**을 배워보겠습니다.

## 학습 목표
- 시계열 데이터의 특성
- RNN, LSTM, GRU 구조
- 시계열 데이터 전처리
- Seq2Seq 모델
- 실전 주가/수요 예측 프로젝트

## 왜 이 단원이 필요한가?

**시계열 데이터는 어디에나 있습니다**:

- **금융**: 주가, 환율, 암호화폐
- **비즈니스**: 매출, 수요, 재고
- **IoT**: 센서 데이터, 로그
- **웹**: 트래픽, 사용자 행동

백엔드 개발자라면 **로그 분석, 트래픽 예측, 이상 탐지**가 익숙할 것입니다. 딥러닝으로 더 정교하게 처리해봅시다!

---
# 1. 시계열 데이터의 특성
---

## 1.1 시계열 vs 일반 데이터

### 일반 데이터 (IID: Independent and Identically Distributed)
```python
# 순서 무관
[이미지1, 이미지2, 이미지3] == [이미지3, 이미지1, 이미지2]
```

### 시계열 데이터
```python
# 순서 중요!
[t1, t2, t3, t4, t5] ≠ [t5, t1, t3, t2, t4]
```

## 1.2 시계열의 구성 요소

```
시계열 = 추세(Trend) + 계절성(Seasonality) + 잔차(Residual)
```

- **추세**: 장기적 방향성 (↗ 상승, ↘ 하락)
- **계절성**: 주기적 패턴 (일별, 주별, 연별)
- **잔차**: 설명되지 않는 노이즈

## 1.3 시계열 예측의 종류

**1. One-step Prediction**
```
[t1, t2, t3, t4, t5] → [t6]
```

**2. Multi-step Prediction**
```
[t1, t2, t3, t4, t5] → [t6, t7, t8]
```

**3. Sequence-to-Sequence**
```
[t1, t2, t3, t4, t5] → [t6, t7, t8, t9, t10]
```

---
# 2. RNN, LSTM, GRU
---

## 2.1 RNN (Recurrent Neural Network)

### 기본 구조
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

- `h_t`: 현재 hidden state
- `h_{t-1}`: 이전 hidden state (메모리)
- `x_t`: 현재 입력

### 문제점: Vanishing Gradient
- 긴 시퀀스에서 gradient 소실
- 장기 의존성 학습 어려움

## 2.2 LSTM (Long Short-Term Memory)

### 핵심 아이디어: Cell State
```
장기 메모리 (Cell State) + 단기 메모리 (Hidden State)
```

### 3개의 게이트
1. **Forget Gate**: 무엇을 잊을까?
2. **Input Gate**: 무엇을 기억할까?
3. **Output Gate**: 무엇을 출력할까?

## 2.3 GRU (Gated Recurrent Unit)

### LSTM의 간소화 버전
- 2개의 게이트 (Update, Reset)
- 더 빠른 학습
- 비슷한 성능

```python
import torch
import torch.nn as nn

# RNN, LSTM, GRU 비교
batch_size = 32
seq_length = 10
input_size = 5
hidden_size = 20

# 입력 데이터: [batch, seq_len, input_size]
x = torch.randn(batch_size, seq_length, input_size)

# === 1. RNN ===
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
output_rnn, h_n = rnn(x)

print("=== RNN ===")
print(f"Output shape: {output_rnn.shape}")  # [batch, seq, hidden]
print(f"Hidden state: {h_n.shape}")         # [1, batch, hidden]

# === 2. LSTM ===
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output_lstm, (h_n, c_n) = lstm(x)

print("\n=== LSTM ===")
print(f"Output shape: {output_lstm.shape}")
print(f"Hidden state: {h_n.shape}")
print(f"Cell state: {c_n.shape}")  # LSTM만 cell state 있음

# === 3. GRU ===
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output_gru, h_n = gru(x)

print("\n=== GRU ===")
print(f"Output shape: {output_gru.shape}")
print(f"Hidden state: {h_n.shape}")

# 파라미터 수 비교
rnn_params = sum(p.numel() for p in rnn.parameters())
lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print("\n=== 파라미터 수 ===")
print(f"RNN:  {rnn_params:,}")
print(f"LSTM: {lstm_params:,} (RNN의 {lstm_params/rnn_params:.1f}배)")
print(f"GRU:  {gru_params:,} (RNN의 {gru_params/rnn_params:.1f}배)")

print("\n선택 가이드:")
print("- RNN: 짧은 시퀀스, 빠른 학습")
print("- LSTM: 긴 시퀀스, 장기 의존성")
print("- GRU: LSTM과 비슷하지만 더 빠름")
```

---
# 3. 시계열 데이터 전처리
---

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 1. 시계열 데이터셋 생성
class TimeSeriesDataset(Dataset):
    """
    시계열 데이터를 sliding window로 변환
    """
    def __init__(self, data, window_size, horizon=1):
        """
        Args:
            data: 시계열 데이터 [time_steps, features]
            window_size: 입력 윈도우 크기
            horizon: 예측 기간
        """
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1
    
    def __getitem__(self, idx):
        # 입력: [idx : idx + window_size]
        x = self.data[idx : idx + self.window_size]
        
        # 타겟: [idx + window_size : idx + window_size + horizon]
        y = self.data[
            idx + self.window_size : idx + self.window_size + self.horizon
        ]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

# 2. 예제 데이터 생성 (사인파 + 노이즈)
np.random.seed(42)
time_steps = 1000
t = np.linspace(0, 100, time_steps)
data = np.sin(t) + 0.1 * np.random.randn(time_steps)
data = data.reshape(-1, 1)  # [time_steps, 1]

print("=== 원본 데이터 ===")
print(f"Shape: {data.shape}")
print(f"Range: [{data.min():.2f}, {data.max():.2f}]")

# 3. 정규화 (중요!)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

print("\n=== 정규화 후 ===")
print(f"Range: [{data_normalized.min():.2f}, {data_normalized.max():.2f}]")

# 4. Train/Val 분할
train_size = int(0.8 * len(data_normalized))
train_data = data_normalized[:train_size]
val_data = data_normalized[train_size:]

print(f"\n=== 데이터 분할 ===")
print(f"Train: {len(train_data)} steps")
print(f"Val: {len(val_data)} steps")

# 5. Dataset & DataLoader 생성
window_size = 50
horizon = 1  # 1-step ahead prediction

train_dataset = TimeSeriesDataset(train_data, window_size, horizon)
val_dataset = TimeSeriesDataset(val_data, window_size, horizon)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"\n=== Dataset ===")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# 샘플 확인
x_sample, y_sample = train_dataset[0]
print(f"\nSample:")
print(f"  Input shape: {x_sample.shape}  (window_size, features)")
print(f"  Target shape: {y_sample.shape}  (horizon, features)")

print("\n✓ 시계열 데이터 전처리 완료!")
```

---
# 4. LSTM 시계열 예측 모델
---

```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    LSTM 기반 시계열 예측 모델
    """
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_layers, 
        output_size,
        dropout=0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            output: [batch, output_size]
        """
        # LSTM forward
        # output: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        # c_n: [num_layers, batch, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 time step의 출력 사용
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # 또는 마지막 hidden state 사용
        # last_hidden = h_n[-1]  # [batch, hidden_size]
        
        # 예측
        prediction = self.fc(last_output)  # [batch, output_size]
        
        return prediction

# 모델 생성
input_size = 1      # 특징 수
hidden_size = 64    # LSTM hidden size
num_layers = 2      # LSTM 레이어 수
output_size = 1     # 예측할 값의 수

model = LSTMPredictor(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    dropout=0.2
)

print("=== LSTM Predictor ===")
print(model)

# 파라미터 수
total_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터: {total_params:,}")

# Forward pass 테스트
batch_size = 32
seq_length = 50
x_test = torch.randn(batch_size, seq_length, input_size)
y_pred = model(x_test)

print(f"\n입력 shape: {x_test.shape}")
print(f"출력 shape: {y_pred.shape}")
print("\n✓ 모델 준비 완료!")
```

---
# 5. 학습 및 평가
---

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    한 epoch 학습
    """
    model.train()
    total_loss = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        # Forward
        optimizer.zero_grad()
        y_pred = model(x)
        
        # y를 [batch, output_size]로 reshape
        y = y.squeeze(1)  # [batch, horizon, features] → [batch, features]
        
        loss = criterion(y_pred, y)
        
        # Backward
        loss.backward()
        
        # Gradient clipping (RNN에서 중요!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    검증
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            y = y.squeeze(1)
            
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    return total_loss / len(dataloader), predictions, actuals

# 학습 함수
def train_time_series(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    device, num_epochs=100, patience=10
):
    """
    시계열 모델 학습
    """
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # 학습
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 검증
        val_loss, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_ts_model.pth')
        else:
            patience_counter += 1
        
        # 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Best Val Loss: {best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ 학습 완료! Best Val Loss: {best_val_loss:.6f}")
    return history

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

print("=== 학습 설정 ===")
print(f"Device: {device}")
print(f"Loss: MSE")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Early Stopping: patience=10")
print("\n학습 함수 준비 완료!")
```

---
# 6. Multi-step Prediction
---

```python
import torch
import torch.nn as nn

class MultiStepLSTM(nn.Module):
    """
    Multi-step 예측을 위한 LSTM
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_steps,  # 예측할 미래 step 수
        dropout=0.2
    ):
        super().__init__()
        
        self.output_steps = output_steps
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-step 출력
        self.fc = nn.Linear(hidden_size, output_steps)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            output: [batch, output_steps]
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions

# === 방법 2: Autoregressive (재귀적 예측) ===
class AutoregressiveLSTM(nn.Module):
    """
    자기회귀 방식의 multi-step 예측
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, future_steps):
        """
        재귀적으로 미래 예측
        """
        outputs = []
        
        # 초기 LSTM state 계산
        _, (h, c) = self.lstm(x)
        
        # 마지막 입력
        current_input = x[:, -1:, :]  # [batch, 1, input_size]
        
        # 재귀적 예측
        for _ in range(future_steps):
            # 한 step 예측
            lstm_out, (h, c) = self.lstm(current_input, (h, c))
            prediction = self.fc(lstm_out)  # [batch, 1, input_size]
            
            outputs.append(prediction)
            
            # 다음 입력으로 사용
            current_input = prediction
        
        # [batch, future_steps, input_size]
        return torch.cat(outputs, dim=1)

# 비교
print("=== Multi-step Prediction 방법 ===")
print("\n1. Direct Multi-output")
print("   - 한 번에 여러 step 예측")
print("   - 빠름")
print("   - 독립적 예측")

print("\n2. Autoregressive")
print("   - 재귀적으로 한 step씩 예측")
print("   - 느림")
print("   - 이전 예측을 활용")
print("   - 오차 누적 가능")

# 모델 생성
model_direct = MultiStepLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_steps=10  # 10 step ahead
)

model_ar = AutoregressiveLSTM(
    input_size=1,
    hidden_size=64,
    num_layers=2
)

print("\n✓ Multi-step 모델 준비 완료!")
```

---
# 7. 실전 팁
---

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 예측 결과 시각화
def plot_predictions(actuals, predictions, title="Predictions"):
    """
    예측 결과 시각화
    """
    plt.figure(figsize=(15, 5))
    
    plt.plot(actuals, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 2. 평가 지표
def calculate_metrics(actuals, predictions):
    """
    시계열 예측 평가 지표
    """
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actuals - predictions))
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    # R² Score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

# 3. 역정규화
def inverse_transform(data, scaler):
    """
    정규화된 데이터를 원래 스케일로 복원
    """
    return scaler.inverse_transform(data)

print("=== 시계열 예측 실전 팁 ===")
print("\n1. 데이터 전처리")
print("   ✓ 정규화 필수 (MinMaxScaler 또는 StandardScaler)")
print("   ✓ 결측치 처리 (interpolation)")
print("   ✓ 이상치 제거 또는 클리핑")

print("\n2. 모델 설계")
print("   ✓ Window size: 계절성 주기 고려")
print("   ✓ Hidden size: 64~256")
print("   ✓ Num layers: 2~3 (깊을수록 좋지 않음)")
print("   ✓ Dropout: 0.2~0.5")

print("\n3. 학습")
print("   ✓ Gradient clipping 필수 (max_norm=1.0)")
print("   ✓ Early stopping (patience=10~20)")
print("   ✓ Learning rate scheduling")
print("   ✓ Batch size: 32~128")

print("\n4. 평가")
print("   ✓ MAE, RMSE, MAPE 모두 확인")
print("   ✓ 시각화로 패턴 확인")
print("   ✓ 역정규화 후 평가")

print("\n5. 백엔드 배포")
print("   ✓ Scaler도 함께 저장")
print("   ✓ Window 관리 (sliding window)")
print("   ✓ 배치 추론으로 성능 향상")
print("   ✓ 캐싱 활용")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 시계열 데이터
- **특성**: 순서 중요, 시간 의존성
- **구성**: 추세 + 계절성 + 잔차
- **전처리**: 정규화, sliding window

### 2. RNN 계열 모델
- **RNN**: 기본, Vanishing gradient 문제
- **LSTM**: Cell state, 장기 의존성 학습
- **GRU**: LSTM 간소화, 빠른 학습

### 3. 예측 방법
- **One-step**: 다음 1개 값 예측
- **Multi-step Direct**: 여러 값 동시 예측
- **Autoregressive**: 재귀적 예측

### 4. 데이터 전처리
```python
1. 정규화 (MinMaxScaler)
2. Sliding window 생성
3. Train/Val 분할 (시간 순서 유지)
4. DataLoader 구성
```

### 5. 학습 팁
- **Gradient clipping**: 필수!
- **Early stopping**: patience=10~20
- **Learning rate**: 0.001 시작
- **Batch size**: 32~128

### 6. 평가 지표
- **MAE**: 평균 절대 오차
- **RMSE**: 평균 제곱근 오차
- **MAPE**: 평균 절대 백분율 오차
- **R²**: 결정 계수

### 모델 선택 가이드
| 상황 | 추천 모델 |
|------|----------|
| 짧은 시퀀스 (< 50) | RNN, GRU |
| 긴 시퀀스 (> 50) | LSTM, GRU |
| 빠른 학습 필요 | GRU |
| 최고 성능 | LSTM (2~3 layers) |

### 백엔드 배포 고려사항
```python
1. Scaler 저장 및 로드
2. Window 관리 (최근 N개 데이터)
3. 배치 추론 (여러 시계열 동시 처리)
4. 캐싱 (중간 결과 재사용)
5. 주기적 재학습 (drift 대응)
```

다음 단원에서는 **텍스트 분류**를 배워보겠습니다!
