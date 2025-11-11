# 18. 텍스트 분류 (Text Classification)

이번 단원에서는 **자연어 처리(NLP)의 기초인 텍스트 분류**를 배워보겠습니다.

## 학습 목표
- 텍스트 데이터 전처리 (토큰화, 정제)
- 단어 임베딩 (Word2Vec, GloVe)
- RNN/LSTM 기반 텍스트 분류
- Transformer 기초
- 실전 감성 분석 프로젝트

## 왜 이 단원이 필요한가?

**텍스트 데이터는 가장 흔한 비정형 데이터**입니다:

- **고객 피드백**: 리뷰, 설문, 문의
- **소셜 미디어**: 트윗, 댓글, 포스트
- **문서**: 이메일, 보고서, 기사
- **로그**: 에러 메시지, 시스템 로그

백엔드 개발자라면 **로그 분석, 스팸 필터링, 챗봇**에 관심이 있을 것입니다!

---
# 1. 텍스트 전처리
---

## 1.1 전처리 파이프라인

```
원본 텍스트
  ↓
정제 (Cleaning)
  ↓
토큰화 (Tokenization)
  ↓
정규화 (Normalization)
  ↓
수치화 (Numericalization)
```

```python
import re
from collections import Counter

# 1. 텍스트 정제
def clean_text(text):
    """
    텍스트 정제
    """
    # 소문자 변환
    text = text.lower()
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # URL 제거
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)
    
    # 특수문자 제거 (일부 유지)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 2. 토큰화
def tokenize(text):
    """
    간단한 토큰화 (공백 기준)
    """
    return text.split()

# 3. 어휘 사전 구축
class Vocabulary:
    """
    단어 <-> 인덱스 매핑
    """
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
    
    def build(self, texts):
        """
        텍스트 리스트로부터 어휘 구축
        """
        # 단어 빈도 계산
        for text in texts:
            tokens = tokenize(clean_text(text))
            self.word_counts.update(tokens)
        
        # 빈도 기준으로 어휘 추가
        idx = 2
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, text):
        """
        텍스트를 인덱스 시퀀스로 변환
        """
        tokens = tokenize(clean_text(text))
        return [self.word2idx.get(token, 1) for token in tokens]  # 1 = <UNK>
    
    def decode(self, indices):
        """
        인덱스 시퀀스를 텍스트로 변환
        """
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)

# 예제
texts = [
    "This is a great movie! I loved it.",
    "Terrible film. Don't waste your time.",
    "Amazing performance by the actors."
]

vocab = Vocabulary(min_freq=1)
vocab.build(texts)

print("=== 어휘 사전 ===")
print(f"어휘 크기: {len(vocab)}")
print(f"\n샘플 단어 → 인덱스:")
for word in ['this', 'great', 'movie', 'unknown']:
    idx = vocab.word2idx.get(word, 1)
    print(f"  {word}: {idx}")

# 인코딩
text = "This movie is great!"
encoded = vocab.encode(text)
print(f"\n원본: {text}")
print(f"인코딩: {encoded}")
print(f"디코딩: {vocab.decode(encoded)}")
```

---
# 2. 단어 임베딩 (Word Embedding)
---

## 2.1 왜 임베딩이 필요한가?

### One-hot Encoding의 문제
```python
어휘 크기 = 10,000
"cat" = [0, 0, 0, ..., 1, ..., 0]  # 10,000 차원
"dog" = [0, 0, 1, ..., 0, ..., 0]  # 10,000 차원
```
- 희소(sparse), 고차원
- 단어 간 유사도 없음

### Word Embedding
```python
"cat" = [0.2, -0.5, 0.8, ...]  # 100~300 차원
"dog" = [0.3, -0.4, 0.7, ...]  # 유사한 벡터!
```
- 밀집(dense), 저차원
- 의미적 유사도 반영

```python
import torch
import torch.nn as nn

# 1. PyTorch Embedding 레이어
vocab_size = 10000
embedding_dim = 300

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=0  # <PAD> 토큰
)

print("=== Embedding 레이어 ===")
print(f"어휘 크기: {vocab_size:,}")
print(f"임베딩 차원: {embedding_dim}")
print(f"파라미터 수: {vocab_size * embedding_dim:,}")

# 사용 예시
# 입력: [batch_size, seq_length]
input_ids = torch.LongTensor([
    [1, 2, 3, 4, 0],  # 문장 1 (0은 패딩)
    [5, 6, 7, 0, 0]   # 문장 2
])

# 출력: [batch_size, seq_length, embedding_dim]
embedded = embedding(input_ids)

print(f"\n입력 shape: {input_ids.shape}")
print(f"출력 shape: {embedded.shape}")

# 2. Pre-trained Embedding 로드
def load_pretrained_embeddings(vocab, embedding_file):
    """
    GloVe 등 pre-trained embedding 로드
    """
    embeddings = {}
    
    # 파일에서 로드 (형식: word vec1 vec2 ...)
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.FloatTensor([float(v) for v in values[1:]])
            embeddings[word] = vector
    
    # Embedding matrix 생성
    embedding_dim = len(next(iter(embeddings.values())))
    embedding_matrix = torch.randn(len(vocab), embedding_dim)
    
    # 어휘에 있는 단어만 로드
    for word, idx in vocab.word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
    
    return embedding_matrix

print("\n=== Pre-trained Embedding ===")
print("사용 가능한 모델:")
print("  - GloVe (Stanford)")
print("  - Word2Vec (Google)")
print("  - FastText (Facebook)")
print("\n다운로드: https://nlp.stanford.edu/projects/glove/")
```

---
# 3. LSTM 텍스트 분류 모델
---

```python
import torch
import torch.nn as nn

class TextClassifierLSTM(nn.Module):
    """
    LSTM 기반 텍스트 분류 모델
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        num_layers=2,
        dropout=0.5,
        bidirectional=True
    ):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        # Embedding: [batch, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # LSTM: [batch, seq_len, hidden_dim * 2]
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # 마지막 hidden state 사용
        # Bidirectional이면 forward와 backward 연결
        if self.lstm.bidirectional:
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]
        
        # 분류: [batch, num_classes]
        logits = self.fc(hidden)
        
        return logits

# 모델 생성
model = TextClassifierLSTM(
    vocab_size=10000,
    embedding_dim=300,
    hidden_dim=256,
    num_classes=2,  # 긍정/부정
    num_layers=2,
    dropout=0.5,
    bidirectional=True
)

print("=== Text Classifier (LSTM) ===")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n총 파라미터: {total_params:,}")

# Forward pass 테스트
batch_size = 32
seq_length = 50
x_test = torch.randint(0, 10000, (batch_size, seq_length))
logits = model(x_test)

print(f"\n입력 shape: {x_test.shape}")
print(f"출력 shape: {logits.shape}")
print("\n✓ 모델 준비 완료!")
```

---
# 4. CNN 텍스트 분류
---

## 4.1 왜 텍스트에 CNN?

- **N-gram 패턴 포착**: "not good", "very bad"
- **빠른 학습**: RNN보다 병렬화 가능
- **짧은 텍스트에 효과적**: 리뷰, 트윗 등

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifierCNN(nn.Module):
    """
    CNN 기반 텍스트 분류 (Kim, 2014)
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_classes,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        dropout=0.5
    ):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 여러 크기의 Conv 필터
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(filter_sizes), num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len]
        """
        # Embedding: [batch, seq_len, embedding_dim]
        embedded = self.embedding(x)
        
        # Conv2d를 위해 차원 추가: [batch, 1, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)
        
        # 각 필터 적용 및 max pooling
        conved = [
            F.relu(conv(embedded)).squeeze(3)  # [batch, num_filters, seq_len - fs + 1]
            for conv in self.convs
        ]
        
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2)  # [batch, num_filters]
            for conv in conved
        ]
        
        # 연결: [batch, num_filters * len(filter_sizes)]
        cat = torch.cat(pooled, dim=1)
        
        # 분류
        logits = self.fc(cat)
        
        return logits

# 모델 생성
model_cnn = TextClassifierCNN(
    vocab_size=10000,
    embedding_dim=300,
    num_classes=2,
    num_filters=100,
    filter_sizes=[3, 4, 5],
    dropout=0.5
)

print("=== Text Classifier (CNN) ===")
print(model_cnn)

total_params = sum(p.numel() for p in model_cnn.parameters())
print(f"\n총 파라미터: {total_params:,}")

# 비교
print("\n=== LSTM vs CNN ===")
print("\nLSTM:")
print("  장점: 장기 의존성, 순서 정보")
print("  단점: 느린 학습, 순차 처리")
print("\nCNN:")
print("  장점: 빠른 학습, N-gram 패턴")
print("  단점: 제한된 문맥, 순서 정보 약함")
print("\n추천: 짧은 텍스트 → CNN, 긴 텍스트 → LSTM")
```

---
# 5. 데이터 로딩 및 학습
---

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    """
    텍스트 분류 데이터셋
    """
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 텍스트를 인덱스로 변환
        indices = self.vocab.encode(text)
        
        return torch.LongTensor(indices), torch.LongTensor([label])

# Collate function (배치 내 시퀀스 길이 맞추기)
def collate_fn(batch):
    """
    가변 길이 시퀀스를 패딩
    """
    texts, labels = zip(*batch)
    
    # 패딩 (가장 긴 시퀀스에 맞춤)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.cat(labels)
    
    return texts_padded, labels

# 학습 함수
def train_text_classifier(
    model, train_loader, val_loader,
    criterion, optimizer, device, num_epochs=10
):
    """
    텍스트 분류 모델 학습
    """
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # 학습
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # 검증
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                logits = model(texts)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_text_model.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    print(f"\n✓ 학습 완료! Best Val Acc: {best_val_acc:.4f}")

print("학습 함수 준비 완료!")
```

---
# 6. 추론 및 해석
---

```python
import torch
import torch.nn.functional as F

class SentimentAnalyzer:
    """
    감성 분석 추론 클래스
    """
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.eval()
        
        self.labels = ['Negative', 'Positive']
    
    def predict(self, text):
        """
        단일 텍스트 감성 분석
        """
        # 전처리 및 인코딩
        indices = self.vocab.encode(text)
        input_tensor = torch.LongTensor(indices).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        # 결과
        predicted_idx = probs.argmax().item()
        predicted_label = self.labels[predicted_idx]
        confidence = probs[predicted_idx].item()
        
        return {
            'text': text,
            'sentiment': predicted_label,
            'confidence': confidence,
            'probabilities': {
                self.labels[i]: probs[i].item()
                for i in range(len(self.labels))
            }
        }
    
    def predict_batch(self, texts):
        """
        배치 추론
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

print("=== Sentiment Analyzer ===")
print("\n사용법:")
print("analyzer = SentimentAnalyzer(model, vocab, device)")
print("result = analyzer.predict('This movie is amazing!')")
print("\n출력:")
print("  - sentiment: Positive/Negative")
print("  - confidence: 0.0 ~ 1.0")
print("  - probabilities: 각 클래스별 확률")
```

---
# 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 텍스트 전처리
```python
1. 정제: 소문자, HTML/URL 제거
2. 토큰화: 단어 단위 분리
3. 어휘 구축: 단어 ↔ 인덱스
4. 인코딩: 텍스트 → 숫자 시퀀스
5. 패딩: 가변 길이 → 고정 길이
```

### 2. 단어 임베딩
- **One-hot**: 희소, 고차원, 유사도 없음
- **Embedding**: 밀집, 저차원, 의미 반영
- **Pre-trained**: GloVe, Word2Vec, FastText

### 3. 모델 아키텍처
**LSTM**:
- 장기 의존성
- 순서 정보
- 긴 텍스트

**CNN**:
- N-gram 패턴
- 빠른 학습
- 짧은 텍스트

### 4. 학습 팁
- **Gradient clipping**: 필수
- **Dropout**: 0.5 권장
- **Bidirectional LSTM**: 성능 향상
- **Pre-trained embedding**: 데이터 적을 때

### 5. 평가
- **Accuracy**: 균형 데이터셋
- **F1 Score**: 불균형 데이터셋
- **Confusion Matrix**: 오분류 패턴

### 실전 체크리스트
```python
1. 데이터 정제 (HTML, URL, 특수문자)
2. 어휘 구축 (min_freq=2~5)
3. 임베딩 차원 (100~300)
4. LSTM hidden (128~512)
5. Dropout (0.3~0.5)
6. Gradient clipping (1.0)
7. 배치 크기 (32~128)
```

### 백엔드 배포
```python
1. Vocabulary 저장 (pickle)
2. 모델 저장 (state_dict)
3. 추론 클래스 구현
4. 배치 추론 지원
5. 캐싱 (자주 쓰는 단어)
```

### 응용 분야
- 감성 분석 (리뷰, SNS)
- 스팸 필터링 (이메일, 댓글)
- 주제 분류 (뉴스, 문서)
- 의도 파악 (챗봇, 고객 문의)

다음 단원에서는 **빅데이터 처리**를 배워보겠습니다!
