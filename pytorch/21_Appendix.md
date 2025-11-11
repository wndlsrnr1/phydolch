# 부록 (Appendix)

이 부록은 교과서 전반에 걸쳐 사용된 핵심 개념을 요약하고, 추가 정보를 제공합니다.

## 내용
- **수학 공식**: 핵심 수학 공식 요약
- **PyTorch API 레퍼런스**: 자주 사용하는 API
- **흔한 에러 해결**: 문제 해결 가이드
- **추가 학습 자료**: 더 깊이 배우기

---
# 1. 수학 공식
---

## 1.1 미적분

**편미분 (Partial Derivative)**:
\[ \frac{\partial f(x, y)}{\partial x} \]

**체인룰 (Chain Rule)**:
\[ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} \]

**Gradient Descent**:
\[ \theta_{t+1} = \theta_t - \eta \nabla J(\theta_t) \]

## 1.2 선형대수

**행렬 곱 (Matrix Multiplication)**:
\[ C_{ij} = \sum_{k=1}^n A_{ik} B_{kj} \]

## 1.3 확률 및 통계

**정규분포 (Normal Distribution)**:
\[ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \]

**최대우도추정 (Maximum Likelihood Estimation, MLE)**:
\[ \hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta|x) = \arg\max_{\theta} \sum_{i=1}^n \log P(x_i|\theta) \]

## 1.4 손실 함수

**MSE (Mean Squared Error)**:
\[ \text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 \]

**Cross-Entropy**:
\[ \text{CE} = -\sum_{i=1}^C y_i \log(\hat{y}_i) \]

---
# 2. PyTorch API 레퍼런스
---

## `torch.Tensor`
- `.shape`, `.dtype`, `.device`
- `.view()`, `.reshape()`
- `.to(device)`
- `.detach()`
- `.numpy()`

## `torch.nn.Module`
- `__init__()`, `forward()`
- `.parameters()`
- `.state_dict()`
- `.load_state_dict()`
- `.train()`, `.eval()`

## `torch.nn` Layers
- `nn.Linear(in, out)`
- `nn.Conv2d(in_ch, out_ch, kernel_size)`
- `nn.MaxPool2d(kernel_size)`
- `nn.ReLU()`, `nn.Sigmoid()`, `nn.Softmax()`
- `nn.Dropout(p)`
- `nn.BatchNorm2d(num_features)`
- `nn.LSTM(input, hidden)`
- `nn.Embedding(vocab, dim)`

## `torch.optim`
- `optim.Adam(params, lr)`
- `optim.SGD(params, lr, momentum)`
- `optimizer.zero_grad()`
- `optimizer.step()`

## `torch.nn.functional`
- `F.relu()`
- `F.softmax()`
- `F.cross_entropy()`

## `torch.utils.data`
- `Dataset`
- `DataLoader(dataset, batch_size, shuffle)`

---
# 3. 흔한 에러 해결
---

| 에러 메시지 | 원인 | 해결책 |
|---|---|---|
| `RuntimeError: CUDA out of memory.` | GPU 메모리 부족 | 배치 크기 감소, AMP, Grad Accumulation |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied.` | 행렬 곱셈 shape 불일치 | `.view()`, `.reshape()`, shape 확인 |
| `RuntimeError: Expected object of scalar type Long but got Float for argument #2 'target'` | CrossEntropy의 target이 float | `target.long()` 또는 `target.to(torch.long)` |
| `ValueError: Expected input batch_size (...) to match target batch_size (...)` | 입력과 타겟의 배치 크기 불일치 | `DataLoader`의 `drop_last=True` |
| `loss is NaN` | Gradient explosion, log(0) | LR 감소, Gradient clipping, 데이터 정규화 |
| `element 0 of tensors does not require grad and does not have a grad_fn` | `.backward()` 호출 시 grad 없음 | `requires_grad=True` 설정, `.detach()` 확인 |
| `Expected all tensors to be on the same device` | 텐서들이 다른 device에 있음 | `tensor.to(device)`로 통일 |
| `TypeError: 'NoneType' object is not iterable` | `forward()`에서 return 없음 | `forward()` 함수에 `return` 추가 |

---
# 4. 추가 학습 자료
---

## 공식 문서
- **PyTorch 공식 튜토리얼**: https://pytorch.org/tutorials/
- **PyTorch API 문서**: https://pytorch.org/docs/stable/index.html

## 추천 강의
- **Stanford CS231n**: Convolutional Neural Networks for Visual Recognition
- **Stanford CS224n**: Natural Language Processing with Deep Learning
- **fast.ai**: Practical Deep Learning for Coders

## 주요 논문
- **ResNet**: Deep Residual Learning for Image Recognition
- **Adam**: A Method for Stochastic Optimization
- **Attention Is All You Need**: The Transformer paper
- **BERT**: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 커뮤니티
- **PyTorch Forums**: https://discuss.pytorch.org/
- **Papers with Code**: https://paperswithcode.com/
