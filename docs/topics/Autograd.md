GPT 정리

# PyTorch의 자동미분과 역전파가 의미 체계적 분석에서 어떤 것을 의미하는가?

1.Conceptual Layer(의미 체계적 분석)

기계학습의 핵심 목표는 단순합니다.

> 모델의 예측이 실제 값과 얼마나 다른지를 측정하고
> 그 차이를 줄이기 위해 내부 매개변수(가중치)를 조정하는 것.

여기에서 등장하는 두 개의 핵심 개념이
1. 스칼라 손실(scalar loss)
2. 미분값(gradient)

## 스칼라 손실 L
- 의미
모델이 출련한 결과값이 정답과 얼마나 멀리 떨어져 있는지를 나타내는 단 하나의 숫자.
즉, 오차(error)를 수치화한 척도.

> Q: 정답은 누가 정의하고 어떻게 정의하는가?

추상화:
결과값에 대한 손실이란 현재의 오류량을 뜻하며, 스칼라 손실이란 다차원의 오류를 한 점수로 압축한 것.

미분값 (gradient)

- 의미:
손실이 줄어들게 하려면 각 변수(가중치, 입력 등)을 어떤 방향으로 얼마나 바꿔야 하는가를 나타내는값. 
즉, 손실에 대한 민감도(sensitivity)

- 언어적 추상화:
모델이 잘못된 이유가 어디에 있는가를 알려주는 지도.
각 가중치가 손실에 기여한 정도를 측정하여 그걸 토대로 어디를 얼마나 수정해야 더 정확해지는가를 계산하는 과정.

- 역전파(backpropagation)이란?
이 민감도를 출력에서 입력 방향으로 전파하여,
모든 변수들이 손실에 미친 영향을 각각 계산해 주는 연쇄적 미분 과정.
즉, 누가 얼마나 잘못했는가를 추적하는 과정이다.

> 변수와의 관계등을 분석하여 각각의 관계성을 가지고 역으로 가중치를 찾아 나가는 (미분값) 과정인가? 연쇄법칙으로 각 변수와 손실량의 관계를 표현


# 🔷 2. 수학적 분석 (Mathematical Layer)

> 아래 내용은 나의 코드 작성은 API의 개념적 구성에 의존하고 pytorch에 의해 자동 계산 될 것이므로 넘어가도록한다. 

## (1) 손실의 정의

예를 들어, 회귀 문제에서 예측값이 (y_{\text{pred}}), 실제값이 (y_{\text{true}})라면

[
L = \frac{1}{2}(y_{\text{pred}} - y_{\text{true}})^2
]

여기서 (L)은 오차 제곱합을 스칼라로 표현한 손실이다.

---

## (2) 미분(gradient)의 정의

가중치 (w), 입력 (x), 편향 (b)를 갖는 간단한 선형 모델:

[
y_{\text{pred}} = w x + b
]

그럼 손실은:

[
L = \frac{1}{2}(wx + b - y_{\text{true}})^2
]

이때 각 변수에 대한 기울기는 다음과 같다.

[
\frac{\partial L}{\partial w} = (wx + b - y_{\text{true}})\cdot x
]
[
\frac{\partial L}{\partial b} = (wx + b - y_{\text{true}})
]
[
\frac{\partial L}{\partial x} = (wx + b - y_{\text{true}})\cdot w
]

이 식들은 **연쇄법칙(chain rule)** 으로 도출된 결과이며,
손실이 각 변수에 얼마나 민감하게 반응하는지를 정량적으로 보여준다.

---

## (3) 의미적 연결

이 식을 자연어로 번역하면 다음과 같다.

> * (\frac{\partial L}{\partial w}): “가중치 (w)가 조금만 바뀌어도 손실이 얼마나 변하는가”
> * (\frac{\partial L}{\partial b}): “편향이 손실에 주는 영향”
> * (\frac{\partial L}{\partial x}): “입력 값이 손실에 주는 영향”

이 미분값들이 바로 **역전파로 계산되는 기울기(gradient)** 들이다.

그리고 경사하강법(gradient descent)은 이 기울기를 사용해:

[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
]
[
b_{\text{new}} = b_{\text{old}} - \eta \frac{\partial L}{\partial b}
]

처럼 손실이 감소하는 방향으로 매개변수를 갱신한다.
여기서 (\eta)는 학습률(learning rate)이다.

---

# Computational Lyaer
## PyTorch에서의 구현적 분석

PyTorch의 autograd 시스템은 이 수학적 과정을 자동으로 구현해주는 동적 미분 엔진이다. 

즉 "수식을 손으로 풀 필요 없이, 코드 흐름을 따라 그래프를 만들고 역전파를 자동 수행한다. "

```
import torch

# 1. 입력값, 가중치, 편향 설정
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3,0. requires_grad=True)
b = torch.tensor(1.0. requires_grad=True)

# 2. 예측과 손실 계산
y_true = torch.tensor(10.0)
y_pred = w * x + b # 모델 예측
L = 0.5 * (y_pred - y_true) ** 2 # 스칼라 손실

L.backward()

print(f"Loss: {L.item(): .2f}")
print(f"dL/dw: {w.grad.item():.2f}")
print(f"dL/db: {b.grad.item():.2f}")
print(f"dL/dx: {x.grad.item():.2f}")

```

3. 동작 구조
- requires_grad=True: autograd가 이 변수의 연산을 추적하도록 지시한다. 
- 연산 수행 시: "계산 그래프(computational graph) 자동 생성"
- L.backward(): 연쇄법칙으로 그래프를 거꾸로 순회하면서 각 변수의 가중치 또는 미분값을 계산한다.
- .grad에 결과 저정한다.

4. 요약 표

| 구분                    | 수학적 표현                                                | 의미 체계적 설명                                   | PyTorch 내 대응                  |
| --------------------- | ----------------------------------------------------- | ------------------------------------------- | ----------------------------- |
| 스칼라 손실 (L)            | (L = f(y_{\text{pred}}, y_{\text{true}}))             | 결과의 오차량, “내가 얼마나 틀렸는가”                      | `L = loss_fn(y_pred, y_true)` |
| 미분값 (Gradient)        | (\frac{\partial L}{\partial w}) 등                     | 손실을 줄이기 위해 각 변수(가중치)를 얼마나, 어느 방향으로 조정해야 하는가 | `L.backward()` 후 `.grad` 값    |
| 역전파 (Backpropagation) | 연쇄법칙으로 ∂L/∂θ 계산                                       | 오류의 원인을 거꾸로 추적해 변수별 책임을 계산                  | autograd 엔진이 그래프를 따라 자동 계산    |
| 학습(Weight Update)     | (w \leftarrow w - \eta \frac{\partial L}{\partial w}) | 손실을 줄이는 방향으로 모델이 스스로 수정                     | `optimizer.step()`            |


Q. 그렇다면 결과값에 대한 조정에 대한 적분이 의미하는 것은 무엇인가? 

적분이란 미분의 총합이다. 
즉 의미 쳬계상 "그 조정들이 쌓여 모델이 어떻게 변화했나"를 의미한다. 

미분은 순간 변화율로서 "지금 어떻게 조정해야하나?"를 의미한다면, 
적분은 변화의 누적합으로서 "그 조정들이 쌓여 모델이 어떻게 변화했나를 의미힌다."

물리적 비유 (가장 직관적)
위치: 가중치 값, 모델의 상태
속도: 미분 가중치 변화율
거리: 학습의 누적 결과

- PyTorch 관저에서의 대응
PyTorch는 미분(gradient) 만 자동 계산합니다.
하지만 훈련 과정 전체(optimizer step)는 사실상 적분의 근사적 구현이다. 

```
for step range(T):
    loss.backword()
    optimizer.step()
    optimizer.zero_grad()

```

> "학습 루프 전체가 미분의 적분이다!"

그 움직임이 쌓여 만들어진 학습된 모델이다.