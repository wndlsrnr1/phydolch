# 00a. 미적분 기초 (Calculus Foundations)

이번 단원에서는 딥러닝에 필요한 **미적분 기초**를 배워보겠습니다.

## 학습 목표
- 미분 (Derivative)의 기하학적 의미 (접선, 기울기)
- 편미분 (Partial Derivative)과 기울기 (Gradient)
- 체인룰 (Chain Rule)의 직관적 이해
- 경사하강법 (Gradient Descent)의 수학적 원리
- 학습률 (Learning Rate)의 역할

## 왜 이 단원이 필요한가?

딥러닝의 핵심은 **최적화 (Optimization)**입니다. 모델이 학습한다는 것은 손실 함수 (Loss Function)를 최소화하는 파라미터를 찾는 것이고, 이를 위해서는 미적분이 필수입니다.

- **미분 (Derivative)**: 함수가 얼마나 빠르게 변하는가?
- **기울기 (Gradient)**: 다변수 함수에서 가장 가파른 방향
- **경사하강법 (Gradient Descent)**: 손실을 줄이는 방향으로 파라미터 업데이트
- **체인룰 (Chain Rule)**: 복잡한 함수의 미분을 자동으로 계산 (자동 미분, Autograd의 핵심)

이 단원을 이해하면 `loss.backward()`가 무엇을 하는지 완전히 이해할 수 있습니다.

## 이 단원을 배우기 전에

**고등학교 수학 복습**: 함수, 그래프, 기울기 개념을 기억하시나요?

## 이 단원 다음에는

**다음 단원 (00b)**: 미적분을 배웠으니, 확률과 통계 (Probability & Statistics)의 기초를 배워봅시다.

---
# 1. 직관적 이해 (Why)
---

## 1.1 미분이란 무엇인가?

### 실생활 비유: 자동차의 속도

- **위치 함수** $s(t)$: 시간 $t$에서 자동차의 위치
- **속도**: 위치가 시간에 따라 얼마나 빠르게 변하는가?
- **미분 (Derivative)** $s'(t)$ 또는 $\frac{ds}{dt}$: 순간 속도

### 기하학적 의미

함수 $f(x)$의 미분 $f'(x)$는:
- 점 $(x, f(x))$에서 그래프에 그은 **접선 (Tangent Line)의 기울기**
- $x$가 조금 변할 때 $f(x)$가 얼마나 변하는지

#### 접선이란?
곡선 위 한 점에서 **곡선과 딱 한 점에서만 만나는 직선**입니다. 그 점 근처에서 곡선의 기울기를 가장 정확히 나타냅니다. 평면처럼 먼저 접선(기울기)을 보고 얼마나 가파른지 판단합니다.

### 왜 딥러닝에 필요한가?

신경망 학습의 목표:
1. 손실 함수 (Loss Function) $L(w)$를 최소화하는 가중치 $w$ 찾기
2. $\frac{dL}{dw}$를 계산하여 손실이 증가하는 방향 파악
3. 반대 방향으로 $w$를 조금씩 이동 → **경사하강법 (Gradient Descent)**

---
# 2. 수학적 기초 (What)
---

## 2.1 미분의 정의 (고등학교 복습)

### 단계별 이해

#### Step 1: 평균 변화율

구간 $[x, x+h]$에서 함수 $f$의 평균 변화율 (Average Rate of Change):

$$\frac{f(x+h) - f(x)}{h}$$

**의미**: 두 점 사이의 **평균 기울기** (직선의 기울기)

**예시**: $f(x) = x^2$에서 $x=1, h=1$일 때
- $f(1) = 1$
- $f(2) = 4$
- 평균 변화율 = $\frac{4-1}{2-1} = 3$

#### Step 2: h를 점점 작게 만들기

$h=1$ → 평균 변화율 = $3$  
$h=0.5$ → 평균 변화율 = $2.5$  
$h=0.1$ → 평균 변화율 = $2.1$  
$h=0.01$ → 평균 변화율 = $2.01$

**관찰**: $h$가 작아질수록 평균 변화율이 **2에 가까워짐**

#### Step 3: 순간 변화율 (순간 변화율, Instantaneous Rate of Change)

$h$를 0에 가까이 보내면:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

이것이 **미분 (Derivative)**의 정의입니다.

**직관**: **한 점에서의 순간적인 기울기** (접선의 기울기)

**예시**: $f(x) = x^2$에서 $f'(1) = 2$ (x=1에서의 접선 기울기)

### 기본 미분 공식 (고등학교 복습)

| 함수 $f(x)$ | 미분 $f'(x)$ |
|------------|-------------|
| $c$ (상수) | $0$ |
| $x$ | $1$ |
| $x^2$ | $2x$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

## 2.2 편미분 (Partial Derivative)

### Step 0: 기호 정리
- $L(w_1, w_2, \ldots, w_n)$: 여러 파라미터에 의존하는 손실 함수
- $\frac{\partial L}{\partial w_i}$: $w_i$만 변화시켰을 때의 변화율
- $\nabla f$: 모든 편미분을 모은 기울기 벡터
- $\mathbf{v}$: 임의의 단위 방향 벡터

### Step 1: 문제 상황
신경망에서는 손실 함수가 여러 변수에 의존합니다:

$$L(w_1, w_2, w_3, \ldots, w_n)$$

**질문**: $w_1$을 조금 바꾸면 손실이 얼마나 변하나?

### Step 2: 하나씩만 움직여 보기
**편미분 (Partial Derivative)**: 하나의 변수만 변화시키고 나머지는 **상수로 취급**

$$\frac{\partial L}{\partial w_1} = \lim_{h \to 0} \frac{L(w_1+h, w_2, w_3, \ldots) - L(w_1, w_2, w_3, \ldots)}{h}$$

**의미**: $w_2, w_3, \ldots$는 그대로 두고, $w_1$만 변화시켰을 때의 변화율

### Step 3: 직관 만들기
- 일반 미분: 1차원 함수에서의 기울기
- 편미분: 다차원 공간에서 **한 방향으로만** 이동했을 때의 변화율
- **비유**: 산에서 "동쪽으로만" 갔을 때 높이가 얼마나 변하는가?

### Step 4: 구체적 예시
함수 $f(x, y) = x^2 + 3xy + y^2$의 편미분:

- $\frac{\partial f}{\partial x} = 2x + 3y$ (y를 상수로 취급)
- $\frac{\partial f}{\partial y} = 3x + 2y$ (x를 상수로 취급)

### Step 5: 기울기 (Gradient) 벡터
모든 편미분을 모아 벡터로 만든 것:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}$$

**의미**: 기울기는 함수가 **가장 빠르게 증가하는 방향**을 가리킵니다.

### Step 6: 왜 가장 빠른가?
**방향도함수 (Directional Derivative)** 정의:  
임의의 단위벡터 $\mathbf{v} = [v_x, v_y]$ 방향으로의 변화율:
$$D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v}$$

**내적의 성질**:
$$\nabla f \cdot \mathbf{v} = |\nabla f| \cdot |\mathbf{v}| \cdot \cos(\theta) = |\nabla f| \cdot \cos(\theta)$$

여기서 $\theta$는 $\nabla f$와 $\mathbf{v}$ 사이의 각도입니다.

- $\cos(\theta) = 1$일 때 최댓값 → $\theta = 0°$ (같은 방향)
- 최댓값 = $|\nabla f|$

**결론**: $\nabla f$ 방향으로 움직일 때 함수가 가장 빨리 증가합니다. 딥러닝에서는 그 반대 방향으로 움직여 손실을 줄입니다.

## 2.3 체인룰 (Chain Rule)

### Step 0: 기호 정리
- $y = f(g(x))$: 외부 함수 $f$, 내부 함수 $g$로 구성된 합성 함수
- $\frac{dy}{dx}$: $x$가 변할 때 최종 출력 $y$의 변화율
- $\frac{dy}{dg}$, $\frac{dg}{dx}$: 각 단계별 변화율

### Step 1: 문제 상황
합성 함수 $y = f(g(x))$가 있습니다. 예를 들어:
- $g(x) = x^2 + 1$ (내부 함수)
- $f(u) = u^3$ (외부 함수)
- 따라서 $y = (x^2 + 1)^3$

**질문**: $x$가 조금 변하면 $y$가 얼마나 변하나?

### Step 2: 체인룰의 직관
$x$가 변하면 → $g$가 변하고 → $f$가 변합니다.

**체인룰**:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**의미**: 
- $\frac{dg}{dx}$: $x$가 변할 때 $g$가 얼마나 변하는가
- $\frac{dy}{dg}$: $g$가 변할 때 $y$가 얼마나 변하는가
- **곱하면**: $x$가 변할 때 $y$가 얼마나 변하는가

### Step 3: 단계별 계산 (상징식)
**예시**: $y = (x^2 + 1)^3$

**Step 3-1**: 내부 함수를 별도 변수로 설정
- $g = x^2 + 1$
- $y = g^3$

**Step 3-2**: 각각을 미분
- $\frac{dg}{dx} = 2x$ (내부 함수의 미분)
- $\frac{dy}{dg} = 3g^2$ (외부 함수의 미분)

**Step 3-3**: 체인룰 적용
- $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = 3g^2 \cdot 2x$

**Step 3-4**: $g$를 원래 식으로 대입
- $\frac{dy}{dx} = 3(x^2+1)^2 \cdot 2x = 6x(x^2+1)^2$

### Step 4: 수치 예시로 확인
$x = 1$일 때:
- $g = 1^2 + 1 = 2$
- $\frac{dg}{dx} = 2x = 2$
- $\frac{dy}{dg} = 3g^2 = 12$
- $\frac{dy}{dx} = 12 \times 2 = 24$

직접 미분한 결과 $6x(x^2+1)^2$에 $x=1$을 대입해도 $24$가 됩니다.  
→ 체인룰 계산이 맞다는 것을 숫자로 확인했습니다.

### Step 5: 신경망에서의 체인룰
신경망은 여러 층의 합성 함수입니다:

$$L = f_n(f_{n-1}(\cdots f_2(f_1(x)) \cdots))$$

체인룰을 반복 적용하면:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial f_n} \cdot \frac{\partial f_n}{\partial f_{n-1}} \cdots \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial w_1}$$

이것이 **역전파 (Backpropagation)**의 수학적 원리입니다!

## 2.4 Gradient Descent의 수학적 원리

### Step 0: 준비 – 무한급수와 근사 이해
- **무한급수**: 항을 무한히 더한 값 (예: $1 + 2 + 3 + \cdots$)
- **근사**: 정확한 값을 구하기 어렵다면 가까운 값을 계산해 활용
- **핵심 질문**: 복잡한 손실 함수 $L(w)$를 현재 위치 근처의 정보만으로 어떻게 예측할까?

### Step 1: 테일러 전개 맛보기
테일러 전개는 함수를 한 점 $w_0$ 주변에서 다항식(무한급수)으로 표현하는 방법입니다.

$$L(w) = L(w_0) + L'(w_0)(w-w_0) + \frac{L''(w_0)}{2}(w-w_0)^2 + \frac{L'''(w_0)}{6}(w-w_0)^3 + \cdots$$

- **0차 항**: 현재 함수값 $L(w_0)$
- **1차 항**: 기울기 × 거리 $L'(w_0)(w-w_0)$ (접선)
- **2차 이상**: 곡률에 대한 보정 항들

### Step 2: 1차 근사로 단순화
경사하강법에서는 작은 이동만 다루므로 **1차 항까지만** 사용합니다.

$$L(w) \approx L(w_0) + \nabla L(w_0) \cdot (w - w_0)$$

- 곡선을 한 점에서의 **접선(선형 함수)**으로 바꿔 계산합니다.
- $w$가 $w_0$와 가까울수록 이 근사는 더 정확합니다.

### Step 3: 경사하강법 공식 유도
현재 위치를 $w$라 두고, 반대 방향으로 조금 이동: $w_{\text{new}} = w - \eta \nabla L(w)$

테일러 전개를 적용하면:

$$L(w_{\text{new}}) \approx L(w) + \nabla L(w) \cdot (w_{\text{new}} - w)$$
$$= L(w) + \nabla L(w) \cdot (-\eta \nabla L(w))$$
$$= L(w) - \eta \|\nabla L(w)\|^2$$

- $\|\nabla L(w)\|^2 \ge 0$ 이므로 $L(w_{\text{new}}) \le L(w)$ (단, $\eta$가 충분히 작을 때)
- **결론**: 기울기의 반대 방향으로 가면 손실이 줄어듭니다.

### Step 4: 학습률 (Learning Rate)의 역할
- $\eta$가 **너무 크면**: 1차 근사가 깨져 발산 가능 (overshooting)
- $\eta$가 **너무 작으면**: 한 번에 이동량이 작아 수렴이 매우 느림
- 적절한 $\eta$: 손실을 안정적으로, 빠르게 줄이는 균형점

### Step 5: 1차원 예시로 감각 잡기
함수 $L(w) = w^2$를 최소화한다고 가정합니다.

1. 초기값 $w_0 = 5$
2. 기울기 $\frac{dL}{dw} = 2w$
3. 학습률 $\eta = 0.1$이라면 업데이트:
   $$w_1 = w_0 - 0.1 \cdot (2 \times 5) = 4$$
4. 반복하면 $w$는 0(최솟값)으로 천천히 접근합니다.

### Step 6: 요약
- 테일러 전개로 "앞으로 이동했을 때의 손실"을 예측한다.
- 기울기의 **반대 방향**으로 이동하면 손실이 감소한다.
- 학습률은 이동 폭을 조절해 안정적 수렴을 돕는다.

### Step 7: 왜 작은 이동만 믿을 수 있을까?
- 테일러 전개는 현재 위치의 함수값과 기울기로 작은 이동을 근사합니다.
- 1차 항만 남기면 $L(w_{\text{new}}) \approx L(w) - \eta \|\nabla L(w)\|^2$ 를 얻습니다.
- 근사가 정확하려면 $\eta$가 작아야 하며, 너무 크면 발산 위험이 커집니다.





### Step 8: 직관 요약
- `loss.backward()`는 현재 기울기 $\nabla L(w)$를 계산합니다.
- `optimizer.step()`은 $w - \eta \nabla L(w)$로 이동해 손실을 줄입니다.
- 작은 $\eta$로 여러 번 이동하면 손실이 안정적으로 감소합니다.

```python
# 체인룰 단계별 예제
import numpy as np
import matplotlib.pyplot as plt

# 예제: y = (x^2 + 1)^3
x = np.linspace(-2, 2, 100)

# 내부 함수
g = x**2 + 1

# 외부 함수 (y = g^3)
y = g**3

# 미분 계산
dg_dx = 2*x  # 내부 함수의 미분
dy_dg = 3*g**2  # 외부 함수의 미분
dy_dx_chain = dy_dg * dg_dx  # 체인룰

# 직접 미분한 결과 (검증)
dy_dx_direct = 6*x*(x**2 + 1)**2

plt.figure(figsize=(15, 5))

# 1) 함수들
plt.subplot(1, 3, 1)
plt.plot(x, g, 'b-', linewidth=2, label='$g(x) = x^2 + 1$ (내부)')
plt.plot(x, y, 'r-', linewidth=2, label='$y = g^3 = (x^2+1)^3$ (외부)')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('합성 함수')
plt.legend()

# 2) 미분들
plt.subplot(1, 3, 2)
plt.plot(x, dg_dx, 'b--', linewidth=2, label='$\\frac{dg}{dx} = 2x$')
plt.plot(x, dy_dg, 'g--', linewidth=2, label='$\\frac{dy}{dg} = 3g^2$')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('미분값')
plt.title('각 단계의 미분')
plt.legend()

# 3) 체인룰 결과
plt.subplot(1, 3, 3)
plt.plot(x, dy_dx_chain, 'r-', linewidth=2, label='체인룰: $\\frac{dy}{dg} \\cdot \\frac{dg}{dx}$')
plt.plot(x, dy_dx_direct, 'k--', linewidth=1, alpha=0.5, label='직접 계산 (검증)')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('$\\frac{dy}{dx}$')
plt.title('체인룰 결과')
plt.legend()

plt.tight_layout()
plt.show()

print("체인룰 검증:")
print(f"- 체인룰 결과와 직접 계산 결과가 일치: {np.allclose(dy_dx_chain, dy_dx_direct)}")
print("\n딥러닝 연결:")
print("- 신경망은 여러 층의 합성 함수: L = fₙ(fₙ₋₁(...f₁(x)...))")
print("- 역전파 (Backpropagation)는 체인룰을 반복 적용하여 모든 파라미터의 gradient 계산")
```

---
# 3. PyTorch 구현 (How)
---

## 3.1 미분 계산 예제

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Step 1: 함수 f(x) = x^2의 미분 시각화
x = np.linspace(-3, 3, 100)
y = x**2
y_prime = 2*x  # 미분

plt.figure(figsize=(15, 5))

# 원함수와 접선
ax1 = plt.subplot(1, 3, 1)
ax1.plot(x, y, 'b-', linewidth=2, label='$f(x) = x^2$')

# 여러 점에서 접선 표시
for x_point in [-2, -1, 0, 1, 2]:
    y_point = x_point**2
    slope = 2 * x_point
    tangent_x = np.linspace(x_point - 1, x_point + 1, 10)
    tangent_y = slope * (tangent_x - x_point) + y_point
    ax1.plot(tangent_x, tangent_y, 'r--', linewidth=1.5, alpha=0.7)
    ax1.plot(x_point, y_point, 'ro', markersize=8)

ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('원함수와 접선들')
ax1.legend()
ax1.set_xlim(-3, 3)
ax1.set_ylim(-1, 10)

# 미분 (기울기)
ax2 = plt.subplot(1, 3, 2)
ax2.plot(x, y_prime, 'r-', linewidth=2, label="$f'(x) = 2x$")
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel("f'(x)")
ax2.set_title('미분 (기울기)')
ax2.legend()
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 평균 변화율과 순간 변화율 비교
ax3 = plt.subplot(1, 3, 3)
x_point = 1
h_values = [1.0, 0.5, 0.1, 0.01]
avg_rates = []
for h in h_values:
    avg_rate = ((x_point + h)**2 - x_point**2) / h
    avg_rates.append(avg_rate)

ax3.plot(h_values, avg_rates, 'bo-', linewidth=2, markersize=10, label='평균 변화율')
ax3.axhline(y=2, color='r', linestyle='--', linewidth=2, label='순간 변화율 (h→0) = 2')
ax3.set_xlabel('h (간격)')
ax3.set_ylabel('평균 변화율')
ax3.set_title('h를 작게 만들면 순간 변화율로 접근')
ax3.set_xlim(0, 1.1)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()  # h가 작아지는 방향

plt.tight_layout()
plt.show()

print("해석:")
print("- x=0에서 기울기=0 (최솟값)")
print("- x>0에서 기울기>0 (증가)")
print("- x<0에서 기울기<0 (감소)")
print("\n단계별 이해:")
print(f"- h=1.0일 때 평균 변화율 = {avg_rates[0]:.2f}")
print(f"- h=0.5일 때 평균 변화율 = {avg_rates[1]:.2f}")
print(f"- h=0.1일 때 평균 변화율 = {avg_rates[2]:.2f}")
print(f"- h→0일 때 순간 변화율 = 2.00 (접선의 기울기)")

# Step 2: Gradient 시각화 (2D 함수)
print("\n" + "="*60)
print("Gradient 시각화: 2차원 함수")
print("="*60)

fig = plt.figure(figsize=(14, 6))

# 2D 함수 정의: f(x,y) = x^2 + y^2
x_grid = np.linspace(-3, 3, 30)
y_grid = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x_grid, y_grid)
Z = X**2 + Y**2

# 등고선 그래프
ax1 = fig.add_subplot(1, 2, 1)
contour = ax1.contour(X, Y, Z, levels=10)
ax1.clabel(contour, inline=True, fontsize=8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('등고선: $f(x,y) = x^2 + y^2$')
ax1.grid(True, alpha=0.3)

# 한 점에서의 Gradient 벡터
point_x, point_y = 1.5, 1.5
grad_x = 2 * point_x  # ∂f/∂x = 2x
grad_y = 2 * point_y  # ∂f/∂y = 2y
ax1.arrow(point_x, point_y, grad_x*0.3, grad_y*0.3, 
          head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2, label='기울기 (Gradient)')
ax1.plot(point_x, point_y, 'ro', markersize=10)
ax1.legend()

# 3D 그래프
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('3D 그래프와 기울기 (Gradient) 방향')
fig.colorbar(surf, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.show()

print(f"\n점 ({point_x}, {point_y})에서의 기울기 (Gradient):")
print(f"  ∇f = [∂f/∂x, ∂f/∂y] = [{grad_x:.1f}, {grad_y:.1f}]")
print(f"의미: 이 방향으로 이동하면 함수가 가장 빠르게 증가함")
print(f"→ 경사하강법 (Gradient Descent)는 반대 방향(-∇f)으로 이동하여 최솟값을 찾음")
```

---
# 5. 핵심 요약
---

## 이번 단원에서 배운 내용

### 1. 미분 (Derivative)의 의미
- **기하학적**: 접선의 기울기
- **물리적**: 순간 변화율
- **딥러닝**: 손실이 파라미터에 얼마나 민감한가

### 2. 편미분 (Partial Derivative)과 기울기 (Gradient)
- **편미분**: 한 변수만 변화시킬 때의 변화율
- **기울기 (Gradient)**: 모든 편미분을 모은 벡터
- **의미**: 함수가 가장 빠르게 증가하는 방향

### 3. 체인룰 (Chain Rule)
- **합성 함수의 미분**: $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$
- **역전파 (Backpropagation)의 핵심**: 여러 층을 거슬러 올라가며 gradient 계산
- **PyTorch의 자동 미분 (Autograd)**: 체인룰을 자동으로 적용

### 4. 경사하강법 (Gradient Descent)
- **원리**: 손실이 감소하는 방향으로 파라미터 업데이트
- **공식**: $w_{\text{new}} = w_{\text{old}} - \eta \nabla L(w)$
- **학습률 (Learning Rate)**: 업데이트 크기 조절 (너무 크면 발산, 너무 작으면 느림)

### 5. 실전 연결
- `loss.backward()`: 체인룰로 모든 gradient 계산
- `optimizer.step()`: 경사하강법 (Gradient Descent) 수행
- `optimizer.zero_grad()`: 이전 gradient 초기화

## 다음 단원 미리보기

**다음 단원 (00b)**: 확률과 통계 (Probability & Statistics)의 기초

- 왜 손실 함수 (Loss Function)가 확률 분포 (Probability Distribution)와 연결되는가?
- CrossEntropy Loss의 통계적 의미
- 정규화와 통계의 관계

미적분을 이해했으니, 이제 확률론적 관점에서 머신러닝을 바라봅시다!
