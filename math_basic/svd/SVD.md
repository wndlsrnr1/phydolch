---
# 📘 SVD 총정리 (행렬 = 기하적 변환으로 보기)
---
## 1. 행렬과 열벡터의 의미
행렬 $A=\begin{bmatrix}2 & 1\\1 & 3\end{bmatrix}$ 는 두 개의 **열벡터**를 가집니다.

$a_1 = \begin{bmatrix}2\\1\end{bmatrix},\quad a_2 = \begin{bmatrix}1\\3\end{bmatrix}$

임의의 벡터 $x=[x_1,x_2]^\top$ 에 대해

$Ax = x_1 a_1 + x_2 a_2$

즉, **행렬곱은 열벡터들의 선형결합(linear combination)** 입니다.

이 벡터들이 만드는 모든 조합(스팬, span)이 바로 **열공간(column space)** 이며, 이번 경우는 $\mathbb{R}^2$ 전체를 채웁니다.

📊 **Column space (연속 평행사변형으로 표현)**
![[svd_column_space_shaded.png]]
---
## 2. 내적과 투영(“그림자”)
두 벡터 $a,b$ 가 있을 때,

$a\cdot b = |a||b|\cos\theta = a^\top b$

이는 **길이와 각도의 곱**이므로, 벡터들의 “정렬도”를 측정합니다.

$a$ 를 $b$ 방향으로 내리면

$\operatorname{proj}_b(a)=\dfrac{a^\top b}{b^\top b} b$

즉, $a$ 가 $b$ 위에 만드는 **그림자**입니다.

📊 **Projection 도해 (최소거리 해로서의 투영)**
![[svd_projection_opt.png]]
---
## 3. SVD의 정의
$A = U\,\Sigma\,V^\top$

- $V$: **입력 공간의 직교기저** (오른쪽 특이벡터, input axes)
- $\Sigma$: 각 축을 $\sigma_i$ 배로 늘리거나 줄이는 **축별 스케일링**
- $U$: **출력 공간의 직교기저** (왼쪽 특이벡터, output axes)

즉, **입력을 주축에 맞추고 → 신장비를 곱하고 → 출력 주축으로 정렬**하는 과정입니다.
---
## 4. 단계별 변환 과정
### Stage 0: 표준좌표
단위벡터 $e_1=[1,0], e_2=[0,1]$ 를 기준으로 시작합니다.

📊 Stage 0 — Standard basis
![[svd_stage0.png]]
---
### Stage 1: $V^\top$
입력 벡터들을 **입력 주축** $(v_1,v_2)$ 로 맞춥니다. 이 과정은 **회전/반사**에 해당하며, 길이와 각도를 그대로 보존합니다.

📊 Stage 1 — $V^T$
![[svd_stage1_Vt.png]]
---
### Stage 2: $\Sigma$
맞춰진 주축 방향에 대해 $(\sigma_1,\sigma_2)$ 만큼 **축별 스케일**을 합니다. 이때 스케일이 표준좌표에서 비스듬해 보이는 이유는, 축이 이미 $V^\top$ 에서 회전했기 때문입니다.

📊 Stage 2 — $\Sigma$
![[svd_stage2_S.png]]
---
### Stage 3: $U$
마지막으로 스케일된 결과를 **출력 주축** $(u_1,u_2)$ 로 회전/정렬합니다.

📊 Stage 3 — $U$
![[svd_stage3_U.png]]
---
## 5. 전체 효과: 원 → 타원
단위원은 $A$ 에 의해 **타원**으로 변합니다.

- 장축: $\sigma_1 u_1$
- 단축: $\sigma_2 u_2$

📊 $A$ maps unit circle to ellipse
![[svd_A_ellipse.png]]
---
## 6. 왜 $U, V$ 가 주축인가
$A^\top A = V\Lambda V^\top$

- $\Lambda=$ 고유값 대각행렬, $\lambda_i = \sigma_i^2$
- 고유벡터 $v_i$ 는 입력 주축, 고유값 제곱근 $\sigma_i$ 는 신장비
- $Av_i = \sigma_i u_i$ 로 정의하면 $u_i$ 가 출력 주축

📊 $A^T A$ eigenvectors and level set
![[svd_ATA_eigs.png]]
---
## 7. 추가: 랭크-1 근사
$A_1 = \sigma_1 u_1 v_1^\top$

이 근사는 “데이터 압축”과 PCA와 직접 연결됩니다.

📊 Rank-1 approximation
![[svd_rank1.png]]
---
## 8. 이번 예시의 실제 수치
- $U \approx \begin{bmatrix}-0.526 & -0.851\\-0.851 & 0.526\end{bmatrix}$
- $\Sigma = \operatorname{diag}(3.618, 1.382)$
- $V \approx \begin{bmatrix}-0.526 & -0.851\\-0.851 & 0.526\end{bmatrix}$
- $\det(U) \approx -1,\ \det(V) = -1$ → 반사 포함 직교변환

따라서,

- 입력 주축 $(v_1,v_2)$ 방향으로 맞추고,
- $(\sigma_1,\sigma_2)$ 만큼 신장한 뒤,
- 출력 주축 $(u_1,u_2)$ 로 정렬하여,
- 최종적으로 단위원을 장·단축 $(3.618, 1.382)$ 인 타원으로 보냅니다.
---
## 🎯 최종 요약
- **행렬곱 = 열벡터들의 선형결합**
- **내적 = 정렬도, 투영 = 그림자**
- **SVD = 회전/반사($V^\top$) → 축별 스케일($\Sigma$) → 회전/반사($U$)**
- **결과 = 단위원이 장·단축이 특이값, 방향이 $u_1,u_2$ 인 타원으로 변환**
