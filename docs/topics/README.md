# PyTorch 학습 자료 - 단원별 정리

이 폴더에는 PyTorch 초보자를 위한 **11개의 단원별 학습 자료**가 포함되어 있습니다.

## 📚 단원 구성

### 00_Math_Foundations.ipynb ⭐ 신규
- 벡터와 행렬의 기하학적 의미
- 행렬곱과 선형결합
- 내적과 투영
- **SVD (Singular Value Decomposition)**
- PyTorch에서의 행렬 연산 실습
- 연습 문제 5개

### 01_Tensor_Basics.ipynb
- 텐서의 기본 개념
- 텐서 초기화 방법
- 데이터 타입
- CUDA 텐서
- 다차원 텐서 표현

### 02_Tensor_Operations.ipynb
- 텐서 연산 (덧셈, 곱셈, 행렬곱 등)
- 텐서 조작 (reshape, transpose 등)
- 브로드캐스팅
- 인덱싱과 슬라이싱

### 03_Autograd.ipynb
- 자동 미분의 개념
- Gradient의 의미
- 계산 그래프
- requires_grad와 backward()
- 학습 가능한 매개변수
- torch.no_grad() 활용

### 04_Data_Preparation.ipynb
- Dataset과 DataLoader
- 데이터 전처리 (transforms)
- 배치 처리 개념
- 데이터 시각화 (matplotlib)

### 05_Neural_Network_Structure.ipynb
- Layer, Module, Model의 개념
- torch.nn 패키지 소개
- 신경망을 복합 함수로 이해하기

### 06_Linear_Layer.ipynb
- nn.Linear의 동작 원리
- 가중치와 편향의 형태 및 의미
- 아핀 변환의 기하학적 의미
- 행렬곱과 내적의 관계
- Linear vs 일반 행렬곱 비교

### 07_Convolution_Layer.ipynb
- Conv2d의 개념과 동작 원리
- Linear Layer와의 차이점
- 커널, stride, padding, dilation의 의미
- 출력 크기 계산 공식
- 특징 추출기로서의 역할
- 파라미터별 효과 정리

### 08_Other_Layers.ipynb
- Pooling Layer (MaxPool, AvgPool)
- Non-linear Activations (ReLU, Sigmoid, Tanh)
- 비선형성의 필요성
- 신경망 종류 개요

### 09_Data_Labeling.ipynb ⭐ 신규
- 데이터 라벨링의 중요성과 목적
- 라벨링 유형별 방법론 (분류, 탐지, 세그멘테이션)
- 라벨링 도구 (CVAT, Label Studio 등)
- 라벨 품질 관리 및 검수 프로세스
- PyTorch Dataset 구현
- 연습 문제 5개

### 10_First_Project.ipynb ⭐ 신규
- **완전한 MNIST 프로젝트 파이프라인**
- 데이터 로드 및 탐색
- CNN 모델 정의
- 학습 루프 구현
- 성능 평가
- 오류 분석
- 실전 디버깅 팁

## 🎯 학습 목표

- 기하, 벡터, 행렬, 통계 배경지식 활용
- 머신러닝/딥러닝 초보자도 이해 가능한 설명
- 핵심 수식과 직관적 설명 제공
- 실행 가능한 코드 예제 포함
- **수학 기초부터 실전 프로젝트까지** 완전한 학습 경로

## 📖 학습 순서 (권장)

```
00. 수학 기초 (SVD, 선형대수)
    ↓
01. 텐서 기초
    ↓
02. 텐서 연산
    ↓
03. 자동 미분 (Autograd)
    ↓
04. 데이터 준비
    ↓
05. 신경망 구조
    ↓
06. 선형 레이어
    ↓
07. 컨볼루션 레이어
    ↓
08. 기타 레이어 (Pooling, Activation)
    ↓
09. 데이터 라벨링 (실전 스킬)
    ↓
10. 첫 프로젝트 (MNIST 완전 파이프라인)
```

## ⏱️ 예상 학습 시간

- 00. 수학 기초: 2-3시간
- 01-08. 기본 개념: 각 2-3시간 (총 16-24시간)
- 09. 데이터 라벨링: 1-2시간
- 10. 첫 프로젝트: 3-4시간
- **총 예상 시간: 22-33시간**

## ✅ 선수 지식 체크리스트

- [ ] 기본적인 Python 프로그래밍 능력
- [ ] 기하/벡터/행렬 개념 (고등학교 수학 수준)
- [ ] 기본 통계 개념
- [ ] NumPy 사용 경험 (권장)
- [ ] Jupyter Notebook 사용법

## 💡 참고

- 각 노트북은 독립적으로 학습 가능하며, 마지막에 핵심 요약 섹션이 포함되어 있습니다.
- 모든 코드는 **바로 실행 가능**하며, 복사-붙여넣기로 바로 따라할 수 있습니다.
- 추가 설명이 필요한 부분은 각 노트북 내부에 상세히 주석 처리되어 있습니다.

## 📞 추가 자료

- **EVALUATION_REPORT.md**: 보조 자료 반영도 평가
- **FINAL_SUMMARY.md**: 프로젝트 완료 요약
- **AI_curriculum.md**: 추천 학습 경로 및 도서

---

**좋은 학습 되시길 바랍니다! 🚀**
