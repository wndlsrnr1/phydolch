> 딥러닝이란 함수를 층층이 쌓은 것이다

- 딥러닝 모델은 복합함수(compositie function) 이다.

복합 함수를 구성하는 하나의 함수 하나 하나가 layer(층) 이다.

즉, 각 layer는 입력 x를 받아 새로운표현으로 변환하는 하나의 함수이다. 


입력층: 원본 데이터를 첫 번째 수학 공간으로 투사
은닉층: 특징 추출 및 비선형 변환
출력층: 예측값 또는 확률로 변환

> grad가 layer 별로 저장된다의 의미

grad는 "손실 L이 각 변수(가중치, 입력)에 얼마나 영향을 받는가"를 나타내느 값이다.
즉 출력에서 입력으로 거꾸로 이동하며 각 layer를 통과할 때마다 계산된 기울기를 저장한다는 뜻이다.

1. Forward pass 

2. Backwoard pass

> 각 Layer가 손실에 얼마나 기여 했는지를 나타내느 기울기를 역전파 중에 저장한다.

```

import torch.nn as nn

# 각각의 식을 표현하는 함수가 Layer이다. 

layer1 = nn.Linear(10, 20)
layer2 = nn.ReLU()
layer3 = nn.Linear(20, 1)

```