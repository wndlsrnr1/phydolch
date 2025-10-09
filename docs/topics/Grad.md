> 기록을 추적하는 것을 방지하기 위해 코드 블럭을 with torch.no_grad() 로 감싸면 기울기 계산은 필요없지만, requires_grad=True로 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용

> "requires_grad=True로 설정되어" "학습 가능한 매개변수를 갖는 모델"을 "평가" 할때 유용하다의 의미?

- PyTorch에서 "이 변수는 학습 대상이다"라고 명시한 상태를 의미한다.

1. learnable parameters (학습 가능한 매개변수)

이건 신경망 내부에서 실제로 손실을 줄이기 위해 조정되는 값들,  가중치(weight)와 편향(bias)이다.

> x, y 즉 데이터 그 자체가 아니라 가중치나 편향을 의미

이 매개변수들은 requires_grad=True로 설정되어 있어야 loss.backword()를 실행할때 param.grad()에 미분값이 기록되고 그걸 기반으로 optimizer가 값을 갱신한다. 

한줄 정리: 
학습 가능한 매개변수 = gradient가 계산되고 업데이트 되는 대상.

Evaludate 평가:

"평가"란 이미 학습된 모델의 성능을 확인하는 단계.
지금 상태로 데이터를 얼마나 잘 처리하는가를 확인

