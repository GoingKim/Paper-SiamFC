##
import torch
import torch.nn as nn
import torch.nn.functional as F


##
class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')

'''
논문과 구현 방식이 조금 다르다.

논문에서는 binary cross entropy를 사용하면서, 
logit function의 x 입력에 모델의 출력값(=예측값, response, score map)과
객체의 존재 유무에 따른 1과 -1의 ground truth 값을 곱하여 loss함수를 구현하였다.

그리고 예측값들이 여러개가 나오기 때문에 이 값들을 통해 얻은 loss들의 평균을 최종 score map loss로 정의했다.

논문에서 D는 예측값들이 있는 공간인듯
ground truth(1, -1)은 train.py의 _create_labels라는 함수에서 만들어내는듯
결국 모델이 학습하는 건 그 자리에 객체가 있냐 없냐고 박스의 실 좌표는 displacement를 별도로 계산하는
(그냥 모델이 한번에 학습 하는게 낫지않나...?)듯

여기 코드에서는 논문처럼 하지 않고 grount truth를 0,1 negative/positve 마스크를 만든 후
그것을 바탕으로 다시 weighted mask만들어서 객체가 없는 자리는 낮은 가중치로 있는 자리는 높은 가중치로 학습시키는듯
(아래 참고하면 weight는 repeated 될때 input과 계산되는 값인듯)
(sum은 다양한 예측값들을 계산 후 합치기 위한 것으로 보인다.)


BCEWithLogitsLoss는 BCELoss에 Sigmoid Layer를 더한 식이다.

torch.nn.functional.binary_cross_entropy_with_logits
(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

weight (Tensor, optional) 
– a manual rescaling weight if provided it’s repeated to match input tensor shape

reduction (string, optional) 
sum': the output will be summed. 
'''
