import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.Cross_entropy_loss import cross_entropy, binary_cross_entropy, partial_cross_entropy, kpos_cross_entropy

class FocalLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 gamma=2,
                 balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.loss_weight = loss_weight

        self.gamma = gamma
        self.balance_param = balance_param

        self.cls_criterion = F.binary_cross_entropy_with_logits

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                **kwargs):

        logpt = - self.cls_criterion(cls_score, label, weight, reduction='none')
        # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        loss = self.loss_weight * balanced_focal_loss
        return loss.sum()
def create_loss():
    return FocalLoss(gamma = 2.0)