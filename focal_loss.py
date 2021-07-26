import torch
import torch.nn as nn

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__(weight)
        # focusing hyper-parameter gamma
        self.gamma = gamma

        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, _input, _target):
        focal_loss = torch.tensor(0)

        for i in range(len(_input)):
            cur_ce_loss = self.ce_loss(_input[i].view(-1, _input[i].size()[-1]), _target[i].view(-1))
            pt = torch.exp(-cur_ce_loss)

            if self.weight is not None:
                cur_focal_loss = self.weight[_target[i]] * (1 - pt) ** self.gamma * cur_ce_loss
            else:
                cur_focal_loss = (1 - pt) ** self.gamma * cur_ce_loss

            focal_loss = focal_loss + cur_focal_loss

        if self.weight is not None:
            return focal_loss / self.weight.sum()

        focal_loss = focal_loss / len(_input)
        return focal_loss
