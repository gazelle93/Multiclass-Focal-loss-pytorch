import torch
import torch.nn as nn

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, device='cpu'):
        super(FocalLoss, self).__init__(weight)
        # focusing hyper-parameter gamma
        self.gamma = gamma

        # class weights will act as the alpha parameter
        self.weight = weight
        
        # using deivce (cpu or gpu)
        self.device = device
        
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, _input, _target):
        focal_loss = 0

        for i in range(len(_input)):
            # -log(pt)
            cur_ce_loss = self.ce_loss(_input[i].view(-1, _input[i].size()[-1]), _target[i].view(-1))
            # pt
            pt = torch.exp(-cur_ce_loss)

            if self.weight is not None:
                # alpha * (1-pt)^gamma * -log(pt)
                cur_focal_loss = self.weight[_target[i]] * ((1 - pt) ** self.gamma) * cur_ce_loss
            else:
                # (1-pt)^gamma * -log(pt)
                cur_focal_loss = ((1 - pt) ** self.gamma) * cur_ce_loss
                
            focal_loss = focal_loss + cur_focal_loss

        if self.weight is not None:
            focal_loss = focal_loss / self.weight.sum()
            return focal_loss.to(self.device)
        
        focal_loss = focal_loss / torch.tensor(len(probs))    
        return focal_loss.to(self.device)
