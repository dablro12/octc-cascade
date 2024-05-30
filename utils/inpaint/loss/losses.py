import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2.0, reduction = None):
        """
        Focal Loss Initialize
        Args:
            alpha (float, optional): Hyper Parameters. Defaults to 0.25.
            gamma (float, optional): Focusing Parameter. Defaults to 2.0.
            reduction (str, optional): How to calculate costed model Type. Defaults to 'None'.
        """
        
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction 
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets):
        """
        Forward Setting
        Args:
            outputs (tensor, optional): model output
            targets (tensor, optional): labels 
        return : Focal Loss 
        """
        outputs, targets = outputs.to(targets.device), targets
        bce_loss = self.bce_loss(outputs, targets)
        pt = torch.exp(-bce_loss)
        
        focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
    
        
        
        