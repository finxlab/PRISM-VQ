import torch
import torch.nn.functional as F
import torch.nn as nn

class ListNetLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred, target):
        """
        pred: (batch_size,) predicted returns
        target: (batch_size,) realized returns
        """
        pred_prob = F.softmax(pred / self.temperature, dim=0)
        target_prob = F.softmax(target / self.temperature, dim=0)

        # Simplified KL divergence form.
        loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-10))

        return loss