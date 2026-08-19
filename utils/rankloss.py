import torch
import torch.nn as nn
import torch.nn.functional as F

class RankLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(RankLoss, self).__init__()
        self.alpha = alpha

    def forward(self, r_pred, r_true):
        """
        Arguments:
        r_pred -- Predicted rankings (tensor of shape [B, N])
        r_true -- Ground truth rankings (tensor of shape [B, N])

        Returns:
        total_loss -- Computed rank loss
        """

        if r_pred.shape[0] != r_true.shape[0]:
            raise ValueError(
                f"r_pred and r_true must have the same number of elements. "
                f"Got r_pred: {r_pred.shape[0]}, r_true: {r_true.shape[0]}"
            )


        r_pred = r_pred.float()
        r_true = r_true.float()

        B = r_pred.shape[0]

        # First term: MSE loss
        mse_loss = F.mse_loss(r_pred, r_true, reduction='mean')


        r_pred_expanded = r_pred.unsqueeze(0)  # shape: [1, B]
        r_true_expanded = r_true.unsqueeze(0)  # shape: [1, B]

        # Compute pairwise differences for r_pred and r_true
        r_pred_diff = r_pred_expanded.unsqueeze(2) - r_pred_expanded.unsqueeze(1)
        r_true_diff = r_true_expanded.unsqueeze(2) - r_true_expanded.unsqueeze(1)

        # Compute rank consistency loss with normalized differences
        rank_loss_matrix = torch.relu(-(r_pred_diff * r_true_diff))  # [B, N*(N-1)]

        # Combine the two terms
        rank_loss = rank_loss_matrix.mean()
        total_loss = mse_loss + self.alpha * rank_loss

        return total_loss
