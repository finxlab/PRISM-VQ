import torch
import torch.nn as nn


class router(nn.Module):
    def __init__(self, seq_len, num_experts, router_size):
        super(router, self).__init__()
        self.input_size = seq_len
        self.num_experts = num_experts
        self.router_size = router_size

        self.distribution_fit = nn.Sequential(nn.Linear(self.input_size, self.router_size, bias=False), nn.ReLU(),
                                              nn.Linear(self.router_size, self.num_experts, bias=False))

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(mean)
        return out