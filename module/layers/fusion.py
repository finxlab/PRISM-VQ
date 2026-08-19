import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .moe import FactorGatedMoE
from typing import Tuple, Optional


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension  
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)
    
class ResBlock(nn.Module):
    """
    Residual Block with Pre-Norm → GEGLU → Linear → Dropout → Residual add
    
    Args:
        dim: Input/output dimension
        hidden: Hidden dimension (should be > dim for expansion)
        drop: Dropout probability
    """
    def __init__(self, dim: int, hidden: int, drop: float = 0.2):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.ff    = GEGLU(dim, hidden)
        self.proj  = nn.Linear(hidden, dim)
        self.drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(self.ff(self.norm(x)))
        return x + self.drop(y)
    


class HyperFusion(nn.Module):
    """
    (h, z) -> alpha, beta_prior, beta_latent.
    - Generates a dynamic base beta from h.
    - Modulates that base via FiLM (gamma, delta) using the z-conditioned MoE output.
    - Predicts alpha from the MoE output.
    """
    def __init__(self,
                 d_h: int,
                 d_z: int,
                 k_prior: int,
                 k_latent: int,
                 drop: float = 0.2,
                 num_experts: int = 4,
                 moe_k: int = 1,
                 hidden_size: int = 64):
        super().__init__()

        # 1. Input projection
        input_dim = d_h + d_z
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_h),
            ResBlock(dim=d_h, hidden=d_h * 2, drop=drop),
            nn.Linear(d_h, d_h)
        )

        self.norm_h = nn.LayerNorm(d_h)
        self.norm_z = nn.LayerNorm(d_z)

        # 2. MoE
        self.moe = FactorGatedMoE(
            gate_input_size=d_z,
            expert_input_size=d_h,
            hidden_size=hidden_size,
            num_experts=num_experts,
            k=moe_k
        )

        # 3. Base beta heads (from h)
        self.base_beta_prior_head = nn.Linear(d_h, k_prior)
        self.base_beta_latent_head = nn.Linear(d_h, k_latent)

        # 4. FiLM (gamma, delta) heads (from MoE output)
        self.film_prior_head = nn.Linear(hidden_size, k_prior * 2)
        self.film_latent_head = nn.Linear(hidden_size, k_latent * 2)

        # 5. Alpha head
        self.alpha_head = nn.Linear(hidden_size, 1)

    def forward(self, h, z):
        h_norm = self.norm_h(h)
        z_norm = self.norm_z(z)

        x_fused = torch.cat([h_norm, z_norm], dim=-1)
        x_proj = self.input_proj(x_fused)
        moe_out, moe_loss = self.moe(x=x_proj, z=z_norm)

        base_beta_p = self.base_beta_prior_head(h)
        base_beta_l = self.base_beta_latent_head(h)

        gamma_p, delta_p = self.film_prior_head(moe_out).chunk(2, dim=-1)
        gamma_l, delta_l = self.film_latent_head(moe_out).chunk(2, dim=-1)

        # FiLM: y = gamma * x + delta
        beta_p = (gamma_p * base_beta_p) + delta_p
        beta_l = (gamma_l * base_beta_l) + delta_l

        alpha = self.alpha_head(moe_out).squeeze(-1)

        beta_reg_loss = (torch.norm(beta_p, p=2) + torch.norm(beta_l, p=2))

        return alpha, beta_p, beta_l, moe_loss + beta_reg_loss