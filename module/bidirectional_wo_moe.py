import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from einops import rearrange

# Ensure top-level imports work when called from project root
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.layers.src import DLinear
from module.layers.temporal import TemporalTransformerEncoder


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation (lightweight copy for no‑MoE path)
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class ResBlock(nn.Module):
    """
    Residual block: Pre-Norm → GEGLU → Linear → Dropout → Residual
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


class HyperFusionNoMoE(nn.Module):
    """
    MoE-free fusion of temporal features h and latent z.
    Produces alpha, beta_prior, beta_latent without Mixture-of-Experts.
    Uses FiLM-like modulation from fused features instead of MoE.
    """
    def __init__(self,
                 d_h: int,
                 d_z: int,
                 k_prior: int,
                 k_latent: int,
                 drop: float = 0.2):
        super().__init__()

        input_dim = d_h + d_z
        self.norm_h = nn.LayerNorm(d_h)
        self.norm_z = nn.LayerNorm(d_z)

        # Lightweight MLP to fuse (h, z)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_h),
            ResBlock(dim=d_h, hidden=d_h * 2, drop=drop),
            nn.Linear(d_h, d_h),
            nn.GELU(),
        )

        # Base beta heads from h
        self.base_beta_prior_head = nn.Linear(d_h, k_prior)
        self.base_beta_latent_head = nn.Linear(d_h, k_latent)

        # FiLM parameters from fused representation
        self.film_prior_head = nn.Linear(d_h, k_prior * 2)
        self.film_latent_head = nn.Linear(d_h, k_latent * 2)

        # Alpha head
        self.alpha_head = nn.Linear(d_h, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor):
        # Normalize and fuse
        h_norm = self.norm_h(h)
        z_norm = self.norm_z(z)
        x_fused = torch.cat([h_norm, z_norm], dim=-1)
        fused = self.input_proj(x_fused)

        # Base betas from h (captures temporal trend)
        base_beta_p = self.base_beta_prior_head(h)
        base_beta_l = self.base_beta_latent_head(h)

        # FiLM parameters from fused features
        gamma_p, delta_p = self.film_prior_head(fused).chunk(2, dim=-1)
        gamma_l, delta_l = self.film_latent_head(fused).chunk(2, dim=-1)

        # Apply FiLM: y = gamma * x + delta
        beta_p = (gamma_p * base_beta_p) + delta_p
        beta_l = (gamma_l * base_beta_l) + delta_l

        # Alpha from fused features
        alpha = self.alpha_head(fused).squeeze(-1)

        # Regularization term (acts as aux loss in place of MoE importance)
        beta_reg_loss = (torch.norm(beta_p, p=2) + torch.norm(beta_l, p=2))

        return alpha, beta_p, beta_l, beta_reg_loss


class LoadingGeneratorWOMoE(nn.Module):
    """
    DLinear → Projection → TemporalTransformer → HyperFusionNoMoE pipeline.
    Interface matches the original LoadingGenerator.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        vqvae_cfg = config['vqvae']
        predictor_cfg = config['predictor']
        predictor_cfg['vq_embed_dim'] = vqvae_cfg['vq_embed_dim']
        self.num_prior_factors = vqvae_cfg['num_prior_factors']
        self.vq_embed_dim = vqvae_cfg['vq_embed_dim']
        self.d_model = predictor_cfg['transformer']['d_model']
        self.num_features = vqvae_cfg['num_features']

        # 0. DLinear
        self.dlinear = DLinear(
            seq_len=vqvae_cfg['seq_len'],
            pred_len=predictor_cfg['pred_len'],
            enc_in=vqvae_cfg['num_features'],
            kernel_size=predictor_cfg['kernel_size'],
            individual=predictor_cfg['individual'],
        )

        # Project per-time-step features to d_model (kept minimal)
        self.dim_projection = nn.Sequential(
            nn.Linear(self.num_features, self.d_model),
            nn.GELU(),
        )

        # 1. Temporal Transformer
        self.temporal_transformer = TemporalTransformerEncoder(predictor_cfg)

        # 2. MoE-free fusion
        self.fusion = HyperFusionNoMoE(
            d_h=self.d_model,
            d_z=self.vq_embed_dim,
            k_prior=self.num_prior_factors,
            k_latent=self.vq_embed_dim,
            drop=predictor_cfg['dropout'],
        )

    def forward(self, feature: torch.Tensor, z_q: torch.Tensor):
        """
        Args:
            feature: (B, T, C)
            z_q: (B, vq_dim)
        Returns:
            alpha: (B,)
            beta_p: (B, num_prior_factors)
            beta_l: (B, vq_dim)
            aux_loss: scalar (beta regularization)
        """
        # 0. DLinear + projection
        dlinear_out = self.dlinear(feature)  # (B, pred_len, C)
        dlinear_out = self.dim_projection(dlinear_out)  # (B, pred_len, d_model)

        # 1. Temporal summary (CLS)
        temp_feature = self.temporal_transformer(dlinear_out, z_q)  # (B, d_model)

        # 2. Fusion without MoE
        alpha, beta_p, beta_l, aux_loss = self.fusion(h=temp_feature, z=z_q)
        return alpha, beta_p, beta_l, aux_loss

