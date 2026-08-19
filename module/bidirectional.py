import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from einops import rearrange
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from module.layers.src import DLinear
from module.layers.temporal import TemporalTransformerEncoder
from module.layers.fusion import HyperFusion
from module.layers.src import RevIN

class LoadingGenerator(nn.Module):
    """DLinear -> Projection -> TemporalTransformer -> HyperFusion pipeline."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        vqvae_cfg = config['vqvae']
        predictor_cfg = config['predictor']
        predictor_cfg['vq_embed_dim'] = vqvae_cfg['vq_embed_dim']
        self.num_prior_factors = vqvae_cfg['num_prior_factors']
        self.vq_embed_dim = vqvae_cfg['vq_embed_dim']
        self.d_model = predictor_cfg['transformer']['d_model']
        self.num_features = vqvae_cfg['num_features']  # 158
               
        # 0. DLinear
        self.dliner = DLinear(
            seq_len = config['vqvae']['seq_len'],
            pred_len = config['predictor']['pred_len'], # what is pred_len?
            enc_in = config['vqvae']['num_features'], # 158
            kernel_size = config['predictor']['kernel_size'], # 3~7
            individual  = config['predictor']['individual'],
        )

        self.dim_projection = nn.Sequential(
            nn.Linear(self.num_features, self.d_model),
            #nn.LayerNorm(self.d_model),
            nn.GELU(),
            #nn.Dropout(self.config['predictor']['dropout']),
        )
        
        # 1. Temporal Transformer
        self.temporal_transformer = TemporalTransformerEncoder(predictor_cfg)
       
        self.fusion = HyperFusion(
            d_h=self.d_model,
            d_z=self.vq_embed_dim,
            k_prior=self.num_prior_factors,
            k_latent=self.vq_embed_dim,
            num_experts=config['predictor']['n_expert'],
            moe_k=config['predictor']['k'],
            hidden_size=config['predictor']['moe_hidden'],
            drop=config['predictor']['dropout'],
        )

    def forward(self, feature, z_q):
        """
        Args:
            feature: (B, T, 158)
            z_q: (B, vq_dim) # (B, 64)
        Returns:
            alpha: (B,) # mixing coefficient
            beta_p: (B, num_prior_factors) # prior factors
            beta_l: (B, vq_dim) # latent factors
            total_moe_loss: scalar # MoE importance loss
        """
        # ---- 0. DLinear ----
        dlinear_out = self.dliner(feature) # (B, T, 158) -> (B, pred_len, 158)
        dlinear_out = self.dim_projection(dlinear_out) # (B, pred_len, 158) -> (B, pred_len, d_model)
        # ---- 1. Temporal Transformer ----
        temp_feature = self.temporal_transformer(dlinear_out, z_q) # (B, pred_len, d_model) -> (B, pred_len, d_model)

        # ---- 2. Advanced MoE with Cross-Attention ----
        alpha, beta_p, beta_l, total_moe_loss = self.fusion(h=temp_feature, z=z_q) # (B, T, d_model), (B,64) -> outputs

        return alpha, beta_p, beta_l, total_moe_loss