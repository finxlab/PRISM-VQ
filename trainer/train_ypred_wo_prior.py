import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import math

# Use the original MoE-based LoadingGenerator; prior factors will be ignored downstream
from module.bidirectional import LoadingGenerator

from utils import get_root_dir, calc_ic
from utils.rankloss import RankLoss
from module.quantise import VectorQuantiser
from module.layers.encoder import SpatialEncoder
from module.layers.src import RevIN
from module.layers.src import ListNetLoss


class LatentValueHead(nn.Module):
    def __init__(self, d_latent, K):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_latent, K),
            nn.LayerNorm(K),
            nn.GELU(),
            nn.Linear(K, K)
        )
    def forward(self, z_q):
        return self.head(z_q)


class GenerateReturn(pl.LightningModule):
    def __init__(self, config, T_max):
        super().__init__()
        self.config = config

        vqvae_cfg = config['vqvae']
        self.num_prior_factors = vqvae_cfg['num_prior_factors']
        self.num_embed     = vqvae_cfg['num_embed']
        self.num_features  = vqvae_cfg['num_features']
        self.hidden_size   = vqvae_cfg['hidden_size']
        self.vq_embed_dim  = vqvae_cfg['vq_embed_dim']
        self.seq_len       = vqvae_cfg['seq_len']
        self.aux_imp       = config['predictor']['aux_imp']

        # Encoder
        self.encoder = SpatialEncoder(
            input_features_C=self.num_features,
            T_window=self.seq_len,
            gru_hidden_size=self.hidden_size,
            num_transformer_heads=vqvae_cfg['encoder']['num_heads'],
            num_transformer_layers=vqvae_cfg['encoder']['num_layers'],
            final_embed_dim_d=self.vq_embed_dim,
        )

        # Quantizer
        self.quantizer = VectorQuantiser(
            num_embed=self.num_embed,
            embed_dim=self.vq_embed_dim,
            beta=vqvae_cfg['quantizer']['commit_weight'],
            distance=vqvae_cfg['quantizer']['distance'],
            anchor=vqvae_cfg['quantizer']['anchor'],
            first_batch=vqvae_cfg['quantizer']['first_batch'],
            contras_loss=vqvae_cfg['quantizer']['contras_loss'],
        )

        # RevIN
        self.revin = RevIN(self.num_features)

        # Load pretrained VQ-VAE components and freeze
        self.saved_model = config['predictor']['saved_model']
        self.load_pretrained_vqvae(os.path.join(get_root_dir(), 'checkpoints', f"{self.saved_model}"))
        self.freeze_vqvae()

        # Loading generator (MoE-based but independent of prior factor values)
        self.loadings = LoadingGenerator(config)

        # Latent factor values
        self.latent_value_head = LatentValueHead(self.vq_embed_dim, self.vq_embed_dim)

        # Return predictor without prior factors
        self.return_predictor = ReturnPredictor(num_prior=self.num_prior_factors,
                                                num_latent=self.vq_embed_dim,
                                                use_prior=False)

        self.n_features = self.num_features
        self.n_prior_factors = self.num_prior_factors
        self.T_max = T_max
        self.target_index = config['predictor']['target_day'] - 1
        self.aux_weight = config['predictor']['aux_weight']

        self.ic = []
        self.ric = []
        self.best_val_loss = float('inf')
        self.best_metrics_at_min_loss = {}
        self.rank = config['predictor']['rank']
        self.rank_loss = RankLoss(alpha=self.rank)
        self.listNet_loss = ListNetLoss(temperature=1.0)

    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'], weight_decay=1e-5)
        total_steps = self.T_max
        warmup_steps = int(0.05 * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]

    def _get_data(self, batch, batch_idx):
        batch   = batch.squeeze(0).float()
        feature = batch[:, :, 0:self.n_features]
        # prior_factor is intentionally not used in wo-prior ablation
        future_returns = batch[:, -1, self.n_features+self.n_prior_factors  : ]
        future_returns = future_returns.squeeze(-1)
        label = future_returns[:, self.target_index]
        return feature, label

    def forward(self, feature):
        # Stage 1: VQ-VAE
        feature_normalized = self.revin(feature, mode="norm")
        h_batch = self.encoder(feature_normalized)
        z_q, _, _ = self.quantizer(h_batch)
        z_q = z_q.detach()

        # Stage 2: Loading generator (produces alpha, betas)
        alpha, beta_p, beta_l, aux_loss = self.loadings(feature, z_q)

        # Prior factor values are removed (zeros) and not used by predictor
        f_prior = torch.zeros((feature.size(0), self.n_prior_factors), device=feature.device, dtype=feature.dtype)
        f_latent = self.latent_value_head(z_q)

        y_pred = self.return_predictor(
            alpha    = alpha,
            beta_p   = beta_p,
            beta_l   = beta_l,
            f_prior  = f_prior,    # ignored internally
            f_latent = f_latent,
        )
        aux_loss = torch.clamp(aux_loss, min=0, max=self.aux_imp)
        return y_pred, aux_loss

    def training_step(self, batch, batch_idx):
        feature, label = self._get_data(batch, batch_idx)
        y_pred, aux_loss = self.forward(feature)
        mse_loss = self.rank_loss(y_pred, label)
        loss = mse_loss + self.aux_weight * aux_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        feature, label = self._get_data(batch, batch_idx)
        y_pred, aux_loss = self.forward(feature)
        mse_loss = self.rank_loss(y_pred, label)
        loss = mse_loss + self.aux_weight * aux_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        daily_ic, daily_ric = calc_ic(y_pred.cpu().numpy(), label.cpu().numpy())
        self.ic.append(daily_ic)
        self.ric.append(daily_ric)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        current_ic = np.mean(self.ic)
        current_ric = np.mean(self.ric)
        current_icir = np.mean(self.ic) / np.std(self.ic) if np.std(self.ic) != 0 else 0
        current_ricir = np.mean(self.ric) / np.std(self.ric) if np.std(self.ric) != 0 else 0

        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')

        self.log('Val_RIC', current_ric, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        other_metrics = {'Val_IC': current_ic, 'Val_ICIR': current_icir, 'Val_RICIR': current_ricir}
        self.log_dict(other_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.ic = []
        self.ric = []

        if val_loss_epoch is not None and val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self.best_metrics_at_min_loss = {
                'Best_Val_Loss': float(val_loss_epoch),
                'Best_Val_IC': current_ic,
                'Best_Val_ICIR': current_icir,
                'Best_Val_RIC': current_ric,
                'Best_Val_RICIR': current_ricir,
            }
            self.log_dict(self.best_metrics_at_min_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def load_pretrained_vqvae(self, checkpoint_path=None):
        print(f"Loading pretrained VQ-VAE from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        revin_state_dict = {}
        encoder_state_dict = {}
        quantizer_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('vqvae.spatial_encoder.'):
                encoder_state_dict[k.replace('vqvae.spatial_encoder.', '')] = v
            elif k.startswith('vqvae.quantizer.'):
                quantizer_state_dict[k.replace('vqvae.quantizer.', '')] = v
            elif k.startswith('vqvae.revin.'):
                revin_state_dict[k.replace('vqvae.revin.', '')] = v

        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        self.quantizer.load_state_dict(quantizer_state_dict, strict=True)
        self.revin.load_state_dict(revin_state_dict, strict=True)
        print("VQ-VAE components loaded.")

    def freeze_vqvae(self):
        for m in [self.encoder, self.quantizer, self.revin]:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
        print("== VQ-VAE weights frozen ==")


class ReturnPredictor(nn.Module):
    def __init__(self, num_prior, num_latent, use_prior=False):
        super().__init__()
        self.num_prior = num_prior
        self.num_latent = num_latent
        self.use_prior = use_prior  # fixed False for wo-prior

    def forward(self, alpha, beta_p, beta_l, f_prior, f_latent):
        # prior term intentionally unused when use_prior=False
        latent_term = (beta_l * f_latent).sum(dim=1)
        if self.use_prior:
            prior_term = (beta_p * f_prior).sum(dim=1)
            output = alpha + prior_term + latent_term
        else:
            output = alpha + latent_term
        return output

