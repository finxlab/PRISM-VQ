import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.autoregressive import LoadingGenerator
from utils import get_root_dir, calc_ic
from utils.rankloss import RankLoss
from module.quantise import VectorQuantiser
from module.layers.encoder import SpatialEncoder
from module.layers.decoder import ReconstructionDecoder
from module.layers.src import RevIN
from utils import corr_cluster_order
from torch.optim.lr_scheduler import LambdaLR
import math
from module.layers.src import ListNetLoss

def softcap_log1p(x, c):
    x = F.relu(x)
    return c * torch.log1p(x / c)

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

def check_vq_idx(idx, K):
    if torch.any(idx >= K) or torch.any(idx < 0):
        bad = idx[(idx >= K) | (idx < 0)]
        raise ValueError(f"[VQ-IDX] out-of-range values: {bad.tolist()} (K_latent={K})")

class GenerateReturn(pl.LightningModule):
    def __init__(self,
                 config,
                 T_max,
                 ):
        super().__init__()
        self.config = config

        vqvae_cfg = config['vqvae']

        # VQVAE parameters
        self.num_prior_factors = vqvae_cfg['num_prior_factors']
        self.num_embed = vqvae_cfg['num_embed']
        self.num_features = vqvae_cfg['num_features']
        self.hidden_size = vqvae_cfg['hidden_size']
        self.vq_embed_dim = vqvae_cfg['vq_embed_dim']
        self.seq_len = vqvae_cfg['seq_len']
        self.aux_imp = config['predictor']['aux_imp']

        # Quantizer parameters
        self.decay = vqvae_cfg['quantizer']['decay']
        self.commit_weight = vqvae_cfg['quantizer']['commit_weight']

        # Encoder parameters
        self.transformer_heads = vqvae_cfg['encoder']['num_heads']
        self.transformer_layers = vqvae_cfg['encoder']['num_layers']

        # Decoder parameters
        self.initial_T = vqvae_cfg['decoder']['initial_T']
        self.hidden_channels = vqvae_cfg['decoder']['hidden_channels']

        # 1. Encoder
        self.encoder = SpatialEncoder(
            input_features_C=self.num_features,
            T_window=self.seq_len,
            gru_hidden_size=self.hidden_size,
            num_transformer_heads=self.transformer_heads,
            num_transformer_layers=self.transformer_layers,
            final_embed_dim_d=self.vq_embed_dim
        )

        # 2. Vector Quantizer
        self.quantizer = VectorQuantiser(
            num_embed=self.num_embed,
            embed_dim=self.vq_embed_dim,
            beta=self.commit_weight,
            distance=vqvae_cfg['quantizer']['distance'],
            anchor=vqvae_cfg['quantizer']['anchor'],
            first_batch=vqvae_cfg['quantizer']['first_batch'],
            contras_loss=vqvae_cfg['quantizer']['contras_loss']
        )

        # 3. RevIN
        self.revin = RevIN(self.num_features)

        self.saved_model = config['predictor']['saved_model']
        self.load_pretrained_vqvae(checkpoint_path=os.path.join('checkpoints', f"{self.saved_model}"))
        self.freeze_vqvae()

        # 4. Factor Loading (with Autoregressive Transformer)
        self.z_prior_norm = nn.LayerNorm(self.num_prior_factors)
        self.loadings = LoadingGenerator(config)
        self.latent_value_head = LatentValueHead(
            d_latent=self.vq_embed_dim,
            K=self.vq_embed_dim
        )

        self.use_prior = config['predictor']['use_prior']
        # 5. Return Predictor
        self.return_predictor = ReturnPredictor(
            num_prior=self.num_prior_factors,
            num_latent=self.vq_embed_dim,
            use_prior=self.use_prior
        )

        self.n_features = config['vqvae']['num_features']
        self.n_prior_factors = config['vqvae']['num_prior_factors']

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'], weight_decay=1e-5)
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
        batch = batch.squeeze(0)
        batch = batch.float()
        feature = batch[:, :, 0:self.n_features]
        prior_factor = batch[:, -1, self.n_features : self.n_features+self.n_prior_factors]
        future_returns = batch[:, -1, self.n_features+self.n_prior_factors : ]
        future_returns = future_returns.squeeze(-1)

        label = future_returns[:, self.target_index]

        return feature, prior_factor, label

    def forward(self, feature, prior_factor):

        ####### STAGE1: VQVAE #######
        # 1. RevIN + Encoder
        feature_normalized = self.revin(feature, mode="norm")
        h_batch = self.encoder(feature_normalized)
        # 2. Quantization
        z_q, _, (_, min_encodings, vq_idx) = self.quantizer(h_batch)
        z_q = z_q.detach()

        ####### STAGE2: Loading Generator (Autoregressive) #######
        # 1. Loading Generator with autoregressive transformer
        alpha, beta_p, beta_l, loss_imp = self.loadings(feature, z_q)
        prior_factor_normed = self.z_prior_norm(prior_factor)

        f_latent = self.latent_value_head(z_q)

        # 2. Prediction
        y_pred = self.return_predictor(
            alpha=alpha,
            beta_p=beta_p,
            beta_l=beta_l,
            f_prior=prior_factor_normed,
            f_latent=f_latent,
        )
        loss_imp = softcap_log1p(loss_imp, self.aux_imp)
        return y_pred, beta_p, beta_l, z_q, loss_imp


    def training_step(self, batch, batch_idx):
        feature, prior_factor, label = self._get_data(batch, batch_idx)
        y_pred, beta_p, beta_l, z_q, aux_loss = self.forward(feature, prior_factor)

        mse_loss = self.rank_loss(y_pred, label)
        main_loss = mse_loss

        loss = main_loss + self.aux_weight * aux_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        feature, prior_factor, label = self._get_data(batch, batch_idx)
        y_pred, beta_p, beta_l, z_q, aux_loss = self.forward(feature, prior_factor)

        mse_loss = self.rank_loss(y_pred, label)
        main_loss = mse_loss

        loss = main_loss + self.aux_weight * aux_loss

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

        other_metrics = {
            'Val_IC': current_ic,
            'Val_ICIR': current_icir,
            'Val_RICIR': current_ricir,
        }
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


    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load pretrained model weights"""
        sd = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if "state_dict" in sd:
            sd = sd["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]

        self.load_state_dict(sd, strict=False)
        print(f"Model restored from {path}.")

    def load_pretrained_vqvae(self, checkpoint_path=None):
        """
        Load pretrained VQ-VAE components (Encoder, Quantizer, RevIN)
        """
        print(f"Loading pretrained VQ-VAE model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

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

        missing_encoder, unexpected_encoder = self.encoder.load_state_dict(encoder_state_dict, strict=True)
        missing_quantizer, unexpected_quantizer = self.quantizer.load_state_dict(quantizer_state_dict, strict=True)
        missing_revin, unexpected_revin = self.revin.load_state_dict(revin_state_dict, strict=True)

        print(f"--- Encoder loaded: missing={len(missing_encoder)}, unexpected={len(unexpected_encoder)}")
        print(f"--- Quantizer loaded: missing={len(missing_quantizer)}, unexpected={len(unexpected_quantizer)}")
        print(f"--- RevIN loaded: missing={len(missing_revin)}, unexpected={len(unexpected_revin)}")

    def freeze_vqvae(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False
        for param in self.revin.parameters():
            param.requires_grad = False
        print("== VQ-VAE weights frozen ==")

        self.encoder.eval()
        self.quantizer.eval()
        self.revin.eval()

class ReturnPredictor(nn.Module):
    def __init__(self, num_prior, num_latent, use_prior=True):
        super().__init__()
        self.num_prior = num_prior
        self.num_latent = num_latent
        self.use_prior = use_prior

    def forward(self, alpha, beta_p, beta_l, f_prior, f_latent):
        prior_term = (beta_p * f_prior).sum(dim=1)
        latent_term = (beta_l * f_latent).sum(dim=1)

        combined = torch.cat([prior_term.unsqueeze(1), latent_term.unsqueeze(1)], dim=1)

        if self.use_prior:
            output = alpha + prior_term + latent_term
        else:
            output = alpha + latent_term

        return output
