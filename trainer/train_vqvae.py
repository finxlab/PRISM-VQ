import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from module.autoencoder import VQVAE
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class FactorVQVAE(pl.LightningModule):
    def __init__(self,
                 config,
                 T_max,
                 ):
        super().__init__()
        self.config = config
        self.vqvae = VQVAE(config)
        self._alpha_n = config['vqvae'].get('num_features', 158)
        self.n_prior_factors = config['vqvae'].get('num_prior_factors', 13)
        self.epoch_vq_codes = []
        self.T_max = T_max
        self.inference = False

    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'])
        scheduler  = CosineAnnealingLR(optimizer, T_max=self.T_max)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]
    
    def forward(self, feature, prior_factor, future_returns):

        recon_loss, vq_loss, pred_loss, total_loss, z_q,\
              (perplexity, min_encodings, encoding_indices) = self.vqvae(feature, prior_factor, future_returns)
        
        return recon_loss, vq_loss, pred_loss, total_loss, z_q, (perplexity, min_encodings, encoding_indices)
    
    def _get_data(self, batch, batch_idx):
        batch   = batch.squeeze(0)
        batch   = batch.float()
        feature = batch[:, :, 0:self._alpha_n] # (300, 20, 158)
        prior_factor = batch[:, -1, self._alpha_n : self._alpha_n+self.n_prior_factors] # (300, 13)
        future_returns = batch[:, -1, self._alpha_n+self.n_prior_factors  : ] # (300, 1, 10)
        future_returns = future_returns.squeeze(-1) # (300, 10)
        
        return feature, prior_factor, future_returns # (B, T, C), (B, P), (B, 10)
    
    def training_step(self, batch, batch_idx):
        feature, prior_factor, future_returns = self._get_data(batch, batch_idx)

        recon_loss, vq_loss, pred_loss, total_loss, z_q, (perplexity, min_encodings, encoding_indices)= self.forward(feature, prior_factor, future_returns)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_vq_loss', vq_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_pred_loss', pred_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # Collect vq_dict['vq_dict_i'] values
        self.epoch_vq_codes.append(encoding_indices.detach().cpu().numpy())

        return {"loss": total_loss, "recon_loss": recon_loss, "vq_loss": vq_loss, "pred_loss": pred_loss}
    
    def validation_step(self, batch, batch_idx):
        feature, prior_factor, future_returns = self._get_data(batch, batch_idx)
        recon_loss, vq_loss, pred_loss, total_loss, z_q, (perplexity, min_encodings, encoding_indices)= self.forward(feature, prior_factor, future_returns)

        self.log('val_loss', total_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_vq_loss', vq_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_pred_loss', pred_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return {"loss": total_loss, "recon_loss": recon_loss, "vq_loss": vq_loss, "pred_loss": pred_loss}
    
    def on_train_epoch_end(self):

        train_loss_epoch = self.trainer.callback_metrics.get('train_loss')
        if train_loss_epoch is not None:
            self.log('train_loss_epoch', train_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        if self.epoch_vq_codes:
            # Concatenate all collected vq_codes for the epoch
            epoch_vq_codes_np = np.concatenate(self.epoch_vq_codes, axis=0)
            
            # Calculate unique value counts
            unique, counts = np.unique(epoch_vq_codes_np, return_counts=True)
            unique_counts = {int(k): int(v) for k, v in zip(unique, counts)}
            
            # Plot bar graph
            plt.figure(figsize=(25, 5))
            plt.bar(unique_counts.keys(), unique_counts.values())
            plt.xlabel('Codebook Index')
            plt.ylabel('Frequency')
            plt.title(f'Codebook Utilization at Epoch {self.current_epoch}')
            plt.xticks(list(unique_counts.keys()))
            plt.grid(True)
            
            # Log bar graph using wandb
            wandb.log({"Codebook Utilization": wandb.Image(plt)})
            
            #* Clear the list for the next epoch
            self.epoch_vq_codes = []
            plt.close()

    def on_validation_epoch_end(self):
        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        if val_loss_epoch is not None:  
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        if self.epoch_vq_codes:
            # Concatenate all collected vq_codes for the epoch
            epoch_vq_codes_np = np.concatenate(self.epoch_vq_codes, axis=0)
            
            # Calculate unique value counts
            unique, counts = np.unique(epoch_vq_codes_np, return_counts=True)
            unique_counts = {int(k): int(v) for k, v in zip(unique, counts)}
            
            # Plot bar graph
            plt.figure(figsize=(25, 5))
            plt.bar(unique_counts.keys(), unique_counts.values())
            plt.xlabel('Codebook Index')
            plt.ylabel('Frequency')
            plt.title(f'Codebook Utilization at Epoch {self.current_epoch}')
            plt.xticks(list(unique_counts.keys()))
            plt.grid(True)
            plt.close()


    # def preprocessing(self, feature, label):
    #     # drop duplicate and drop label nan
    #     if feature.shape[0] > 0:
    #         dup_mask = drop_duplicates(feature)
    #         feature = feature[dup_mask, :, :]
    #         label = label[dup_mask]

    #     mask, label = drop_na(label)
    #     feature = feature[mask.flatten(),:, :]

    #     return feature, label.unsqueeze(-1)