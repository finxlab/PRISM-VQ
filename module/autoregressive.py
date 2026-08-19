import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from einops import rearrange
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from module.layers.src import DLinear
from module.layers.fusion import HyperFusion
from module.layers.src import RevIN
from module.layers.src.encoding import RotaryPE
import copy


def _get_activation_fn(activation):
    if callable(activation):
        return activation
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {activation}")


class RotaryTransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer with RoPE-aware causal self-attention (GPT-style)."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        activation="relu",
        rotary_pe=None,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for RoPE layer")

        self.d_model = d_model
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.rotary_pe = rotary_pe

        # Self-attention (causal)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Autoregressive forward pass with causal masking.

        Args:
            tgt: Target sequence (B, T, d_model)
            tgt_mask: Attention mask (not typically used with is_causal=True)
            tgt_key_padding_mask: Padding mask (B, T)
        """
        if not self.batch_first:
            tgt = tgt.transpose(0, 1)

        if self.norm_first:
            tgt = tgt + self._self_attn_block(
                self.norm1(tgt), tgt_mask, tgt_key_padding_mask, is_causal=True
            )
            tgt = tgt + self._ff_block(self.norm2(tgt))
        else:
            tgt = self.norm1(
                tgt + self._self_attn_block(tgt, tgt_mask, tgt_key_padding_mask, is_causal=True)
            )
            tgt = self.norm2(tgt + self._ff_block(tgt))

        if not self.batch_first:
            tgt = tgt.transpose(0, 1)
        return tgt

    def _self_attn_block(self, x, attn_mask, key_padding_mask, is_causal):
        attn_output = self._scaled_dot_product_attention(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal
        )
        return self.dropout1(attn_output)

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_ff(x)
        x = self.linear2(x)
        return self.dropout2(x)

    def _scaled_dot_product_attention(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
    ):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_pe is not None:
            q = self._apply_rope(q, seq_len)
            k = self._apply_rope(k, seq_len)

        attn_scores = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attn_mask is not None:
            attn_scores = attn_scores + self._expand_attn_mask(attn_mask, attn_scores)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(
                dtype=torch.bool, device=attn_scores.device
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(attn_output)

    def _apply_rope(self, tensor, seq_len):
        batch_size, num_heads, seq_len_tensor, head_dim = tensor.shape
        if seq_len_tensor != seq_len:
            raise ValueError("Sequence length mismatch when applying RoPE")

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        tensor = self.rotary_pe.apply_rotary(tensor, seq_len)
        tensor = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return tensor

    def _expand_attn_mask(self, attn_mask, attn_scores):
        if attn_mask.dim() == 2:
            expanded = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            expanded = attn_mask.unsqueeze(1)
        else:
            raise ValueError("Unsupported attn_mask dimensions for RoPE layer")

        expanded = expanded.to(attn_scores.device)
        if expanded.dtype == torch.bool:
            expanded = expanded.masked_fill(expanded, float("-inf"))
        return expanded


class RotaryTransformerDecoder(nn.Module):
    """Autoregressive decoder stack with causal masking (GPT-style)."""

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TemporalTransformerDecoder(nn.Module):
    """
    Autoregressive transformer decoder with RoPE positional encoding (GPT-style).
    Uses causal masking to prevent attending to future tokens.
    """
    def __init__(self, config_predictor):
        super().__init__()
        self.max_len = config_predictor['pred_len'] + 1
        self.d_model = config_predictor['transformer']['d_model']
        self.num_heads = config_predictor['transformer']['num_heads']
        self.dim_feedforward = config_predictor['transformer']['dim_feedforward']
        self.dropout = config_predictor['transformer']['dropout']
        self.batch_first = config_predictor['transformer']['batch_first']
        self.num_layers = config_predictor['transformer']['num_layers']
        self.z_q_dim = config_predictor['vq_embed_dim']
        self.num_features = config_predictor['num_features']

        self.rotary_pe = RotaryPE(self.d_model, self.max_len)

        layer = RotaryTransformerDecoderLayer(
            self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
            rotary_pe=self.rotary_pe,
        )

        self.z_q_proj = nn.Linear(self.z_q_dim, self.d_model)
        self.normalization = nn.LayerNorm(self.d_model)
        self.transformer = RotaryTransformerDecoder(layer, num_layers=self.num_layers)

        # Input projection layer
        self.input_proj = nn.Linear(self.num_features, self.d_model)

    def forward(self, x, z_q):
        """
        Autoregressive forward pass.

        Args:
            x: Input sequence (B, T, D)
            z_q: Quantized latent (B, z_q_dim)

        Returns:
            CLS token representation (B, d_model)
        """
        B, T, D = x.shape

        # Project input to d_model if needed
        if D != self.d_model:
            x = self.input_proj(x)
            D = self.d_model

        # Project z_q and prepend as CLS token
        if z_q.shape[1] != D:
            z_q = self.z_q_proj(z_q)

        # Prepend z_q as the first token (CLS token)
        x = torch.cat((z_q.unsqueeze(1), x), dim=1)  # (B, 1+T, D)

        # Apply autoregressive transformer with causal masking
        out = self.transformer(x)  # (B, T+1, D)

        # Return the last token (which has seen all previous tokens in causal manner)
        return out[:, -1]  # (B, D)


class LoadingGenerator(nn.Module):
    """
    DLinear → Projection → Autoregressive Transformer → HyperFusion pipeline
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
        self.dliner = DLinear(
            seq_len=config['vqvae']['seq_len'],
            pred_len=config['predictor']['pred_len'],
            enc_in=config['vqvae']['num_features'],
            kernel_size=config['predictor']['kernel_size'],
            individual=config['predictor']['individual'],
        )

        self.dim_projection = nn.Sequential(
            nn.Linear(self.num_features, self.d_model),
            nn.GELU(),
        )

        # 1. Temporal Transformer (Autoregressive)
        self.temporal_transformer = TemporalTransformerDecoder(predictor_cfg)

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
            z_q: (B, vq_dim)
        Returns:
            alpha: (B,)
            beta_p: (B, num_prior_factors)
            beta_l: (B, vq_dim)
            total_moe_loss: scalar
        """
        # ---- 0. DLinear ----
        dlinear_out = self.dliner(feature)  # (B, T, 158) -> (B, pred_len, 158)
        dlinear_out = self.dim_projection(dlinear_out)  # (B, pred_len, 158) -> (B, pred_len, d_model)

        # ---- 1. Autoregressive Temporal Transformer ----
        temp_feature = self.temporal_transformer(dlinear_out, z_q)  # (B, d_model)

        # ---- 2. Advanced MoE with Cross-Attention ----
        alpha, beta_p, beta_l, total_moe_loss = self.fusion(h=temp_feature, z=z_q)

        return alpha, beta_p, beta_l, total_moe_loss
