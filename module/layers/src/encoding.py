import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Optional

# ----------------------------------
# 1. Positional Encoding Modules
# ----------------------------------
class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class LearnableAbsPE(nn.Module):
    """Learnable absolute positional encoding."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class RelativePE(nn.Module):
    """Shaw et al. 2018 relative positional encoding (T^2 memory)."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # This is a placeholder - actual implementation would be in attention
        
    def forward(self, x):
        # RelativePE is applied within attention mechanism
        return x

class RotaryPE(nn.Module):
    """RoPE: Rotary Position Embedding"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # Create rotation frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('freqs', freqs)
        
    def apply_rotary(self, x, seq_len):
        device = x.device
        freqs = self.freqs.to(device)
        
        # Create position indices
        pos = torch.arange(seq_len, device=device).float()
        
        # Compute rotation angles
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, d_model/2)
        
        # Split x into real and imaginary parts
        x_real = x[..., 0::2]  # (B, seq_len, d_model/2)
        x_imag = x[..., 1::2]  # (B, seq_len, d_model/2)
        
        # Apply rotation
        cos_angles = torch.cos(angles).unsqueeze(0)  # (1, seq_len, d_model/2)
        sin_angles = torch.sin(angles).unsqueeze(0)  # (1, seq_len, d_model/2)
        
        rotated_real = x_real * cos_angles - x_imag * sin_angles
        rotated_imag = x_real * sin_angles + x_imag * cos_angles
        
        # Recombine
        rotated = torch.zeros_like(x)
        rotated[..., 0::2] = rotated_real
        rotated[..., 1::2] = rotated_imag
        
        return rotated
        
    def forward(self, x):
        return x  # RoPE is applied in attention, not here


class Time2VecPE(nn.Module):
    """Time2Vec (Kazemi 2019)"""
    def __init__(self, d_model):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(d_model-1))
        self.B = nn.Parameter(torch.randn(d_model-1))

    def forward(self, x):
        t = torch.arange(x.size(1), dtype=x.dtype, device=x.device)
        v0 = (self.w0 * t + self.b0).unsqueeze(1)          # (T,1)
        v1 = torch.sin(torch.outer(t, self.W) + self.B)    # (T,d-1)
        return x + torch.cat([v0, v1], dim=1)
    

class Time2VecPE2(nn.Module):
    """Time2Vec: Learning a Vector Representation of Time"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Linear transformation for the first dimension
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        
        # Periodic transformations for remaining dimensions
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.randn(d_model - 1))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create time indices
        time_idx = torch.arange(seq_len, device=x.device).float().unsqueeze(0).unsqueeze(-1)
        time_idx = time_idx.expand(batch_size, -1, -1)  # (B, T, 1)
        
        # Linear component
        linear_part = self.w0 * time_idx + self.b0  # (B, T, 1)
        
        # Periodic components
        periodic_parts = []
        for i in range(self.d_model - 1):
            periodic = torch.sin(self.w[i] * time_idx.squeeze(-1) + self.b[i])
            periodic_parts.append(periodic.unsqueeze(-1))
        
        periodic_part = torch.cat(periodic_parts, dim=-1)  # (B, T, d_model-1)
        
        # Combine linear and periodic parts
        time_encoding = torch.cat([linear_part, periodic_part], dim=-1)  # (B, T, d_model)
        
        return x + time_encoding

class tAPE(nn.Module):
    """time Absolute Position Encoding - Adaptive sinusoidal encoding"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(1).float()
        
        # Modified frequency term considering sequence length
        for k in range(d_model // 2):
            omega_k_new = k * d_model / seq_len  # Key modification
            
            pe[:, 2*k] = torch.sin(position.squeeze(1) * omega_k_new)
            if 2*k + 1 < d_model:
                pe[:, 2*k + 1] = torch.cos(position.squeeze(1) * omega_k_new)
        
        return x + pe.unsqueeze(0)

class TUPE(nn.Module):
    """Transformer with Untied Positional Encoding"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # Separate position embeddings for queries and keys
        self.pos_q = nn.Parameter(torch.randn(max_len, d_model))
        self.pos_k = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        # TUPE is applied in attention mechanism
        return x
    
    def get_position_bias(self, seq_len):
        """Get position bias for attention computation"""
        pos_q = self.pos_q[:seq_len]  # (seq_len, d_model)
        pos_k = self.pos_k[:seq_len]  # (seq_len, d_model)
        
        # Compute position-to-position interactions
        pos_bias = torch.matmul(pos_q, pos_k.transpose(-2, -1))  # (seq_len, seq_len)
        pos_bias = pos_bias / math.sqrt(self.d_model)
        
        return pos_bias

class ConvSPE(nn.Module):
    """Convolutional Sinusoidal Positional Encoding"""
    def __init__(self, d_model, kernel_size=3, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Convolutional layers for position encoding
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        
        # Random matrix for stochastic encoding
        self.register_buffer('Z', torch.randn(max_len, d_model))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Get random features
        z = self.Z[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, d_model)
        
        # Apply convolutions
        z_t = z.transpose(1, 2)  # (B, d_model, T)
        q_encoding = self.conv_q(z_t).transpose(1, 2)  # (B, T, d_model)
        k_encoding = self.conv_k(z_t).transpose(1, 2)  # (B, T, d_model)
        
        # For simplicity, just add query encoding to input
        return x + q_encoding

class TemporalPE(nn.Module):
    """Temporal Positional Encoding (T-PE) - Combines geometric and semantic components"""
    def __init__(self, d_model, max_len=512, sigma=1.0):
        super().__init__()
        self.d_model = d_model
        self.sigma = sigma
        
        # Geometric component (enhanced sinusoidal)
        self.geometric_pe = SinusoidalPE(d_model, max_len)
        
        # Semantic component parameters
        self.semantic_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Apply geometric PE
        x_geo = self.geometric_pe(x)
        
        # Compute semantic similarity
        batch_size, seq_len, d_model = x.shape
        semantic_sim = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Compute token similarity
                diff = x[:, i] - x[:, j]  # (B, d_model)
                dist_sq = torch.sum(diff ** 2, dim=-1)  # (B,)
                sim = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # (B,)
                semantic_sim[:, i, j] = sim
        
        # Apply semantic weighting (simplified)
        semantic_encoding = torch.mean(semantic_sim, dim=-1, keepdim=True)  # (B, T, 1)
        semantic_encoding = semantic_encoding.expand(-1, -1, d_model)  # (B, T, d_model)
        
        return x_geo + self.semantic_weight * semantic_encoding

class AdaptiveTemporalPE(nn.Module):
    """Adaptive temporal positional encoding."""
    def __init__(self, d_model, max_len=512, adaptation_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.adaptation_rate = adaptation_rate
        
        # Multi-scale time embeddings
        self.short_term_pe = Time2VecPE(d_model // 2)
        self.long_term_pe = SinusoidalPE(d_model // 2, max_len)
        
        # Adaptive weighting network
        self.adaptation_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),  # weights for short/long term
            nn.Softmax(dim=-1)
        )
        
        # Learnable scale factors
        self.scale_short = nn.Parameter(torch.ones(1))
        self.scale_long = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Split input for multi-scale processing
        x_short = x[..., :d_model//2]
        x_long = x[..., d_model//2:]
        
        # Apply different PE to different scales
        short_encoded = self.short_term_pe(x_short) * self.scale_short
        long_encoded = self.long_term_pe(x_long) * self.scale_long
        
        # Compute adaptive weights based on input content
        weights = self.adaptation_net(x)  # (B, T, 2)
        w_short = weights[..., 0:1]  # (B, T, 1)
        w_long = weights[..., 1:2]   # (B, T, 1)
        
        # Apply adaptive weighting
        short_weighted = short_encoded * w_short.expand(-1, -1, d_model//2)
        long_weighted = long_encoded * w_long.expand(-1, -1, d_model//2)
        
        # Combine
        result = torch.cat([short_weighted, long_weighted], dim=-1)
        
        return result

class FourierPE(nn.Module):
    """Fourier-based Positional Encoding for capturing periodicities"""
    def __init__(self, d_model, max_len=512, n_frequencies=None):
        super().__init__()
        self.d_model = d_model
        if n_frequencies is None:
            n_frequencies = d_model // 4
        
        # Learnable frequency components
        self.frequencies = nn.Parameter(torch.randn(n_frequencies))
        self.phases = nn.Parameter(torch.randn(n_frequencies))
        self.amplitudes = nn.Parameter(torch.ones(n_frequencies))
        
        # Linear projection to full dimension
        self.proj = nn.Linear(n_frequencies * 2, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create time indices
        t = torch.arange(seq_len, device=x.device).float().unsqueeze(0).unsqueeze(-1)
        t = t.expand(batch_size, -1, len(self.frequencies))
        
        # Compute Fourier features
        angles = 2 * math.pi * self.frequencies.unsqueeze(0).unsqueeze(0) * t + self.phases.unsqueeze(0).unsqueeze(0)
        cos_features = self.amplitudes.unsqueeze(0).unsqueeze(0) * torch.cos(angles)
        sin_features = self.amplitudes.unsqueeze(0).unsqueeze(0) * torch.sin(angles)
        
        # Combine and project
        fourier_features = torch.cat([cos_features, sin_features], dim=-1)  # (B, T, n_freq*2)
        pe = self.proj(fourier_features)  # (B, T, d_model)
        
        return x + pe