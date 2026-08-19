import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. 1-D PixelShuffle (channels -> length, factor r)
# ---------------------------------------------------------------------------
class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        if not (isinstance(upscale_factor, int) and upscale_factor >= 2):
            raise ValueError("upscale_factor must be integer ≥ 2")
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_r, T = x.shape
        if C_r % self.r != 0:
            raise RuntimeError(f"Input-channels {C_r} not divisible by r={self.r}")
        C = C_r // self.r
        # (N, C·r, T) → (N, C, T·r)
        return (
            x.view(N, C, self.r, T)
             .permute(0, 1, 3, 2)           # (N, C, T, r)
             .reshape(N, C, T * self.r)
        )

# ---------------------------------------------------------------------------
# 1. Orthogonal projector (z_q ⟂ z_prior)
# ---------------------------------------------------------------------------
class OrthogonalProjector(nn.Module):
    def __init__(self, d_q: int, d_prior: int):
        super().__init__()
        self.Wp = nn.Linear(d_prior, d_q, bias=False)

    def forward(self, z_q: torch.Tensor, z_p: torch.Tensor):
        """
        z_q : (N, d_q)
        z_p : (N, d_prior)
        returns hat_z_q, hat_z_p  (shape : (N, d_q))
        """
        z_p_proj = self.Wp(z_p)                              # (N, d_q)
        proj_coef = (z_q * z_p_proj).sum(-1, keepdim=True) / (
            z_p_proj.pow(2).sum(-1, keepdim=True).clamp(min=1e-8)
        )
        z_q_orth = z_q - proj_coef * z_p_proj                # orthogonal component
        hat_z_q = F.layer_norm(z_q_orth, z_q_orth.shape[-1:])
        hat_z_p = F.layer_norm(z_p_proj,  z_p_proj.shape[-1:])
        return hat_z_q, hat_z_p

# ---------------------------------------------------------------------------
# 2. FiLM gating (condition -> gamma, beta)
# ---------------------------------------------------------------------------
class FiLM1D(nn.Module):
    def __init__(self, cond_dim: int, n_channels: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(32, cond_dim * 2)
        self.to_gb = nn.Sequential(
            nn.Linear(cond_dim, h), nn.GELU(),
            nn.Linear(h, n_channels * 2)        # (γ, β)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x    : (N, C, T)
        cond : (N, cond_dim)
        """
        # If cond is >2D, average all but the last dim.
        if cond.dim() > 2:
            dims = tuple(range(1, cond.dim() - 1))
            cond = cond.mean(dim=dims)
        gamma, beta = self.to_gb(cond).chunk(2, dim=-1)      # (N, C)
        gamma = gamma.unsqueeze(-1)
        beta  = beta .unsqueeze(-1)
        return x * (1 + gamma) + beta                        # residual-style FiLM
    





# ---------------------------------------------------------------------------
# 3. Simplified Decoder
# ---------------------------------------------------------------------------
class ReconstructionDecoder(nn.Module):
    """
    Simplified FiLM Decoder with PixelShuffle Upsampling and Normalization.
    z_q is used directly, z_prior conditions via FiLM.
    """
    def __init__(self,
                 latent_dim: int,       # z_q dim
                 prior_factor_dim: int, # z_prior dim (used for FiLM conditioning)
                 output_T: int,
                 output_C: int = 158,
                 hidden_channels: int = 128,
                 initial_T: int = 5,
                 num_blocks: int | None = None,
                 norm_type: str = 'none',  # 'batch', 'group', 'layer', 'instance'
                 num_groups: int = 8):     # number of groups for GroupNorm
        super().__init__()

        # 0. T resolution check
        if num_blocks is None:
            ratio = output_T / initial_T
            if not (ratio.is_integer() and math.log2(ratio).is_integer()):
                raise ValueError("initial_T * 2^k must equal output_T.")
            num_blocks = int(math.log2(ratio))
        if initial_T * (2 ** num_blocks) != output_T:
            raise ValueError("T mismatch: initial_T * (2^num_blocks) must equal output_T.")

        self.H = hidden_channels
        self.T0 = initial_T
        self.K = num_blocks
        self.norm_type = norm_type.lower()
        self.num_groups = num_groups

        valid_norms = ['batch', 'group', 'layer', 'instance', 'none']
        if self.norm_type not in valid_norms:
            raise ValueError(f"norm_type must be one of {valid_norms}, got {norm_type}")

        # 1. Base embed (uses z_q directly)
        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * initial_T),
            nn.GELU()
        )

        # 2. Upsample stack
        self.conv_expand_layers = nn.ModuleList()
        self.norm_expand_layers = nn.ModuleList()
        self.shuffle_layers     = nn.ModuleList()
        self.conv_rest_layers   = nn.ModuleList()
        self.norm_rest_layers   = nn.ModuleList()
        self.film_layers        = nn.ModuleList()
        self.skip_layers        = nn.ModuleList()

        for _ in range(self.K):
            # Conv expand: H → 2H
            self.conv_expand_layers.append(nn.Conv1d(hidden_channels, hidden_channels * 2,
                                                     kernel_size=3, padding=1))
            # Normalization after expand
            self.norm_expand_layers.append(self._make_norm_layer(hidden_channels * 2))
            
            # PixelShuffle: 2H,T → H,2T
            self.shuffle_layers.append(PixelShuffle1D(2))
            
            # Conv rest: H → H
            self.conv_rest_layers.append(nn.Conv1d(hidden_channels, hidden_channels,
                                                   kernel_size=3, padding=1))
            # Normalization after rest
            self.norm_rest_layers.append(self._make_norm_layer(hidden_channels))
            
            # FiLM conditioning
            self.film_layers.append(FiLM1D(prior_factor_dim, hidden_channels))

            # Skip connection: upsample previous resolution to match current output
            self.skip_layers.append(
                nn.ConvTranspose1d(hidden_channels, hidden_channels,
                                   kernel_size=4, stride=2, padding=1)
            )

        # 3. Output projection
        self.conv_out = nn.Conv1d(hidden_channels, output_C, kernel_size=1)

    def _make_norm_layer(self, num_channels: int) -> nn.Module:
        """Build the configured normalization layer for `num_channels` channels."""
        if self.norm_type == 'batch':
            return nn.BatchNorm1d(num_channels)
        elif self.norm_type == 'group':
            # Clamp group count if it exceeds channel count, then make it divide channels.
            groups = min(self.num_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, num_channels)
        elif self.norm_type == 'layer':
            return nn.GroupNorm(1, num_channels)  # LayerNorm for 1D conv
        elif self.norm_type == 'instance':
            return nn.GroupNorm(num_channels, num_channels)  # InstanceNorm for 1D conv
        elif self.norm_type == 'none':
            return nn.Identity()  # No normalization
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

    def forward(self, z_q: torch.Tensor, z_prior: torch.Tensor):
        B = z_q.size(0)
        
        # Base embedding: (B, latent_dim) → (B, H, T0)
        x = self.fc_in(z_q).view(B, self.H, self.T0)

        # Upsample blocks
        for k in range(self.K):
            residual = x

            # 1. Conv expand + Norm + GELU: (B, H, T) → (B, 2H, T)
            x = self.conv_expand_layers[k](x)
            x = self.norm_expand_layers[k](x)
            x = F.gelu(x)
            
            # 2. PixelShuffle: (B, 2H, T) → (B, H, 2T)
            x = self.shuffle_layers[k](x)
            
            # 3. Conv rest + Norm + GELU: (B, H, 2T) → (B, H, 2T)
            x = self.conv_rest_layers[k](x)
            x = self.norm_rest_layers[k](x)
            x = F.gelu(x)
            
            # 4. FiLM conditioning: (B, H, 2T) → (B, H, 2T)
            x = self.film_layers[k](x, z_prior)

            # 5. Residual skip: bring previous-resolution features forward
            skip = self.skip_layers[k](residual)
            x = x + skip

        # Output projection: (B, H, final_T) → (B, output_C, final_T) → (B, final_T, output_C)
        x_rec_permuted = self.conv_out(x)
        return x_rec_permuted.permute(0, 2, 1)



class SequencePredictorGRU(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 prior_factor_dim,
                 hidden_dim,
                 output_seq_len=10, 
                 output_dim=1, 
                 num_gru_layers=1,
                 dropout=0.1):
        super().__init__()
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.output_dim = output_dim

        # 1. Initial hidden state generator
        self.init_mlp = nn.Sequential(
            nn.Linear(latent_dim + prior_factor_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * num_gru_layers),
            nn.Tanh()  # bound hidden state via tanh
        )

        # 2. Multi-layer GRU
        self.gru = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True
        )

        # 3. Output projection with residual connection
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 4. Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.1)

    def forward(self, z_q, f_prior):
        """
        z_q: (N, latent_dim) - quantized latent vectors
        f_prior: (N, prior_factor_dim) - prior factors
        """
        N = z_q.shape[0]

        combined_input = torch.cat((z_q, f_prior), dim=1)
        h_0 = self.init_mlp(combined_input)
        h_0 = h_0.view(N, self.num_gru_layers, self.hidden_dim).transpose(0, 1).contiguous()

        start_tokens = self.start_token.expand(N, 1, self.output_dim)

        outputs = []
        hidden = h_0
        current_input = start_tokens

        # Autoregressive generation (no teacher forcing).
        for step in range(self.output_seq_len):
            gru_out, hidden = self.gru(current_input, hidden)
            output_step = self.output_mlp(gru_out.squeeze(1))
            outputs.append(output_step)
            current_input = output_step.unsqueeze(1)

        predictions = torch.stack(outputs, dim=1)  # (N, output_seq_len, output_dim)

        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)

        return predictions

    def forward_with_teacher_forcing(self, z_q, f_prior, target_sequence=None):
        """
        Training forward pass with teacher forcing.
        target_sequence: (N, output_seq_len, output_dim) or (N, output_seq_len) if output_dim==1.
        """
        if target_sequence is None:
            return self.forward(z_q, f_prior)
            
        N = z_q.shape[0]
        
        # 1. Generate initial hidden state
        combined_input = torch.cat((z_q, f_prior), dim=1)
        h_0 = self.init_mlp(combined_input)
        h_0 = h_0.view(N, self.num_gru_layers, self.hidden_dim).transpose(0, 1).contiguous()

        # 2. Prepare input sequence for teacher forcing
        if target_sequence.dim() == 2:
            target_sequence = target_sequence.unsqueeze(-1)  # (N, seq_len, 1)
            
        # Shift target sequence: [start_token, target[:-1]]
        start_tokens = self.start_token.expand(N, 1, self.output_dim)
        inputs = torch.cat([start_tokens, target_sequence[:, :-1]], dim=1)  # (N, seq_len, output_dim)
        
        # 3. Single GRU forward pass (much more efficient)
        gru_outputs, _ = self.gru(inputs, h_0)  # (N, seq_len, hidden_dim)
        
        # 4. Output projection for all timesteps at once
        predictions = self.output_mlp(gru_outputs.reshape(-1, self.hidden_dim))  # (N*seq_len, output_dim)
        predictions = predictions.view(N, self.output_seq_len, self.output_dim)  # (N, seq_len, output_dim)
        
        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)
            
        return predictions
