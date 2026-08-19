import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, n_feature, hidden_dim):
        super().__init__()
        self.n_feature = n_feature
        self.hidden_dim = hidden_dim

        self.normalize = nn.LayerNorm(n_feature)
        self.linear = nn.Linear(n_feature, n_feature)
        self.leakyrelu = nn.LeakyReLU()
        self.get_h = nn.GRU(n_feature, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (B, T, C)
        x = self.linear(x)
        x = self.normalize(x)
        x = self.leakyrelu(x)

        _, h_n = self.get_h(x)  # h_n: (1, B, H) for num_layers=1
        return h_n.squeeze(0)   # (B, H)

class CrossAssetTransformerEncoder(nn.Module):
    """Transformer encoder that models cross-asset relationships."""
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_layers,
                 d_out,
                 dim_feedforward=None):

        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * embed_dim

        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim))  # kept for state-dict compatibility

            def forward(self, x):
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                x = x / rms * self.weight
                return x

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        
        # Swap default LayerNorm for RMSNorm.
        encoder_layer.norm1 = RMSNorm(embed_dim)
        encoder_layer.norm2 = RMSNorm(embed_dim)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, d_out)
        )


    def forward(self, temporal_summaries):
        # Expect (N_t, embed_dim) or (B, N_t, embed_dim). Treat (N_t, embed_dim)
        # as Batch=1, Seq=N_t. Variable N_t would require padding/masking.
        if temporal_summaries.dim() == 2:
            temporal_summaries = temporal_summaries.unsqueeze(0)  # (1, N_t, embed_dim)

        refined_representation = self.transformer_encoder(temporal_summaries)

        if refined_representation.shape[0] == 1:
            refined_representation = refined_representation.squeeze(0)
        out = self.out_layer(refined_representation)  # project to vq_dim (factor dim)
        return out


class SpatialEncoder(nn.Module):
    """Pretrained VQ-VAE encoder: Temporal GRU + Cross-Asset Transformer."""
    def __init__(self,
                 input_features_C,
                 T_window,
                 gru_hidden_size,
                 num_transformer_heads,
                 num_transformer_layers,
                 final_embed_dim_d
                 ):
        super().__init__()
        self.T_window = T_window
        self.C = input_features_C
        self.gru_hidden_size = gru_hidden_size

        # 1. Temporal feature extractor (GRU)
        self.temporal_extractor = FeatureExtractor(input_features_C, gru_hidden_size)

        # 2. Cross-asset attention (Transformer). Input dim matches GRU output.
        self.cross_asset_transformer = CrossAssetTransformerEncoder(embed_dim=gru_hidden_size,
                                                                    num_heads=num_transformer_heads,
                                                                    num_layers=num_transformer_layers,
                                                                    d_out=final_embed_dim_d)


    def forward(self, x_batch):
        # x_batch: (N_t, T_window, C) — N_t stocks at timestep t.
        # Step 1: per-stock GRU summary -> (N_t, gru_hidden_size).
        temporal_summaries = self.temporal_extractor(x_batch)

        # Step 2: cross-asset Transformer -> (N_t, final_embed_dim_d).
        refined_representation = self.cross_asset_transformer(temporal_summaries)

        return refined_representation