# src/models/pixel_decoder.py
import torch.nn as nn

class PixelDecoderMLP(nn.Module):
    """
    token: [B, K, D]
    out : [B, K, patch_dim]
    """
    def __init__(self, token_dim: int, patch_dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        h = token_dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, patch_dim),
        )

    def forward(self, x):
        return self.net(x)
