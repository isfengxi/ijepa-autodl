# src/utils/patchify.py
import torch

def patchify_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: [B, C, H, W]
    return: [B, N, patch_dim], N=(H/P)*(W/P), patch_dim=C*P*P
    """
    B, C, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, f"H,W must be divisible by patch_size={P}"
    h = H // P
    w = W // P
    x = x.reshape(B, C, h, P, w, P).permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B, h * w, C * P * P)

def gather_patches(patches: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    patches: [B, N, Dp]
    idx: [B, K]  (indices of target/masked patches)
    return: [B, K, Dp]
    """
    B, N, Dp = patches.shape
    assert idx.dim() == 2 and idx.shape[0] == B
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, Dp)  # [B, K, Dp]
    return torch.gather(patches, dim=1, index=idx_exp)
