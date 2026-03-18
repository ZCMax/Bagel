import torch
from torch import nn
import torch.nn.functional as F


class GeoPoseHead(nn.Module):
    """Predict relative pose delta (se3 6D) from source/generated global features."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 6):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, (x.shape[-1],))
        return self.net(x)


class GeoDepthHead(nn.Module):
    """Predict per-token depth from generated latent tokens."""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, token_feats: torch.Tensor) -> torch.Tensor:
        # token_feats: [N, D]
        token_feats = F.layer_norm(token_feats, (token_feats.shape[-1],))
        return self.net(token_feats).squeeze(-1)
