import torch
from torch import nn
import torch.nn.functional as F


class SpatialDistillAdapter(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: int = 0,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.normalize_input = normalize_input

        if hidden_dim and hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim),
            )
        else:
            self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_features: torch.Tensor) -> torch.Tensor:
        x = student_features
        if self.normalize_input:
            x = F.layer_norm(x, (x.shape[-1],))
        return self.proj(x)


def compute_alignment_loss(
    pred_teacher_features: torch.Tensor,
    teacher_features: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    if loss_type == "cosine":
        pred = F.normalize(pred_teacher_features, dim=-1)
        target = F.normalize(teacher_features, dim=-1)
        return 1.0 - (pred * target).sum(dim=-1).mean()
    if loss_type == "mse":
        return F.mse_loss(pred_teacher_features, teacher_features, reduction="mean")
    raise ValueError(f"Unsupported distill loss_type: {loss_type}")
