import torch
from torch import nn
import torch.nn.functional as F


class PoseAuxHead(nn.Module):
    def __init__(self, student_dim: int, hidden_dim: int = 1024, out_dim: int = 6):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(student_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, (x.shape[-1],))
        return self.net(x)


def compute_pose_aux_loss(
    pred_delta: torch.Tensor,
    gt_delta: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_type: str = "smooth_l1",
    trans_weight: float = 1.0,
    rot_weight: float = 1.0,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    if valid_mask is None or valid_mask.sum().item() == 0:
        return pred_delta.new_tensor(0.0)

    pred_v = pred_delta[valid_mask]
    gt_v = gt_delta[valid_mask]

    pred_t, pred_r = pred_v[:, :3], pred_v[:, 3:]
    gt_t, gt_r = gt_v[:, :3], gt_v[:, 3:]

    if loss_type == "smooth_l1":
        loss_t = F.smooth_l1_loss(pred_t, gt_t, beta=smooth_l1_beta, reduction="mean")
        loss_r = F.smooth_l1_loss(pred_r, gt_r, beta=smooth_l1_beta, reduction="mean")
    elif loss_type == "mse":
        loss_t = F.mse_loss(pred_t, gt_t, reduction="mean")
        loss_r = F.mse_loss(pred_r, gt_r, reduction="mean")
    else:
        raise ValueError(f"Unsupported pose aux loss_type: {loss_type}")

    return trans_weight * loss_t + rot_weight * loss_r
