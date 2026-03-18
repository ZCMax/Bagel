import math
from typing import Dict, Optional, Tuple

import torch


class PoseProbeBuffer:
    """A fixed-size CPU buffer for pose-probe features and labels."""

    def __init__(self, max_samples: int = 2048):
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        self.max_samples = int(max_samples)
        self._features: Optional[torch.Tensor] = None  # [N, D], float32, cpu
        self._targets: Optional[torch.Tensor] = None   # [N, 6], float32, cpu

    @property
    def num_samples(self) -> int:
        if self._features is None:
            return 0
        return int(self._features.shape[0])

    def add(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if features is None or targets is None:
            return
        if features.numel() == 0 or targets.numel() == 0:
            return
        if features.dim() != 2:
            raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
        if targets.dim() != 2 or targets.shape[1] != 6:
            raise ValueError(f"targets must be [N, 6], got {tuple(targets.shape)}")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have the same first dimension")

        f = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
        t = targets.detach().to(device="cpu", dtype=torch.float32).contiguous()

        if self._features is None:
            self._features = f
            self._targets = t
        else:
            if f.shape[1] != self._features.shape[1]:
                raise ValueError(
                    f"feature dim mismatch: {f.shape[1]} vs existing {self._features.shape[1]}"
                )
            self._features = torch.cat([self._features, f], dim=0)
            self._targets = torch.cat([self._targets, t], dim=0)

        if self._features.shape[0] > self.max_samples:
            self._features = self._features[-self.max_samples :]
            self._targets = self._targets[-self.max_samples :]

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._features is None:
            return torch.empty(0, 0, dtype=torch.float32), torch.empty(0, 6, dtype=torch.float32)
        return self._features, self._targets


def _append_bias(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=1)


def _fit_ridge_probe(train_x: torch.Tensor, train_y: torch.Tensor, ridge: float) -> torch.Tensor:
    # Closed-form ridge regression: W = (X^T X + lambda I)^-1 X^T Y
    x = _append_bias(train_x).to(dtype=torch.float64)
    y = train_y.to(dtype=torch.float64)
    dim = x.shape[1]
    xtx = x.T @ x
    reg = torch.eye(dim, dtype=torch.float64, device=x.device) * float(ridge)
    reg[-1, -1] = 0.0  # do not regularize bias
    rhs = x.T @ y
    try:
        w = torch.linalg.solve(xtx + reg, rhs)
    except RuntimeError:
        w = torch.linalg.pinv(xtx + reg) @ rhs
    return w


def _predict_probe(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x = _append_bias(x).to(dtype=torch.float64)
    return (x @ w).to(dtype=torch.float32)


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    # Rodrigues formula in batch form, with small-angle handling.
    if rotvec.numel() == 0:
        return torch.empty((0, 3, 3), dtype=rotvec.dtype, device=rotvec.device)
    if rotvec.dim() != 2 or rotvec.shape[1] != 3:
        raise ValueError(f"rotvec must be [N, 3], got {tuple(rotvec.shape)}")

    wx, wy, wz = rotvec[:, 0], rotvec[:, 1], rotvec[:, 2]
    zeros = torch.zeros_like(wx)
    K = torch.stack(
        [
            torch.stack([zeros, -wz, wy], dim=1),
            torch.stack([wz, zeros, -wx], dim=1),
            torch.stack([-wy, wx, zeros], dim=1),
        ],
        dim=1,
    )  # [N, 3, 3]
    K2 = K @ K

    theta = torch.linalg.norm(rotvec, dim=1, keepdim=True).unsqueeze(-1)  # [N,1,1]
    theta2 = theta * theta
    eps = 1e-6
    small = theta < eps

    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0,
        (1.0 - torch.cos(theta)) / theta2,
    )

    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).unsqueeze(0).expand(rotvec.shape[0], -1, -1)
    return identity + a * K + b * K2


def _rotation_geodesic_deg(pred_rotvec: torch.Tensor, gt_rotvec: torch.Tensor) -> torch.Tensor:
    pred_r = _rotvec_to_matrix(pred_rotvec)
    gt_r = _rotvec_to_matrix(gt_rotvec)
    rel = pred_r.transpose(1, 2) @ gt_r
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / math.pi)


def _compute_pose_metrics(
    pred_delta: torch.Tensor,
    gt_delta: torch.Tensor,
    trans_success_thr: float,
    rot_success_thr_deg: float,
) -> Dict[str, float]:
    diff = pred_delta - gt_delta
    mse = (diff * diff).mean()
    trans_l1 = diff[:, :3].abs().mean()
    trans_l2 = torch.linalg.norm(diff[:, :3], dim=1).mean()
    rot_vec_l1 = diff[:, 3:].abs().mean()
    rot_geo_deg = _rotation_geodesic_deg(pred_delta[:, 3:], gt_delta[:, 3:])
    rot_geo_deg_mean = rot_geo_deg.mean()

    success = (
        (torch.linalg.norm(diff[:, :3], dim=1) < trans_success_thr)
        & (rot_geo_deg < rot_success_thr_deg)
    ).float().mean()

    return {
        "mse": float(mse.item()),
        "trans_l1": float(trans_l1.item()),
        "trans_l2": float(trans_l2.item()),
        "rot_vec_l1": float(rot_vec_l1.item()),
        "rot_geodesic_deg": float(rot_geo_deg_mean.item()),
        "success": float(success.item()),
    }


def evaluate_pose_linear_probe(
    features: torch.Tensor,
    pose_delta: torch.Tensor,
    val_ratio: float = 0.2,
    ridge: float = 1e-4,
    seed: int = 3407,
    trans_success_thr: float = 0.1,
    rot_success_thr_deg: float = 5.0,
) -> Dict[str, float]:
    if features.dim() != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if pose_delta.dim() != 2 or pose_delta.shape[1] != 6:
        raise ValueError(f"pose_delta must be [N, 6], got {tuple(pose_delta.shape)}")
    if features.shape[0] != pose_delta.shape[0]:
        raise ValueError("features and pose_delta must have the same first dimension")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    num_samples = int(features.shape[0])
    if num_samples < 4:
        return {
            "probe_num_samples": float(num_samples),
            "probe_train_samples": 0.0,
            "probe_val_samples": 0.0,
            "probe_valid": 0.0,
        }

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=g)
    val_size = max(1, int(round(num_samples * val_ratio)))
    train_size = num_samples - val_size
    if train_size < 2:
        return {
            "probe_num_samples": float(num_samples),
            "probe_train_samples": float(train_size),
            "probe_val_samples": float(val_size),
            "probe_valid": 0.0,
        }

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    x_train = features[train_idx].to(dtype=torch.float32, device="cpu")
    y_train = pose_delta[train_idx].to(dtype=torch.float32, device="cpu")
    x_val = features[val_idx].to(dtype=torch.float32, device="cpu")
    y_val = pose_delta[val_idx].to(dtype=torch.float32, device="cpu")

    w = _fit_ridge_probe(x_train, y_train, ridge=float(ridge))
    pred_train = _predict_probe(x_train, w)
    pred_val = _predict_probe(x_val, w)

    train_metrics = _compute_pose_metrics(
        pred_delta=pred_train,
        gt_delta=y_train,
        trans_success_thr=trans_success_thr,
        rot_success_thr_deg=rot_success_thr_deg,
    )
    val_metrics = _compute_pose_metrics(
        pred_delta=pred_val,
        gt_delta=y_val,
        trans_success_thr=trans_success_thr,
        rot_success_thr_deg=rot_success_thr_deg,
    )

    out = {
        "probe_num_samples": float(num_samples),
        "probe_train_samples": float(train_size),
        "probe_val_samples": float(val_size),
        "probe_valid": 1.0,
    }
    for key, value in train_metrics.items():
        out[f"probe_train_{key}"] = value
    for key, value in val_metrics.items():
        out[f"probe_val_{key}"] = value
    return out
