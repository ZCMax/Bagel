from typing import List, Tuple

import torch
import torch.nn.functional as F


def split_tokens_by_hw(tokens: torch.Tensor, hw_list: List[Tuple[int, int]]) -> List[torch.Tensor]:
    out = []
    start = 0
    for h, w in hw_list:
        n = int(h) * int(w)
        out.append(tokens[start : start + n])
        start += n
    if start != tokens.shape[0]:
        raise ValueError(
            f"Token split mismatch: consumed={start}, total={tokens.shape[0]}, len(hw_list)={len(hw_list)}"
        )
    return out


def unpatchify_latent_tokens(
    tokens: torch.Tensor,
    h: int,
    w: int,
    latent_patch_size: int,
    latent_channel: int,
) -> torch.Tensor:
    # tokens: [h*w, p*p*c] -> latent: [c, h*p, w*p]
    p = int(latent_patch_size)
    c = int(latent_channel)
    x = tokens.reshape(int(h), int(w), p, p, c)
    x = torch.einsum("hwpqc->chpwq", x)
    x = x.reshape(c, int(h) * p, int(w) * p)
    return x


def se3_l1_pose_loss(pred_delta: torch.Tensor, gt_delta: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask is None or valid_mask.sum().item() == 0:
        return pred_delta.new_tensor(0.0)
    pred = pred_delta[valid_mask]
    gt = gt_delta[valid_mask]
    return F.l1_loss(pred, gt, reduction="mean")


def scale_invariant_log_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    si_lambda: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    # pred_depth/gt_depth: [H, W] or [1, H, W]
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)

    valid = valid_mask & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    valid = valid & (pred_depth > 0) & (gt_depth > 0)
    if valid.sum().item() == 0:
        return pred_depth.new_tensor(0.0)

    log_diff = torch.log(pred_depth[valid] + eps) - torch.log(gt_depth[valid] + eps)
    mean = log_diff.mean()
    return (log_diff * log_diff).mean() - si_lambda * (mean * mean)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(eps)
    return (x * mask.float()).sum() / denom


def _ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x, y: [1, 3, H, W], range [-1, 1]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = num / (den + 1e-6)
    return ssim.clamp(-1.0, 1.0)


def reproject_source_to_target_nn(
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_k: torch.Tensor,
    target_k: torch.Tensor,
    source_pose_c2w: torch.Tensor,
    target_pose_c2w: torch.Tensor,
    depth_tol: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward-project source pixels into target plane using nearest-neighbor splat + z-buffer.

    Args:
      source_rgb: [3, H, W], in [-1, 1]
      source_depth: [H, W], metric-like depth
      target_depth: [H, W], metric-like depth
      source_k, target_k: [3, 3]
      source_pose_c2w, target_pose_c2w: [4, 4]
    Returns:
      warped_rgb: [3, H, W]
      valid_mask: [H, W] bool
    """
    device = source_rgb.device
    dtype = source_rgb.dtype
    _, h, w = source_rgb.shape

    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x_grid)

    z_src = source_depth.to(dtype)
    valid_src = torch.isfinite(z_src) & (z_src > 0)
    if valid_src.sum().item() == 0:
        return torch.zeros_like(source_rgb), torch.zeros((h, w), device=device, dtype=torch.bool)

    pix = torch.stack([x_grid, y_grid, ones], dim=0).reshape(3, -1)  # [3, HW]
    z_flat = z_src.reshape(-1)
    valid_flat = valid_src.reshape(-1)

    src_k_inv = torch.linalg.inv(source_k.to(dtype))
    tgt_k = target_k.to(dtype)

    xyz_src = (src_k_inv @ pix) * z_flat.unsqueeze(0)  # [3, HW]
    xyz_src = xyz_src[:, valid_flat]  # [3, N]
    n = xyz_src.shape[1]
    ones_n = torch.ones((1, n), device=device, dtype=dtype)
    xyz_src_h = torch.cat([xyz_src, ones_n], dim=0)  # [4, N]

    src_to_world = source_pose_c2w.to(dtype)
    world_to_tgt = torch.linalg.inv(target_pose_c2w.to(dtype))
    xyz_world_h = src_to_world @ xyz_src_h
    xyz_tgt_h = world_to_tgt @ xyz_world_h
    xyz_tgt = xyz_tgt_h[:3]  # [3, N]

    z_tgt = xyz_tgt[2]
    positive_z = z_tgt > 1e-6
    if positive_z.sum().item() == 0:
        return torch.zeros_like(source_rgb), torch.zeros((h, w), device=device, dtype=torch.bool)

    xyz_tgt = xyz_tgt[:, positive_z]
    z_tgt = z_tgt[positive_z]

    uvw = tgt_k @ xyz_tgt
    u = uvw[0] / (uvw[2] + 1e-6)
    v = uvw[1] / (uvw[2] + 1e-6)
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    in_view = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    if in_view.sum().item() == 0:
        return torch.zeros_like(source_rgb), torch.zeros((h, w), device=device, dtype=torch.bool)

    u_int = u_int[in_view]
    v_int = v_int[in_view]
    z_tgt = z_tgt[in_view]
    src_idx_all = torch.nonzero(valid_flat, as_tuple=False).flatten()[positive_z][in_view]
    flat_idx = v_int * w + u_int

    tgt_depth_flat = target_depth.to(dtype).reshape(-1)
    depth_tgt_pix = tgt_depth_flat[flat_idx]
    visible = torch.isfinite(depth_tgt_pix) & (depth_tgt_pix > 0)
    visible = visible & (torch.abs(z_tgt - depth_tgt_pix) <= depth_tol * torch.clamp(depth_tgt_pix, min=1e-3))
    if visible.sum().item() == 0:
        return torch.zeros_like(source_rgb), torch.zeros((h, w), device=device, dtype=torch.bool)

    flat_idx = flat_idx[visible]
    z_tgt = z_tgt[visible]
    src_idx_all = src_idx_all[visible]

    min_z = torch.full((h * w,), float("inf"), device=device, dtype=dtype)
    min_z.scatter_reduce_(0, flat_idx, z_tgt, reduce="amin", include_self=True)
    keep = z_tgt <= (min_z[flat_idx] + 1e-4)
    if keep.sum().item() == 0:
        return torch.zeros_like(source_rgb), torch.zeros((h, w), device=device, dtype=torch.bool)

    flat_idx = flat_idx[keep]
    src_idx_all = src_idx_all[keep]

    src_rgb_flat = source_rgb.reshape(3, -1)
    warped_flat = torch.zeros((3, h * w), device=device, dtype=dtype)
    counts = torch.zeros((h * w,), device=device, dtype=dtype)
    ones_count = torch.ones_like(flat_idx, dtype=dtype, device=device)
    counts.scatter_add_(0, flat_idx, ones_count)
    for c in range(3):
        warped_flat[c].scatter_add_(0, flat_idx, src_rgb_flat[c, src_idx_all])

    nonzero = counts > 0
    warped_flat[:, nonzero] = warped_flat[:, nonzero] / counts[nonzero].unsqueeze(0)
    warped_rgb = warped_flat.reshape(3, h, w)
    valid_mask = nonzero.reshape(h, w)
    return warped_rgb, valid_mask


def reprojection_consistency_loss(
    generated_rgb: torch.Tensor,
    source_rgb: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    source_k: torch.Tensor,
    target_k: torch.Tensor,
    source_pose_c2w: torch.Tensor,
    target_pose_c2w: torch.Tensor,
    ssim_weight: float = 0.3,
    depth_tol: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      generated_rgb/source_rgb: [3, H, W], range [-1, 1]
      depth: [H, W], intrinsics [3, 3], poses [4, 4]
    Returns:
      loss, valid_mask
    """
    warped_rgb, valid_mask = reproject_source_to_target_nn(
        source_rgb=source_rgb,
        source_depth=source_depth,
        target_depth=target_depth,
        source_k=source_k,
        target_k=target_k,
        source_pose_c2w=source_pose_c2w,
        target_pose_c2w=target_pose_c2w,
        depth_tol=depth_tol,
    )
    if valid_mask.sum().item() == 0:
        return generated_rgb.new_tensor(0.0), valid_mask

    gen = generated_rgb.unsqueeze(0)
    wrp = warped_rgb.unsqueeze(0)
    l1_map = (gen - wrp).abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
    ssim_map = _ssim_map(gen, wrp).mean(dim=1, keepdim=True)
    photo = (1.0 - ssim_map) * 0.5
    mixed = (1.0 - ssim_weight) * l1_map + ssim_weight * photo
    loss = _masked_mean(mixed.squeeze(0).squeeze(0), valid_mask)
    return loss, valid_mask
