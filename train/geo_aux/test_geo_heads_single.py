import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure repo-root imports work when launching via absolute script path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.geo_aux.heads import GeoDepthHead, GeoPoseHead
from train.pose_aux.pose_utils import parse_pose_matrix, relative_pose_delta_se3


def _load_ae_fn():
    ae_py = os.path.join(REPO_ROOT, "modeling", "autoencoder.py")
    spec = importlib.util.spec_from_file_location("bagel_local_autoencoder", ae_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load autoencoder module from: {ae_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load_ae"):
        raise AttributeError("autoencoder.py does not define load_ae")
    return mod.load_ae


def _get_by_image_name(image_name: str):
    # Lazy import to avoid hard dependency on cv2 for --help usage.
    from train.geo_aux.get_more import _get_by_image_name as _impl

    return _impl(image_name)


def _str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


def _resolve_geo_aux_ckpt(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
        cand = os.path.join(abs_path, "geo_aux.pt")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"No geo_aux.pt under directory: {abs_path}")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"geo_aux checkpoint not found: {abs_path}")
    return abs_path


def _infer_pose_head_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    if "net.0.weight" in sd:
        hidden_dim, in_dim = sd["net.0.weight"].shape
        out_dim, hidden_dim_2 = sd["net.2.weight"].shape
        if hidden_dim != hidden_dim_2:
            raise ValueError("Pose head hidden dim mismatch in checkpoint.")
        return int(in_dim), int(hidden_dim), int(out_dim)
    out_dim, in_dim = sd["net.weight"].shape
    return int(in_dim), 0, int(out_dim)


def _infer_depth_head_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "net.0.weight" in sd:
        hidden_dim, in_dim = sd["net.0.weight"].shape
        out_dim, hidden_dim_2 = sd["net.2.weight"].shape
        if int(out_dim) != 1:
            raise ValueError(f"Depth head output dim should be 1, got {out_dim}")
        if hidden_dim != hidden_dim_2:
            raise ValueError("Depth head hidden dim mismatch in checkpoint.")
        return int(in_dim), int(hidden_dim)
    out_dim, in_dim = sd["net.weight"].shape
    if int(out_dim) != 1:
        raise ValueError(f"Depth head output dim should be 1, got {out_dim}")
    return int(in_dim), 0


def _load_geo_heads(geo_aux_ckpt: str, device: torch.device):
    state = torch.load(geo_aux_ckpt, map_location="cpu")
    if "pose_head" not in state or "depth_head" not in state:
        raise KeyError("geo_aux checkpoint must contain pose_head and depth_head")

    pose_sd = state["pose_head"]
    depth_sd = state["depth_head"]

    pose_in_dim, pose_hidden_dim, pose_out_dim = _infer_pose_head_dims(pose_sd)
    if pose_out_dim != 6:
        raise ValueError(f"Pose head output dim should be 6, got {pose_out_dim}")
    depth_in_dim, depth_hidden_dim = _infer_depth_head_dims(depth_sd)

    pose_head = GeoPoseHead(in_dim=pose_in_dim, hidden_dim=pose_hidden_dim, out_dim=6)
    depth_head = GeoDepthHead(in_dim=depth_in_dim, hidden_dim=depth_hidden_dim)

    pose_head.load_state_dict(pose_sd, strict=True)
    depth_head.load_state_dict(depth_sd, strict=True)

    pose_head = pose_head.to(device=device)
    depth_head = depth_head.to(device=device)
    pose_head.eval()
    depth_head.eval()
    return pose_head, depth_head


def _crop_to_factor(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    h2 = h - (h % factor)
    w2 = w - (w % factor)
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Image too small for factor={factor}: {(h, w)}")
    return img[:h2, :w2]


def _prepare_rgb_tensor(rgb: np.ndarray, factor: int, auto_bgr_to_rgb: bool, image_name: str) -> torch.Tensor:
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.shape[2] > 3:
        rgb = rgb[..., :3]

    # get_more returns BGR for matterport3d via cv2.imread; align to RGB.
    if auto_bgr_to_rgb and image_name.lower().startswith("matterport3d/"):
        rgb = rgb[..., ::-1]

    rgb = _crop_to_factor(rgb, factor)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    if float(x.max().item()) > 1.5:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)
    x = (x - 0.5) / 0.5
    return x


def _patchify_latent(latent_chw: torch.Tensor, latent_patch_size: int) -> Tuple[torch.Tensor, int, int]:
    p = int(latent_patch_size)
    c, h, w = latent_chw.shape
    h_tok = h // p
    w_tok = w // p
    if h_tok <= 0 or w_tok <= 0:
        raise ValueError(f"Invalid latent shape {tuple(latent_chw.shape)} for latent_patch_size={p}")
    latent = latent_chw[:, : h_tok * p, : w_tok * p].reshape(c, h_tok, p, w_tok, p)
    tokens = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * c)
    return tokens, h_tok, w_tok


def _silog(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, si_lambda: float = 0.5) -> float:
    mask = valid & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
    if mask.sum().item() == 0:
        return float("nan")
    d = torch.log(pred[mask] + 1e-6) - torch.log(gt[mask] + 1e-6)
    return float(((d * d).mean() - si_lambda * (d.mean() * d.mean())).item())


def _depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, si_lambda: float = 0.5) -> Dict[str, float]:
    mask = valid & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
    if mask.sum().item() == 0:
        return {"num_valid": 0.0}

    pred_v = pred[mask]
    gt_v = gt[mask]

    abs_rel_raw = (pred_v - gt_v).abs().div(gt_v).mean()
    rmse_raw = torch.sqrt(((pred_v - gt_v) ** 2).mean())

    scale = torch.median(gt_v) / torch.clamp(torch.median(pred_v), min=1e-6)
    pred_s = pred_v * scale

    abs_rel_scaled = (pred_s - gt_v).abs().div(gt_v).mean()
    rmse_scaled = torch.sqrt(((pred_s - gt_v) ** 2).mean())

    ratio = torch.maximum(pred_s / torch.clamp(gt_v, min=1e-6), gt_v / torch.clamp(pred_s, min=1e-6))
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < (1.25 ** 2)).float().mean()
    delta3 = (ratio < (1.25 ** 3)).float().mean()

    return {
        "num_valid": float(mask.sum().item()),
        "scale_median": float(scale.item()),
        "silog_raw": _silog(pred, gt, mask, si_lambda=si_lambda),
        "silog_scaled": _silog(pred * scale, gt, mask, si_lambda=si_lambda),
        "abs_rel_raw": float(abs_rel_raw.item()),
        "rmse_raw": float(rmse_raw.item()),
        "abs_rel_scaled": float(abs_rel_scaled.item()),
        "rmse_scaled": float(rmse_scaled.item()),
        "delta1_scaled": float(delta1.item()),
        "delta2_scaled": float(delta2.item()),
        "delta3_scaled": float(delta3.item()),
    }


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    # rotvec: [N, 3]
    wx, wy, wz = rotvec[:, 0], rotvec[:, 1], rotvec[:, 2]
    zeros = torch.zeros_like(wx)
    k = torch.stack(
        [
            torch.stack([zeros, -wz, wy], dim=1),
            torch.stack([wz, zeros, -wx], dim=1),
            torch.stack([-wy, wx, zeros], dim=1),
        ],
        dim=1,
    )
    k2 = k @ k
    theta = torch.linalg.norm(rotvec, dim=1, keepdim=True).unsqueeze(-1)
    theta2 = theta * theta
    eps = 1e-6
    small = theta < eps
    a = torch.where(small, 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0, (1.0 - torch.cos(theta)) / theta2)
    i = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).unsqueeze(0).expand(rotvec.shape[0], -1, -1)
    return i + a * k + b * k2


def _rotation_geodesic_deg(pred_rotvec: torch.Tensor, gt_rotvec: torch.Tensor) -> torch.Tensor:
    pred_r = _rotvec_to_matrix(pred_rotvec)
    gt_r = _rotvec_to_matrix(gt_rotvec)
    rel = pred_r.transpose(1, 2) @ gt_r
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / math.pi)


def _colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if valid.sum() <= 0:
        return out
    d = depth.copy().astype(np.float32)
    v = valid.astype(bool)
    low = np.percentile(d[v], 2)
    high = np.percentile(d[v], 98)
    if not np.isfinite(low):
        low = float(d[v].min())
    if not np.isfinite(high) or high <= low:
        high = max(low + 1e-6, float(d[v].max()))
    norm = np.clip((d - low) / (high - low), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    # Lightweight pseudo-color without external deps:
    # R increases with depth, B decreases with depth, G peaks mid-range.
    r = norm_u8
    b = 255 - norm_u8
    g = (255 - np.abs(norm_u8.astype(np.int16) * 2 - 255)).astype(np.uint8)
    cm = np.stack([r, g, b], axis=-1)
    out[v] = cm[v]
    return out


def _run_depth_eval(
    image_name: str,
    vae_model,
    depth_head,
    latent_patch_size: int,
    device: torch.device,
    si_lambda: float,
    auto_bgr_to_rgb: bool,
    save_dir: Optional[str],
) -> Dict[str, float]:
    _, _, rgb, depth = _get_by_image_name(image_name)
    if depth is None:
        raise ValueError(f"No depth returned for image_name={image_name}")

    vae_downsample = 8
    factor = int(latent_patch_size) * int(vae_downsample)
    x = _prepare_rgb_tensor(np.asarray(rgb), factor=factor, auto_bgr_to_rgb=auto_bgr_to_rgb, image_name=image_name)

    depth_np = np.asarray(depth, dtype=np.float32)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np = _crop_to_factor(depth_np, factor=factor)

    vae_dtype = next(vae_model.parameters()).dtype
    depth_head_dtype = next(depth_head.parameters()).dtype

    with torch.no_grad():
        latent = vae_model.encode(x.unsqueeze(0).to(device=device, dtype=vae_dtype))[0]
        tokens, h_tok, w_tok = _patchify_latent(latent, latent_patch_size)
        pred_depth_tokens = depth_head(tokens.to(device=device, dtype=depth_head_dtype))
        pred_depth_hw = F.softplus(pred_depth_tokens.float().reshape(h_tok, w_tok)) + 1e-6
        pred_depth = F.interpolate(
            pred_depth_hw.unsqueeze(0).unsqueeze(0),
            size=depth_np.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    gt_depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    valid = torch.isfinite(gt_depth) & (gt_depth > 0)
    metrics = _depth_metrics(pred=pred_depth, gt=gt_depth, valid=valid, si_lambda=si_lambda)
    metrics.update(
        {
            "image_name": image_name,
            "pred_token_h": float(h_tok),
            "pred_token_w": float(w_tok),
            "gt_h": float(depth_np.shape[0]),
            "gt_w": float(depth_np.shape[1]),
        }
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        pred_np = pred_depth.detach().cpu().numpy()
        gt_np = gt_depth.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()

        pred_png = _colorize_depth(pred_np, valid_np)
        gt_png = _colorize_depth(gt_np, valid_np)
        Image.fromarray(pred_png).save(os.path.join(save_dir, "depth_pred_vis.png"))
        Image.fromarray(gt_png).save(os.path.join(save_dir, "depth_gt_vis.png"))

        # Save raw arrays for later quantitative checks.
        np.save(os.path.join(save_dir, "depth_pred.npy"), pred_np)
        np.save(os.path.join(save_dir, "depth_gt.npy"), gt_np)
        np.save(os.path.join(save_dir, "depth_valid.npy"), valid_np.astype(np.uint8))

    return metrics


def _run_pose_eval(
    source_image_name: str,
    target_image_name: str,
    vae_model,
    pose_head,
    latent_patch_size: int,
    device: torch.device,
    auto_bgr_to_rgb: bool,
) -> Dict[str, float]:
    _, src_pose_obj, src_rgb, _ = _get_by_image_name(source_image_name)
    _, tgt_pose_obj, tgt_rgb, _ = _get_by_image_name(target_image_name)

    src_pose = parse_pose_matrix(src_pose_obj)
    tgt_pose = parse_pose_matrix(tgt_pose_obj)
    if src_pose is None or tgt_pose is None:
        raise ValueError("Cannot parse source/target pose matrices.")

    gt_delta_np = relative_pose_delta_se3(src_pose, tgt_pose)
    if gt_delta_np is None:
        raise ValueError("Cannot compute relative pose delta from source/target poses.")

    vae_downsample = 8
    factor = int(latent_patch_size) * int(vae_downsample)
    src_x = _prepare_rgb_tensor(np.asarray(src_rgb), factor=factor, auto_bgr_to_rgb=auto_bgr_to_rgb, image_name=source_image_name)
    tgt_x = _prepare_rgb_tensor(np.asarray(tgt_rgb), factor=factor, auto_bgr_to_rgb=auto_bgr_to_rgb, image_name=target_image_name)

    vae_dtype = next(vae_model.parameters()).dtype
    pose_head_dtype = next(pose_head.parameters()).dtype

    with torch.no_grad():
        src_latent = vae_model.encode(src_x.unsqueeze(0).to(device=device, dtype=vae_dtype))
        tgt_latent = vae_model.encode(tgt_x.unsqueeze(0).to(device=device, dtype=vae_dtype))
        src_pool = src_latent.mean(dim=(2, 3))
        tgt_pool = tgt_latent.mean(dim=(2, 3))
        pose_feat = torch.cat([src_pool, tgt_pool, tgt_pool - src_pool], dim=-1)
        pred_delta = pose_head(pose_feat.to(dtype=pose_head_dtype)).float().squeeze(0)

    gt_delta = torch.from_numpy(gt_delta_np).to(device=device, dtype=torch.float32)
    diff = pred_delta - gt_delta

    rot_geo = _rotation_geodesic_deg(pred_delta[None, 3:], gt_delta[None, 3:])[0]

    return {
        "source_image_name": source_image_name,
        "target_image_name": target_image_name,
        "pred_tx": float(pred_delta[0].item()),
        "pred_ty": float(pred_delta[1].item()),
        "pred_tz": float(pred_delta[2].item()),
        "pred_rx": float(pred_delta[3].item()),
        "pred_ry": float(pred_delta[4].item()),
        "pred_rz": float(pred_delta[5].item()),
        "gt_tx": float(gt_delta[0].item()),
        "gt_ty": float(gt_delta[1].item()),
        "gt_tz": float(gt_delta[2].item()),
        "gt_rx": float(gt_delta[3].item()),
        "gt_ry": float(gt_delta[4].item()),
        "gt_rz": float(gt_delta[5].item()),
        "trans_l1": float(diff[:3].abs().mean().item()),
        "trans_l2": float(torch.linalg.norm(diff[:3]).item()),
        "rot_vec_l1": float(diff[3:].abs().mean().item()),
        "rot_geodesic_deg": float(rot_geo.item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Single-sample sanity test for geo pose/depth heads.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to BAGEL model root (contains ae.safetensors).")
    parser.add_argument("--geo_aux_ckpt", type=str, required=True, help="Path to geo_aux.pt or its parent dir.")
    parser.add_argument("--latent_patch_size", type=int, default=2, help="latent_patch_size used during training.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    parser.add_argument("--depth_image_name", type=str, default=None, help="Run depth-head test on this image_name.")
    parser.add_argument("--pose_source_image_name", type=str, default=None, help="Source image_name for pose-head test.")
    parser.add_argument("--pose_target_image_name", type=str, default=None, help="Target image_name for pose-head test.")

    parser.add_argument("--si_lambda", type=float, default=0.5, help="Lambda for SILog metric.")
    parser.add_argument("--auto_bgr_to_rgb", type=_str2bool, default=True, help="Convert matterport3d BGR to RGB.")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional output dir for depth visualizations.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.depth_image_name is None and (args.pose_source_image_name is None or args.pose_target_image_name is None):
        raise ValueError("Provide --depth_image_name and/or both --pose_source_image_name --pose_target_image_name.")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    geo_aux_ckpt = _resolve_geo_aux_ckpt(args.geo_aux_ckpt)
    ae_path = os.path.join(os.path.abspath(args.model_path), "ae.safetensors")
    if not os.path.exists(ae_path):
        raise FileNotFoundError(f"Cannot find ae.safetensors at: {ae_path}")

    load_ae = _load_ae_fn()
    vae_model, _ = load_ae(local_path=ae_path)
    vae_model = vae_model.to(device=device)
    vae_model.eval()

    pose_head, depth_head = _load_geo_heads(geo_aux_ckpt=geo_aux_ckpt, device=device)

    out = {
        "device": str(device),
        "geo_aux_ckpt": geo_aux_ckpt,
        "model_path": os.path.abspath(args.model_path),
    }

    if args.depth_image_name is not None:
        out["depth_eval"] = _run_depth_eval(
            image_name=args.depth_image_name,
            vae_model=vae_model,
            depth_head=depth_head,
            latent_patch_size=args.latent_patch_size,
            device=device,
            si_lambda=args.si_lambda,
            auto_bgr_to_rgb=args.auto_bgr_to_rgb,
            save_dir=args.save_dir,
        )

    if args.pose_source_image_name is not None and args.pose_target_image_name is not None:
        out["pose_eval"] = _run_pose_eval(
            source_image_name=args.pose_source_image_name,
            target_image_name=args.pose_target_image_name,
            vae_model=vae_model,
            pose_head=pose_head,
            latent_patch_size=args.latent_patch_size,
            device=device,
            auto_bgr_to_rgb=args.auto_bgr_to_rgb,
        )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
