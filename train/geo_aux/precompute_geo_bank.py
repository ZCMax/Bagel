import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_info import DATASET_INFO  # noqa: E402
from train.pose_aux.pose_utils import parse_pose_matrix, pick_source_idx, relative_pose_delta_se3  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute geometry auxiliary bank (get_more based).")
    parser.add_argument("--dataset_config_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--depth_size", type=int, default=128, help="Resize depth to depth_size x depth_size.")
    parser.add_argument("--depth_clip_max", type=float, default=1e6, help="Clip very large depth values.")
    parser.add_argument(
        "--allow_missing_get_more",
        action="store_true",
        help="Do not fail when train.geo_aux.get_more cannot be imported (debug only; valid_rows may be 0).",
    )
    return parser.parse_args()


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _parse_intrinsics_matrix(x: Any) -> Optional[np.ndarray]:
    arr = _as_float_array(x)
    if arr is None:
        return None
    if arr.shape == (3, 3):
        return arr.astype(np.float64)
    if arr.shape == (4, 4):
        return arr[:3, :3].astype(np.float64)
    if arr.ndim == 1 and arr.size == 9:
        return arr.reshape(3, 3).astype(np.float64)
    return None


def _to_depth_map(x: Any) -> Optional[np.ndarray]:
    arr = _as_float_array(x)
    if arr is None or arr.ndim < 2:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _resize_depth(depth: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="nearest")
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _scale_intrinsics(k: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int) -> np.ndarray:
    out = k.copy().astype(np.float64)
    sx = float(dst_w) / float(max(src_w, 1))
    sy = float(dst_h) / float(max(src_h, 1))
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def _default_intrinsics_from_hw(h: int, w: int) -> np.ndarray:
    f = float(max(h, w))
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _dataset_from_image_name(image_name: str) -> Optional[str]:
    lower = image_name.lower()
    if lower.startswith("scannet/"):
        return "scannet"
    if lower.startswith("dl3dv/"):
        return "dl3dv"
    if lower.startswith("matterport3d/"):
        return "matterport3d"
    return None


def _canonical_image_name(image_name: Any) -> Optional[str]:
    if not isinstance(image_name, str) or len(image_name) == 0:
        return None
    norm = image_name.replace("\\", "/")
    if not os.path.isabs(norm):
        return norm
    lower = norm.lower()
    for key in ("scannet/", "dl3dv/", "matterport3d/"):
        pos = lower.find(key)
        if pos >= 0:
            return norm[pos:]
    return None


def _extract_sample_id(row: Dict[str, Any], row_idx: int, dataset_name: str) -> str:
    # Prefer explicit row id for stable alignment between training dataloader and geo bank.
    if isinstance(row, dict):
        for key in ("id", "sample_id", "uid"):
            if key in row and row[key] is not None:
                return str(row[key])
    # Fallback keeps uniqueness within a dataset even if id is absent.
    return f"{dataset_name}:{row_idx}"


def _load_get_more_module(require_get_more: bool):
    try:
        from train.geo_aux import get_more as gm  # noqa: WPS433

        return gm
    except Exception as e:
        if require_get_more:
            raise RuntimeError(
                "Failed to import train.geo_aux.get_more. "
                "Please ensure its dependencies are installed (e.g. cv2) and paths are valid."
            ) from e
        return None


def _get_more_fetch(gm, image_name: str) -> Optional[Dict[str, Any]]:
    if gm is None:
        return None
    dataset = _dataset_from_image_name(image_name)
    if dataset is None:
        return None

    fn_name = {
        "scannet": "get_scannet",
        "dl3dv": "get_dl3dv",
        "matterport3d": "get_matterport3d",
    }[dataset]
    if not hasattr(gm, fn_name):
        return None

    fn = getattr(gm, fn_name)
    try:
        k, pose, rgb, depth = fn(image_name)
    except Exception:
        return None

    k = _parse_intrinsics_matrix(k)
    pose = parse_pose_matrix(pose)
    depth = _to_depth_map(depth)
    if k is None or pose is None or depth is None:
        return None

    rgb_hw = None
    if isinstance(rgb, np.ndarray) and rgb.ndim >= 2:
        rgb_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    return {"k": k, "pose": pose, "depth": depth, "rgb_hw": rgb_hw}


def _pick_source_target(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int, int]:
    context = row.get("context", [])
    if isinstance(context, list):
        num_context = len(context)
    else:
        num_context = 0

    if num_context > 0:
        source_idx = pick_source_idx(row)
        if source_idx is None:
            source_idx = num_context - 1
        source_idx = int(max(0, min(int(source_idx), num_context - 1)))
        source_name = _canonical_image_name(context[source_idx])
    else:
        source_idx = 0
        source_name = _canonical_image_name(
            row.get("source")
            or row.get("source_image")
            or row.get("image_name")
            or row.get("src")
        )

    target_name = _canonical_image_name(
        row.get("target")
        or row.get("target_image")
        or row.get("gt_image")
        or row.get("image_name_target")
    )
    return source_name, target_name, num_context, source_idx


def iter_group_rows(group_name: str, group_cfg: Dict):
    dataset_type = group_cfg.get("dataset_type", group_name)
    if dataset_type not in DATASET_INFO:
        raise KeyError(f"dataset_type '{dataset_type}' not found in DATASET_INFO")

    dataset_names = group_cfg["dataset_names"]
    num_used_data = group_cfg.get("num_used_data", [None] * len(dataset_names))
    if len(num_used_data) != len(dataset_names):
        raise ValueError(
            f"group '{group_name}' has mismatched num_used_data({len(num_used_data)}) and dataset_names({len(dataset_names)})"
        )

    shuffle_lines = bool(group_cfg.get("shuffle_lines", False))
    shuffle_seed = int(group_cfg.get("shuffle_seed", 0))

    row_idx = 0
    for dataset_name, limit in zip(dataset_names, num_used_data):
        meta = DATASET_INFO[dataset_type][dataset_name]
        if "jsonl_path" not in meta:
            raise ValueError(
                f"group '{group_name}' / dataset '{dataset_name}' does not provide jsonl_path; "
                "only jsonl datasets are supported by this precompute script."
            )
        jsonl_path = meta["jsonl_path"]
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
        if shuffle_lines:
            rng = random.Random(shuffle_seed)
            rng.shuffle(raw_lines)
        if limit is not None:
            raw_lines = raw_lines[: int(limit)]

        for line in raw_lines:
            row = {}
            try:
                row = json.loads(line)
            except Exception:
                pass
            yield row_idx, row, dataset_name
            row_idx += 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    depth_size = int(args.depth_size)
    depth_clip_max = float(args.depth_clip_max)
    gm = _load_get_more_module(require_get_more=not bool(args.allow_missing_get_more))
    if gm is None:
        print(
            "[WARN] train.geo_aux.get_more is unavailable. "
            "No geometry metadata will be fetched. This mode is for debugging only."
        )

    with open(args.dataset_config_file, "r", encoding="utf-8") as f:
        dataset_meta = yaml.safe_load(f)

    manifest = {"datasets": {}, "depth_size": depth_size, "source": "get_more"}

    for group_name, group_cfg in dataset_meta.items():
        rows = list(iter_group_rows(group_name, group_cfg))
        if len(rows) == 0:
            continue

        n = len(rows)
        src_depth_t = torch.zeros((n, depth_size, depth_size), dtype=torch.float32)
        tgt_depth_t = torch.zeros((n, depth_size, depth_size), dtype=torch.float32)
        src_pose_t = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        tgt_pose_t = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        pose_delta_t = torch.zeros((n, 6), dtype=torch.float32)
        src_k_t = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        tgt_k_t = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        num_context_t = torch.zeros((n,), dtype=torch.long)
        source_idx_t = torch.zeros((n,), dtype=torch.long)
        valid_t = torch.zeros((n,), dtype=torch.bool)
        sample_id_list: List[str] = ["" for _ in range(n)]

        for row_idx, row, _dataset_name in tqdm(rows, desc=f"[GeoBank] {group_name}"):
            sample_id_list[row_idx] = _extract_sample_id(row=row, row_idx=row_idx, dataset_name=group_name)
            src_name, tgt_name, num_context, source_idx = _pick_source_target(row)
            num_context_t[row_idx] = int(max(num_context, 0))
            source_idx_t[row_idx] = int(max(source_idx, 0))

            if src_name is None or tgt_name is None:
                print('step1')
                continue

            src_item = _get_more_fetch(gm, src_name)
            tgt_item = _get_more_fetch(gm, tgt_name)
            if src_item is None or tgt_item is None:
                print('step2')
                continue

            src_depth = np.nan_to_num(src_item["depth"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_depth = np.nan_to_num(tgt_item["depth"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            src_depth = np.clip(src_depth, 0.0, depth_clip_max)
            tgt_depth = np.clip(tgt_depth, 0.0, depth_clip_max)

            src_h0, src_w0 = src_depth.shape[:2]
            tgt_h0, tgt_w0 = tgt_depth.shape[:2]
            src_depth_rs = _resize_depth(src_depth, (depth_size, depth_size))
            tgt_depth_rs = _resize_depth(tgt_depth, (depth_size, depth_size))

            src_k = src_item["k"]
            tgt_k = tgt_item["k"]
            if src_k is None:
                src_k = _default_intrinsics_from_hw(src_h0, src_w0)
            if tgt_k is None:
                tgt_k = _default_intrinsics_from_hw(tgt_h0, tgt_w0)
            src_k = _scale_intrinsics(src_k, src_h0, src_w0, depth_size, depth_size)
            tgt_k = _scale_intrinsics(tgt_k, tgt_h0, tgt_w0, depth_size, depth_size)

            src_pose = src_item["pose"]
            tgt_pose = tgt_item["pose"]
            delta = relative_pose_delta_se3(src_pose, tgt_pose)
            if delta is None:
                print('step3')
                continue

            src_depth_t[row_idx] = torch.from_numpy(src_depth_rs)
            tgt_depth_t[row_idx] = torch.from_numpy(tgt_depth_rs)
            src_pose_t[row_idx] = torch.from_numpy(src_pose.astype(np.float32))
            tgt_pose_t[row_idx] = torch.from_numpy(tgt_pose.astype(np.float32))
            pose_delta_t[row_idx] = torch.from_numpy(delta.astype(np.float32))
            src_k_t[row_idx] = torch.from_numpy(src_k.astype(np.float32))
            tgt_k_t[row_idx] = torch.from_numpy(tgt_k.astype(np.float32))
            valid_t[row_idx] = True

        names = {
            "source_depth_path": f"{group_name}.source_depth.pt",
            "target_depth_path": f"{group_name}.target_depth.pt",
            "source_pose_path": f"{group_name}.source_pose.pt",
            "target_pose_path": f"{group_name}.target_pose.pt",
            "pose_delta_path": f"{group_name}.pose_delta.pt",
            "source_k_path": f"{group_name}.source_k.pt",
            "target_k_path": f"{group_name}.target_k.pt",
            "num_context_path": f"{group_name}.num_context.pt",
            "source_idx_path": f"{group_name}.source_idx.pt",
            "valid_mask_path": f"{group_name}.valid.pt",
            "sample_id_path": f"{group_name}.sample_id.json",
        }
        sample_id_set = {str(x) for x in sample_id_list}
        if len(sample_id_set) != len(sample_id_list):
            raise ValueError(
                f"Duplicate sample_id detected in group={group_name}: "
                f"num_rows={len(sample_id_list)}, unique_sample_id={len(sample_id_set)}"
            )
        torch.save(src_depth_t, os.path.join(args.output_dir, names["source_depth_path"]))
        torch.save(tgt_depth_t, os.path.join(args.output_dir, names["target_depth_path"]))
        torch.save(src_pose_t, os.path.join(args.output_dir, names["source_pose_path"]))
        torch.save(tgt_pose_t, os.path.join(args.output_dir, names["target_pose_path"]))
        torch.save(pose_delta_t, os.path.join(args.output_dir, names["pose_delta_path"]))
        torch.save(src_k_t, os.path.join(args.output_dir, names["source_k_path"]))
        torch.save(tgt_k_t, os.path.join(args.output_dir, names["target_k_path"]))
        torch.save(num_context_t, os.path.join(args.output_dir, names["num_context_path"]))
        torch.save(source_idx_t, os.path.join(args.output_dir, names["source_idx_path"]))
        torch.save(valid_t, os.path.join(args.output_dir, names["valid_mask_path"]))
        with open(os.path.join(args.output_dir, names["sample_id_path"]), "w", encoding="utf-8") as f:
            json.dump(sample_id_list, f, ensure_ascii=False)

        manifest["datasets"][group_name] = {
            **names,
            "num_rows": int(n),
            "valid_rows": int(valid_t.sum().item()),
            "depth_size": depth_size,
        }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Geo aux manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
