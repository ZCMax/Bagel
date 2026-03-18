import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_info import DATASET_INFO
from train.pose_aux.pose_utils import (
    parse_pose_matrix,
    parse_pose_list,
    pick_source_idx,
    relative_pose_delta_se3,
    so3_log_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute pose-delta bank for pose auxiliary loss.")
    parser.add_argument("--dataset_config_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def _se3_from_delta_field(delta_obj) -> Optional[torch.Tensor]:
    if delta_obj is None:
        return None
    if isinstance(delta_obj, dict) and "se3" in delta_obj:
        delta_obj = delta_obj["se3"]

    try:
        arr = torch.as_tensor(delta_obj, dtype=torch.float32).flatten()
    except Exception:
        arr = None
    if arr is not None and arr.numel() == 6 and torch.isfinite(arr).all():
        return arr

    mat = parse_pose_matrix(delta_obj)
    if mat is None:
        return None
    t = torch.tensor(mat[:3, 3], dtype=torch.float32)
    r = torch.tensor(so3_log_map(mat[:3, :3]), dtype=torch.float32)
    out = torch.cat([t, r], dim=0)
    if not torch.isfinite(out).all():
        return None
    return out


def extract_pose_delta_se3(row: Dict) -> Optional[torch.Tensor]:
    for key in ("delta_pose_se3", "pose_delta_se3", "target_minus_source_se3"):
        if key in row:
            out = _se3_from_delta_field(row[key])
            if out is not None:
                return out

    for key in ("delta_pose", "pose_delta", "relative_pose"):
        if key in row:
            out = _se3_from_delta_field(row[key])
            if out is not None:
                return out

    source_pose = None
    target_pose = None
    for key in ("source_pose", "source_c2w", "src_pose", "src_c2w"):
        if key in row:
            source_pose = parse_pose_matrix(row[key])
            if source_pose is not None:
                break
    for key in ("target_pose", "target_c2w", "gt_pose", "gt_c2w"):
        if key in row:
            target_pose = parse_pose_matrix(row[key])
            if target_pose is not None:
                break

    if source_pose is not None and target_pose is not None:
        delta = relative_pose_delta_se3(source_pose, target_pose)
        return torch.tensor(delta, dtype=torch.float32) if delta is not None else None

    context_pose_fields = (
        "context_poses",
        "context_pose_list",
        "context_pose",
        "context_c2w",
        "input_poses",
    )
    context_poses = None
    for key in context_pose_fields:
        if key in row:
            context_poses = parse_pose_list(row[key])
            if context_poses is not None:
                break

    if context_poses is None:
        return None

    source_idx = pick_source_idx(row)
    if source_idx is None or source_idx < 0 or source_idx >= len(context_poses):
        return None

    target_pose = None
    for key in ("target_pose", "target_c2w", "gt_pose", "gt_c2w"):
        if key in row:
            target_pose = parse_pose_matrix(row[key])
            if target_pose is not None:
                break
    if target_pose is None:
        return None

    delta = relative_pose_delta_se3(context_poses[source_idx], target_pose)
    return torch.tensor(delta, dtype=torch.float32) if delta is not None else None


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
            yield row_idx, row
            row_idx += 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset_config_file, "r", encoding="utf-8") as f:
        dataset_meta = yaml.safe_load(f)

    manifest = {"datasets": {}}
    for group_name, group_cfg in dataset_meta.items():
        rows = list(iter_group_rows(group_name, group_cfg))
        if len(rows) == 0:
            continue

        pose_delta_tensor = torch.zeros((len(rows), 6), dtype=torch.float32)
        valid_mask = torch.zeros((len(rows),), dtype=torch.bool)

        for row_idx, row in tqdm(rows, desc=f"[PoseBank] {group_name}"):
            se3 = extract_pose_delta_se3(row)
            if se3 is None or se3.numel() != 6 or (not torch.isfinite(se3).all()):
                continue
            pose_delta_tensor[row_idx] = se3
            valid_mask[row_idx] = True

        delta_name = f"{group_name}.pose_delta.pt"
        valid_name = f"{group_name}.valid.pt"
        delta_path = os.path.join(args.output_dir, delta_name)
        valid_path = os.path.join(args.output_dir, valid_name)
        torch.save(pose_delta_tensor, delta_path)
        torch.save(valid_mask, valid_path)

        manifest["datasets"][group_name] = {
            "pose_delta_path": delta_name,
            "valid_mask_path": valid_name,
            "num_rows": int(pose_delta_tensor.shape[0]),
            "valid_rows": int(valid_mask.sum().item()),
        }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Pose manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
