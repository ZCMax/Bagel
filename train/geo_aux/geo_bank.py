import json
import os
from typing import Any, Dict, List, Optional

import torch


def _safe_int_index(index_obj: Any) -> Optional[int]:
    if isinstance(index_obj, int):
        return index_obj
    if isinstance(index_obj, str):
        try:
            return int(index_obj)
        except ValueError:
            return None
    return None


class GeoAuxBank:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.root_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_specs = manifest.get("datasets", {})
        if len(dataset_specs) == 0:
            raise ValueError(f"No dataset entries found in geo aux manifest: {manifest_path}")

        self.source_depth: Dict[str, torch.Tensor] = {}
        self.target_depth: Dict[str, torch.Tensor] = {}
        self.source_pose: Dict[str, torch.Tensor] = {}
        self.target_pose: Dict[str, torch.Tensor] = {}
        self.pose_delta: Dict[str, torch.Tensor] = {}
        self.source_k: Dict[str, torch.Tensor] = {}
        self.target_k: Dict[str, torch.Tensor] = {}
        self.num_context: Dict[str, torch.Tensor] = {}
        self.source_idx: Dict[str, torch.Tensor] = {}
        self.valid_mask: Dict[str, torch.Tensor] = {}
        self.sample_id_to_row: Dict[str, Dict[str, int]] = {}
        self.default_depth_hw = None

        for dataset_name, spec in dataset_specs.items():
            def _load_tensor(path_key: str) -> torch.Tensor:
                rel_path = spec[path_key]
                abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.root_dir, rel_path)
                t = torch.load(abs_path, map_location="cpu")
                if not torch.is_tensor(t):
                    raise TypeError(f"{path_key} at {abs_path} is not a tensor")
                return t

            source_depth = _load_tensor("source_depth_path").float().contiguous()
            target_depth = _load_tensor("target_depth_path").float().contiguous()
            source_pose = _load_tensor("source_pose_path").float().contiguous()
            target_pose = _load_tensor("target_pose_path").float().contiguous()
            pose_delta = _load_tensor("pose_delta_path").float().contiguous()
            source_k = _load_tensor("source_k_path").float().contiguous()
            target_k = _load_tensor("target_k_path").float().contiguous()
            num_context = _load_tensor("num_context_path").long().flatten()
            source_idx = _load_tensor("source_idx_path").long().flatten()
            valid_mask = _load_tensor("valid_mask_path").to(torch.bool).flatten()

            n = source_depth.shape[0]
            checks = {
                "target_depth": target_depth.shape[0],
                "source_pose": source_pose.shape[0],
                "target_pose": target_pose.shape[0],
                "pose_delta": pose_delta.shape[0],
                "source_k": source_k.shape[0],
                "target_k": target_k.shape[0],
                "num_context": num_context.shape[0],
                "source_idx": source_idx.shape[0],
                "valid_mask": valid_mask.shape[0],
            }
            for name, m in checks.items():
                if m != n:
                    raise ValueError(
                        f"Geo bank length mismatch for dataset={dataset_name}: "
                        f"source_depth={n}, {name}={m}"
                    )

            if source_depth.dim() != 3 or target_depth.dim() != 3:
                raise ValueError(f"Depth tensors for {dataset_name} must be [N,H,W]")
            if source_pose.dim() != 3 or source_pose.shape[-2:] != (4, 4):
                raise ValueError(f"source_pose for {dataset_name} must be [N,4,4]")
            if target_pose.dim() != 3 or target_pose.shape[-2:] != (4, 4):
                raise ValueError(f"target_pose for {dataset_name} must be [N,4,4]")
            if source_k.dim() != 3 or source_k.shape[-2:] != (3, 3):
                raise ValueError(f"source_k for {dataset_name} must be [N,3,3]")
            if target_k.dim() != 3 or target_k.shape[-2:] != (3, 3):
                raise ValueError(f"target_k for {dataset_name} must be [N,3,3]")
            if pose_delta.dim() != 2 or pose_delta.shape[-1] != 6:
                raise ValueError(f"pose_delta for {dataset_name} must be [N,6]")

            self.source_depth[dataset_name] = source_depth
            self.target_depth[dataset_name] = target_depth
            self.source_pose[dataset_name] = source_pose
            self.target_pose[dataset_name] = target_pose
            self.pose_delta[dataset_name] = pose_delta
            self.source_k[dataset_name] = source_k
            self.target_k[dataset_name] = target_k
            self.num_context[dataset_name] = num_context
            self.source_idx[dataset_name] = source_idx
            self.valid_mask[dataset_name] = valid_mask

            sample_id_path = spec.get("sample_id_path")
            if sample_id_path is None:
                raise ValueError(
                    f"Geo manifest for dataset={dataset_name} misses sample_id_path. "
                    "Please regenerate geo bank with the updated precompute_geo_bank.py."
                )
            abs_sample_id_path = (
                sample_id_path
                if os.path.isabs(sample_id_path)
                else os.path.join(self.root_dir, sample_id_path)
            )
            if not os.path.exists(abs_sample_id_path):
                raise FileNotFoundError(
                    f"sample_id_path not found for dataset={dataset_name}: {abs_sample_id_path}"
                )
            with open(abs_sample_id_path, "r", encoding="utf-8") as f:
                sample_ids = json.load(f)
            if not isinstance(sample_ids, list):
                raise TypeError(
                    f"sample_id_path for dataset={dataset_name} is not a json list: {abs_sample_id_path}"
                )
            if len(sample_ids) != n:
                raise ValueError(
                    f"sample_id length mismatch for dataset={dataset_name}: "
                    f"len(sample_ids)={len(sample_ids)} vs num_rows={n}"
                )
            id_map: Dict[str, int] = {}
            for row_id, sample_id in enumerate(sample_ids):
                sid = str(sample_id)
                if sid in id_map:
                    raise ValueError(
                        f"Duplicate sample_id in dataset={dataset_name}: sample_id={sid}, "
                        f"first_row={id_map[sid]}, duplicate_row={row_id}"
                    )
                id_map[sid] = row_id
            self.sample_id_to_row[dataset_name] = id_map

            if self.default_depth_hw is None:
                self.default_depth_hw = (int(source_depth.shape[1]), int(source_depth.shape[2]))

        if self.default_depth_hw is None:
            self.default_depth_hw = (64, 64)

    def lookup_batch(self, batch_data_indexes: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
        src_depth_list = []
        tgt_depth_list = []
        src_pose_list = []
        tgt_pose_list = []
        pose_delta_list = []
        src_k_list = []
        tgt_k_list = []
        num_context_list = []
        source_idx_list = []
        valid_list = []

        default_h, default_w = self.default_depth_hw
        for item in batch_data_indexes:
            dataset_name = item.get("dataset_name")
            row_idx = None
            sample_id = item.get("sample_id")
            if dataset_name in self.sample_id_to_row and sample_id is not None:
                row_idx = self.sample_id_to_row[dataset_name].get(str(sample_id))
            has = dataset_name in self.source_depth and row_idx is not None

            if not has:
                src_depth_list.append(torch.zeros((default_h, default_w), dtype=torch.float32))
                tgt_depth_list.append(torch.zeros((default_h, default_w), dtype=torch.float32))
                src_pose_list.append(torch.eye(4, dtype=torch.float32))
                tgt_pose_list.append(torch.eye(4, dtype=torch.float32))
                pose_delta_list.append(torch.zeros(6, dtype=torch.float32))
                src_k_list.append(torch.eye(3, dtype=torch.float32))
                tgt_k_list.append(torch.eye(3, dtype=torch.float32))
                num_context_list.append(torch.tensor(0, dtype=torch.long))
                source_idx_list.append(torch.tensor(0, dtype=torch.long))
                valid_list.append(False)
                continue

            if row_idx < 0 or row_idx >= self.source_depth[dataset_name].shape[0]:
                src_depth_list.append(torch.zeros((default_h, default_w), dtype=torch.float32))
                tgt_depth_list.append(torch.zeros((default_h, default_w), dtype=torch.float32))
                src_pose_list.append(torch.eye(4, dtype=torch.float32))
                tgt_pose_list.append(torch.eye(4, dtype=torch.float32))
                pose_delta_list.append(torch.zeros(6, dtype=torch.float32))
                src_k_list.append(torch.eye(3, dtype=torch.float32))
                tgt_k_list.append(torch.eye(3, dtype=torch.float32))
                num_context_list.append(torch.tensor(0, dtype=torch.long))
                source_idx_list.append(torch.tensor(0, dtype=torch.long))
                valid_list.append(False)
                continue

            src_depth = self.source_depth[dataset_name][row_idx]
            tgt_depth = self.target_depth[dataset_name][row_idx]
            src_pose = self.source_pose[dataset_name][row_idx]
            tgt_pose = self.target_pose[dataset_name][row_idx]
            delta = self.pose_delta[dataset_name][row_idx]
            src_k = self.source_k[dataset_name][row_idx]
            tgt_k = self.target_k[dataset_name][row_idx]
            nctx = self.num_context[dataset_name][row_idx]
            sidx = self.source_idx[dataset_name][row_idx]
            v = bool(self.valid_mask[dataset_name][row_idx])

            src_depth_list.append(src_depth)
            tgt_depth_list.append(tgt_depth)
            src_pose_list.append(src_pose)
            tgt_pose_list.append(tgt_pose)
            pose_delta_list.append(delta)
            src_k_list.append(src_k)
            tgt_k_list.append(tgt_k)
            num_context_list.append(nctx)
            source_idx_list.append(sidx)
            valid_list.append(v)

        # Assume all depth maps in one bank share same shape.
        source_depth = torch.stack(src_depth_list, dim=0).to(device=device, dtype=torch.float32)
        target_depth = torch.stack(tgt_depth_list, dim=0).to(device=device, dtype=torch.float32)
        source_pose = torch.stack(src_pose_list, dim=0).to(device=device, dtype=torch.float32)
        target_pose = torch.stack(tgt_pose_list, dim=0).to(device=device, dtype=torch.float32)
        pose_delta = torch.stack(pose_delta_list, dim=0).to(device=device, dtype=torch.float32)
        source_k = torch.stack(src_k_list, dim=0).to(device=device, dtype=torch.float32)
        target_k = torch.stack(tgt_k_list, dim=0).to(device=device, dtype=torch.float32)
        num_context = torch.stack(num_context_list, dim=0).to(device=device, dtype=torch.long)
        source_idx = torch.stack(source_idx_list, dim=0).to(device=device, dtype=torch.long)
        valid_mask = torch.tensor(valid_list, device=device, dtype=torch.bool)

        return {
            "source_depth": source_depth,
            "target_depth": target_depth,
            "source_pose": source_pose,
            "target_pose": target_pose,
            "pose_delta": pose_delta,
            "source_k": source_k,
            "target_k": target_k,
            "num_context": num_context,
            "source_idx": source_idx,
            "valid_mask": valid_mask,
        }
