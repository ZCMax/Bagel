import json
import os
from typing import Any, Dict, List, Optional, Tuple

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


class PoseDeltaBank:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.root_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_specs = manifest.get("datasets", {})
        if len(dataset_specs) == 0:
            raise ValueError(f"No dataset entries found in pose manifest: {manifest_path}")

        self.pose_deltas: Dict[str, torch.Tensor] = {}
        self.valid_masks: Dict[str, torch.Tensor] = {}

        for dataset_name, spec in dataset_specs.items():
            delta_rel = spec["pose_delta_path"]
            delta_path = delta_rel if os.path.isabs(delta_rel) else os.path.join(self.root_dir, delta_rel)

            deltas = torch.load(delta_path, map_location="cpu")
            if isinstance(deltas, dict) and "pose_deltas" in deltas:
                deltas = deltas["pose_deltas"]
            if not torch.is_tensor(deltas):
                raise TypeError(f"Pose deltas at {delta_path} must be a tensor")
            deltas = deltas.float().contiguous()
            if deltas.dim() != 2 or deltas.shape[1] != 6:
                raise ValueError(
                    f"Pose deltas at {delta_path} must have shape [N, 6], got {tuple(deltas.shape)}"
                )

            valid_rel = spec.get("valid_mask_path")
            if valid_rel is not None:
                valid_path = valid_rel if os.path.isabs(valid_rel) else os.path.join(self.root_dir, valid_rel)
                valid_mask = torch.load(valid_path, map_location="cpu")
                if not torch.is_tensor(valid_mask):
                    raise TypeError(f"Valid mask at {valid_path} must be a tensor")
                valid_mask = valid_mask.to(torch.bool).flatten()
                if valid_mask.numel() != deltas.shape[0]:
                    raise ValueError(
                        f"valid_mask length mismatch for dataset '{dataset_name}': "
                        f"{valid_mask.numel()} vs deltas {deltas.shape[0]}"
                    )
            else:
                valid_mask = torch.ones(deltas.shape[0], dtype=torch.bool)

            self.pose_deltas[dataset_name] = deltas
            self.valid_masks[dataset_name] = valid_mask

    def lookup_batch(
        self,
        batch_data_indexes: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_pose_delta = []
        valid = []

        for item in batch_data_indexes:
            dataset_name = item.get("dataset_name")
            row_idx = _safe_int_index(item.get("data_indexes"))

            if dataset_name not in self.pose_deltas or row_idx is None:
                gt_pose_delta.append(torch.zeros(6, dtype=torch.float32))
                valid.append(False)
                continue

            deltas = self.pose_deltas[dataset_name]
            mask = self.valid_masks[dataset_name]
            if row_idx < 0 or row_idx >= deltas.shape[0]:
                gt_pose_delta.append(torch.zeros(6, dtype=torch.float32))
                valid.append(False)
                continue

            gt_pose_delta.append(deltas[row_idx])
            valid.append(bool(mask[row_idx]))

        gt_pose_delta = torch.stack(gt_pose_delta, dim=0).to(device=device, dtype=torch.float32)
        valid_mask = torch.tensor(valid, device=device, dtype=torch.bool)
        return gt_pose_delta, valid_mask
