import json
import os
from typing import Dict, List, Optional, Tuple, Any

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


class TeacherFeatureBank:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.root_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_specs = manifest.get("datasets", {})
        if len(dataset_specs) == 0:
            raise ValueError(f"No dataset entries found in manifest: {manifest_path}")

        self.feature_dim = None
        self.features: Dict[str, torch.Tensor] = {}
        self.valid_masks: Dict[str, torch.Tensor] = {}

        for dataset_name, spec in dataset_specs.items():
            feat_rel = spec["features_path"]
            feat_path = feat_rel if os.path.isabs(feat_rel) else os.path.join(self.root_dir, feat_rel)
            feats = torch.load(feat_path, map_location="cpu")
            if isinstance(feats, dict) and "features" in feats:
                feats = feats["features"]
            if not torch.is_tensor(feats):
                raise TypeError(f"Features at {feat_path} must be a tensor")
            feats = feats.float().contiguous()
            if feats.dim() != 2:
                raise ValueError(f"Features at {feat_path} must have shape [N, D], got {tuple(feats.shape)}")
            if self.feature_dim is None:
                self.feature_dim = feats.shape[1]
            elif feats.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Feature dim mismatch for dataset '{dataset_name}': "
                    f"expected {self.feature_dim}, got {feats.shape[1]}"
                )

            valid_rel = spec.get("valid_mask_path")
            if valid_rel is not None:
                valid_path = valid_rel if os.path.isabs(valid_rel) else os.path.join(self.root_dir, valid_rel)
                valid_mask = torch.load(valid_path, map_location="cpu")
                if not torch.is_tensor(valid_mask):
                    raise TypeError(f"Valid mask at {valid_path} must be a tensor")
                valid_mask = valid_mask.to(torch.bool).flatten()
                if valid_mask.numel() != feats.shape[0]:
                    raise ValueError(
                        f"valid_mask length mismatch for dataset '{dataset_name}': "
                        f"{valid_mask.numel()} vs features {feats.shape[0]}"
                    )
            else:
                valid_mask = torch.ones(feats.shape[0], dtype=torch.bool)

            self.features[dataset_name] = feats
            self.valid_masks[dataset_name] = valid_mask

    def lookup_batch(
        self,
        batch_data_indexes: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        teacher_features = []
        valid = []

        for item in batch_data_indexes:
            dataset_name = item.get("dataset_name")
            row_idx = _safe_int_index(item.get("data_indexes"))

            if dataset_name not in self.features or row_idx is None:
                teacher_features.append(torch.zeros(self.feature_dim, dtype=torch.float32))
                valid.append(False)
                continue

            feats = self.features[dataset_name]
            mask = self.valid_masks[dataset_name]
            if row_idx < 0 or row_idx >= feats.shape[0]:
                teacher_features.append(torch.zeros(self.feature_dim, dtype=torch.float32))
                valid.append(False)
                continue

            teacher_features.append(feats[row_idx])
            valid.append(bool(mask[row_idx]))

        teacher_features = torch.stack(teacher_features, dim=0).to(device=device, dtype=torch.float32)
        valid_mask = torch.tensor(valid, device=device, dtype=torch.bool)
        return teacher_features, valid_mask
