import argparse
import json
import os
import random
from contextlib import nullcontext
import sys
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_info import DATASET_INFO

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except Exception:
    VGGT = None
    load_and_preprocess_images = None


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute VGGT feature bank for spatial distillation.")
    parser.add_argument("--dataset_config_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vggt_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--fallback_dim", type=int, default=1024)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


class VGGTFeatureExtractor:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        if VGGT is None or load_and_preprocess_images is None:
            raise ImportError("VGGT is not installed. Please install `vggt` first.")
        self.device = device
        self.dtype = dtype
        self.model = VGGT.from_pretrained(model_path).to(device).eval()

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        images = load_and_preprocess_images([image_path]).to(self.device)[None]
        use_amp = self.device.startswith("cuda") and torch.cuda.is_available() and self.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        amp_ctx = torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True) if use_amp else nullcontext()
        with amp_ctx:
            tokens, _ = self.model.aggregator(images)

        feat = self._pool_tokens(tokens).float().cpu()
        return feat

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if tokens.dim() == 5:  # [B, V, T, P, D] or similar
            return tokens.mean(dim=(1, 2, 3))[0]
        if tokens.dim() == 4:  # [B, V, T, D]
            return tokens.mean(dim=(1, 2))[0]
        if tokens.dim() == 3:  # [B, T, D]
            return tokens.mean(dim=1)[0]
        if tokens.dim() == 2:  # [T, D]
            return tokens.mean(dim=0)
        raise ValueError(f"Unsupported token shape from VGGT: {tuple(tokens.shape)}")


def collect_rows_from_group(group_name: str, group_cfg: Dict) -> List[Tuple[int, Optional[str]]]:
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

    rows: List[Tuple[int, Optional[str]]] = []
    row_idx = 0
    for dataset_name, limit in zip(dataset_names, num_used_data):
        meta = DATASET_INFO[dataset_type][dataset_name]
        if "jsonl_path" not in meta:
            raise ValueError(
                f"group '{group_name}' / dataset '{dataset_name}' does not provide jsonl_path; "
                "only jsonl datasets are supported by this precompute script."
            )

        jsonl_path = meta["jsonl_path"]
        image_dir = meta["data_dir"]
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        if shuffle_lines:
            rng = random.Random(shuffle_seed)
            rng.shuffle(raw_lines)

        if limit is not None:
            raw_lines = raw_lines[: int(limit)]

        for line in raw_lines:
            target_abs = None
            try:
                item = json.loads(line)
                target_rel = item.get("target")
                if isinstance(target_rel, str) and len(target_rel) > 0:
                    target_abs = os.path.join(image_dir, target_rel)
            except Exception:
                target_abs = None
            rows.append((row_idx, target_abs))
            row_idx += 1
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    extractor = VGGTFeatureExtractor(args.vggt_model_path, args.device, dtype)

    with open(args.dataset_config_file, "r", encoding="utf-8") as f:
        dataset_meta = yaml.safe_load(f)

    manifest = {"datasets": {}}
    global_feature_dim = None

    for group_name, group_cfg in dataset_meta.items():
        rows = collect_rows_from_group(group_name, group_cfg)
        if len(rows) == 0:
            continue

        features: List[Optional[torch.Tensor]] = []
        valid_mask = []
        group_feature_dim = None

        for _, target_abs in tqdm(rows, desc=f"[VGGT] {group_name}"):
            if target_abs is None or (not os.path.exists(target_abs)):
                features.append(None)
                valid_mask.append(False)
                continue

            try:
                feat = extractor.encode_image(target_abs)
                features.append(feat)
                valid_mask.append(True)
                if group_feature_dim is None:
                    group_feature_dim = feat.shape[-1]
            except Exception:
                features.append(None)
                valid_mask.append(False)

        if group_feature_dim is None:
            group_feature_dim = global_feature_dim if global_feature_dim is not None else int(args.fallback_dim)
        if global_feature_dim is None:
            global_feature_dim = group_feature_dim

        fixed_features = []
        for feat in features:
            if feat is None:
                fixed_features.append(torch.zeros(group_feature_dim, dtype=torch.float32))
            elif feat.shape[-1] != group_feature_dim:
                fixed_features.append(torch.zeros(group_feature_dim, dtype=torch.float32))
            else:
                fixed_features.append(feat.to(torch.float32))

        feature_tensor = torch.stack(fixed_features, dim=0).contiguous()
        valid_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        feat_name = f"{group_name}.features.pt"
        valid_name = f"{group_name}.valid.pt"
        feat_path = os.path.join(args.output_dir, feat_name)
        valid_path = os.path.join(args.output_dir, valid_name)
        torch.save(feature_tensor, feat_path)
        torch.save(valid_tensor, valid_path)

        manifest["datasets"][group_name] = {
            "features_path": feat_name,
            "valid_mask_path": valid_name,
            "num_rows": int(feature_tensor.shape[0]),
            "feature_dim": int(feature_tensor.shape[1]),
            "valid_rows": int(valid_tensor.sum().item()),
        }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
