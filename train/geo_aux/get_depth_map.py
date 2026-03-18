import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.dataloader import read_image_cv2_local, resize_image_half

DEFAULT_SCANNET_ROOT = Path("/mnt/inspurfs/mozi_t/linjingli/transfer/ScanNet_v2")

DATASET_ALIASES = {
    "dl3dv": "dl3dv",
    "scannet": "scannet",
    "scannetpp": "scannetpp",
    "arkitscene": "arkitscenes",
    "arkitscenes": "arkitscenes",
}


def _normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower().replace("-", "").replace("_", "")
    if key not in DATASET_ALIASES:
        allowed = ", ".join(sorted(DATASET_ALIASES.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {allowed}")
    return DATASET_ALIASES[key]


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _apply_percentile_cleanup(depthmap: np.ndarray) -> np.ndarray:
    depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
    valid = depthmap > 0
    if valid.any():
        threshold = np.percentile(depthmap[valid], 98)
        depthmap[depthmap > threshold] = 0.0
    return depthmap.astype(np.float32)


def _resolve_scannet_image_path(frame_name: str, scannet_root: Path) -> Path:
    image_path = Path(frame_name)
    if image_path.exists():
        return image_path

    frame_name_str = str(frame_name)
    if frame_name_str.startswith("scannet/"):
        candidate = scannet_root / frame_name_str[len("scannet/") :]
        if candidate.exists():
            return candidate

    if "scannet" in frame_name_str:
        candidate = Path(frame_name_str.replace("scannet", str(scannet_root), 1))
        if candidate.exists():
            return candidate

    return image_path


def _build_scannet_depth_path(image_path: Path) -> Path:
    if image_path.parent.name == "color":
        return image_path.parent.parent / "depth" / f"{image_path.stem}.png"

    path_str = str(image_path)
    if "/color/" in path_str:
        return Path(path_str.replace("/color/", "/depth/")).with_suffix(".png")
    if "\\color\\" in path_str:
        return Path(path_str.replace("\\color\\", "\\depth\\")).with_suffix(".png")
    return Path(path_str.replace("jpg", "png"))


def _process_dl3dv(frame_name: str) -> np.ndarray:
    image_path = Path(frame_name)
    _require_file(image_path, "DL3DV frame")

    rgb_image = read_image_cv2_local(str(image_path))
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot read DL3DV frame: {image_path}")

    scene_dir = image_path.parent.parent
    stem = image_path.stem
    depth_path = scene_dir / "dense" / "depth" / f"{stem}.npy"
    sky_mask_path = scene_dir / "dense" / "sky_mask" / f"{stem}.png"
    outlier_mask_path = scene_dir / "dense" / "outlier_mask" / f"{stem}.png"

    _require_file(depth_path, "DL3DV depth")
    _require_file(sky_mask_path, "DL3DV sky mask")
    _require_file(outlier_mask_path, "DL3DV outlier mask")

    depthmap = np.load(depth_path).astype(np.float32)
    depthmap[~np.isfinite(depthmap)] = 0.0

    with Image.open(sky_mask_path) as img:
        sky_mask = np.array(img).astype(np.float32) >= 127
    with Image.open(outlier_mask_path) as img:
        outlier_mask = np.array(img).astype(np.float32)

    depthmap[sky_mask] = -1.0
    depthmap[outlier_mask >= 127] = 0.0
    depthmap = _apply_percentile_cleanup(depthmap)

    h, w = rgb_image.shape[:2]
    depthmap = cv2.resize(depthmap, (w, h), interpolation=cv2.INTER_NEAREST)
    return depthmap.astype(np.float32)


def _process_scannetpp(frame_name: str) -> np.ndarray:
    image_path = Path(frame_name)
    _require_file(image_path, "ScanNet++ frame")

    depth_path = image_path.parent.parent / "depth" / f"{image_path.stem}.png"
    _require_file(depth_path, "ScanNet++ depth")

    with Image.open(depth_path) as depth_img:
        depthmap = np.array(depth_img).astype(np.int32)

    depthmap = depthmap.astype(np.float32) / 1000.0
    depthmap[~np.isfinite(depthmap)] = 0.0
    depthmap = _apply_percentile_cleanup(depthmap)
    return resize_image_half(depthmap).astype(np.float32)


def _process_arkitscenes(frame_name: str) -> np.ndarray:
    image_path = Path(frame_name)
    _require_file(image_path, "ARKitScenes frame")

    rgb_image = read_image_cv2_local(str(image_path))
    if rgb_image is None:
        raise FileNotFoundError(f"Cannot read ARKitScenes frame: {image_path}")

    depth_path = image_path.parent.parent / "lowres_depth" / f"{image_path.stem}.png"
    _require_file(depth_path, "ARKitScenes depth")

    with Image.open(depth_path) as depth_img:
        depthmap = np.array(depth_img).astype(np.int32)

    depthmap = depthmap.astype(np.float32) / 1000.0
    h, w = rgb_image.shape[:2]
    depthmap = cv2.resize(depthmap, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
    depthmap[~np.isfinite(depthmap)] = 0.0
    return resize_image_half(depthmap).astype(np.float32)


def _process_scannet(frame_name: str, scannet_root: Path) -> np.ndarray:
    image_path = _resolve_scannet_image_path(frame_name, scannet_root)
    _require_file(image_path, "ScanNet frame")

    depth_path = _build_scannet_depth_path(image_path)
    _require_file(depth_path, "ScanNet depth")

    rgb_image_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_image_raw is None:
        raise FileNotFoundError(f"Cannot read ScanNet frame: {image_path}")
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot read ScanNet depth: {depth_path}")

    rgb_image = resize_image_half(rgb_image_raw)
    depthmap = depth_raw.astype(np.float32) / 1000.0
    depthmap[~np.isfinite(depthmap)] = 0.0

    if depthmap.shape[:2] != rgb_image.shape[:2]:
        depthmap = cv2.resize(
            depthmap,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    depthmap = _apply_percentile_cleanup(depthmap)
    return resize_image_half(depthmap).astype(np.float32)


def get_processed_depth_map(
    dataset_name: str,
    frame_name: str,
    scannet_root: str = str(DEFAULT_SCANNET_ROOT),
) -> np.ndarray:
    dataset = _normalize_dataset_name(dataset_name)
    if dataset == "dl3dv":
        return _process_dl3dv(frame_name)
    if dataset == "scannetpp":
        return _process_scannetpp(frame_name)
    if dataset == "arkitscenes":
        return _process_arkitscenes(frame_name)
    if dataset == "scannet":
        return _process_scannet(frame_name, Path(scannet_root))
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def save_depth_png_mm(depthmap: np.ndarray, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depthmap * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    ok = cv2.imwrite(str(output), depth_mm)
    if not ok:
        raise IOError(f"Failed to write depth image: {output}")


def save_depth_visualization(depthmap: np.ndarray, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    valid = depthmap > 0
    norm = np.zeros_like(depthmap, dtype=np.uint8)
    if valid.any():
        depth_valid = depthmap[valid]
        vmin = np.percentile(depth_valid, 2)
        vmax = np.percentile(depth_valid, 98)
        if vmax <= vmin:
            vmax = max(vmin + 1e-6, float(depth_valid.max()))
        norm = np.clip((depthmap - vmin) / (vmax - vmin), 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)

    vis = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    ok = cv2.imwrite(str(output), vis)
    if not ok:
        raise IOError(f"Failed to write depth visualization: {output}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate processed depth map for one frame path."
    )
    parser.add_argument("--dataset", required=True, help="dl3dv/scannet/scannetpp/arkitscenes")
    parser.add_argument("--frame-name", required=True, help="Frame path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for processed depth PNG (uint16, millimeters).",
    )
    parser.add_argument(
        "--vis-output",
        default=None,
        help="Optional colorized depth visualization output path.",
    )
    parser.add_argument(
        "--scannet-root",
        default=str(DEFAULT_SCANNET_ROOT),
        help="ScanNet root used when frame_name starts with 'scannet/'.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    depthmap = get_processed_depth_map(
        dataset_name=args.dataset,
        frame_name=args.frame_name,
        scannet_root=args.scannet_root,
    )
    save_depth_png_mm(depthmap, args.output)
    if args.vis_output:
        save_depth_visualization(depthmap, args.vis_output)

    valid = depthmap > 0
    valid_count = int(valid.sum())
    depth_max = float(depthmap.max()) if depthmap.size else 0.0
    depth_min_valid = float(depthmap[valid].min()) if valid_count > 0 else 0.0
    print(
        f"Saved depth: {args.output} | shape={depthmap.shape} | "
        f"valid_pixels={valid_count} | min_valid={depth_min_valid:.4f}m | max={depth_max:.4f}m"
    )
    if args.vis_output:
        print(f"Saved visualization: {args.vis_output}")


if __name__ == "__main__":
    main()
