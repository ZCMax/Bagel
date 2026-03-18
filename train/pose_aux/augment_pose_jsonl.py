import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.pose_condition import parse_pose_matrix, pick_source_idx  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment jsonl rows with pose matrices: context_poses/target_pose/start_image_id."
    )
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input jsonl path.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl path.")
    parser.add_argument(
        "--set_start_image_id",
        type=str,
        default="infer",
        choices=["keep", "infer", "last"],
        help="How to set start_image_id when writing output rows.",
    )
    parser.add_argument(
        "--require_valid_pose",
        action="store_true",
        help="Drop rows that fail pose extraction.",
    )
    parser.add_argument(
        "--allow_missing_get_more",
        action="store_true",
        help="Do not fail if train.geo_aux.get_more import fails (debug only).",
    )
    return parser.parse_args()


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


def _dataset_from_image_name(image_name: str) -> Optional[str]:
    lower = image_name.lower()
    if lower.startswith("scannet/"):
        return "scannet"
    if lower.startswith("dl3dv/"):
        return "dl3dv"
    if lower.startswith("matterport3d/"):
        return "matterport3d"
    return None


def _load_get_more_module(allow_missing: bool):
    try:
        from train.geo_aux import get_more as gm  # noqa: WPS433

        return gm
    except Exception as e:
        if allow_missing:
            print(f"[WARN] Failed to import train.geo_aux.get_more: {e}")
            return None
        raise RuntimeError(
            "Failed to import train.geo_aux.get_more. "
            "Please install missing dependencies (e.g. cv2) or run with --allow_missing_get_more for debug."
        ) from e


def _fetch_pose_matrix(gm, image_name: str, cache: Dict[str, Optional[Any]]) -> Optional[Any]:
    if image_name in cache:
        return cache[image_name]
    if gm is None:
        cache[image_name] = None
        return None

    dataset = _dataset_from_image_name(image_name)
    fn_name = {
        "scannet": "get_scannet",
        "dl3dv": "get_dl3dv",
        "matterport3d": "get_matterport3d",
    }.get(dataset)
    if fn_name is None or (not hasattr(gm, fn_name)):
        cache[image_name] = None
        return None

    try:
        _k, pose, _rgb, _depth = getattr(gm, fn_name)(image_name)
    except Exception:
        cache[image_name] = None
        return None

    pose_mat = parse_pose_matrix(pose)
    cache[image_name] = pose_mat
    return pose_mat


def _infer_start_image_id(row: Dict[str, Any], num_context: int) -> int:
    idx = pick_source_idx(row)
    if idx is None:
        idx = num_context - 1
    idx = int(max(0, min(int(idx), max(num_context - 1, 0))))
    return idx


def main():
    args = parse_args()
    gm = _load_get_more_module(allow_missing=bool(args.allow_missing_get_more))

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    cache: Dict[str, Optional[Any]] = {}

    total = 0
    kept = 0
    valid_pose = 0
    dropped_invalid = 0
    parse_errors = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin:
        raw_lines = fin.readlines()

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in tqdm(raw_lines, desc="augment_pose_jsonl"):
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                parse_errors += 1
                continue

            if not isinstance(row, dict):
                parse_errors += 1
                continue

            context = row.get("context", [])
            target = row.get("target", None)
            if isinstance(context, str):
                context = [context]

            ok = isinstance(context, list) and len(context) > 0 and isinstance(target, str) and len(target) > 0
            context_mats = []
            target_mat = None
            if ok:
                for image_name in context:
                    canonical = _canonical_image_name(image_name)
                    if canonical is None:
                        ok = False
                        break
                    pose_mat = _fetch_pose_matrix(gm, canonical, cache)
                    if pose_mat is None:
                        ok = False
                        break
                    context_mats.append(pose_mat)
                canonical_target = _canonical_image_name(target)
                if ok and canonical_target is not None:
                    target_mat = _fetch_pose_matrix(gm, canonical_target, cache)
                    if target_mat is None:
                        ok = False
                else:
                    ok = False

            if ok:
                row["context_poses"] = [mat.tolist() for mat in context_mats]
                row["target_pose"] = target_mat.tolist()
                if args.set_start_image_id == "last":
                    row["start_image_id"] = max(len(context_mats) - 1, 0)
                elif args.set_start_image_id == "infer":
                    row["start_image_id"] = _infer_start_image_id(row, len(context_mats))
                elif args.set_start_image_id == "keep":
                    if "start_image_id" not in row:
                        row["start_image_id"] = _infer_start_image_id(row, len(context_mats))
                valid_pose += 1
            elif args.require_valid_pose:
                dropped_invalid += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(
        "[DONE] "
        f"input={args.input_jsonl} output={args.output_jsonl} "
        f"total={total} kept={kept} valid_pose={valid_pose} dropped_invalid={dropped_invalid} "
        f"parse_errors={parse_errors} cache_size={len(cache)}"
    )


if __name__ == "__main__":
    main()
