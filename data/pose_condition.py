from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


CONTEXT_POSE_KEYS = (
    "context_poses",
    "context_pose_list",
    "context_pose",
    "context_c2w",
    "input_poses",
)

TARGET_POSE_KEYS = (
    "target_pose",
    "target_c2w",
    "gt_pose",
    "gt_c2w",
)

START_IMAGE_ID_KEYS = (
    "start_image_id",
    "start_idx",
    "source_idx",
    "source_index",
    "reference_idx",
    "src_idx",
)


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0 or (not np.all(np.isfinite(arr))):
        return None
    return arr


def parse_pose_matrix(pose_obj: Any) -> Optional[np.ndarray]:
    if pose_obj is None:
        return None

    if isinstance(pose_obj, dict):
        for key in ("c2w", "pose", "matrix", "transform", "extrinsic"):
            if key in pose_obj:
                return parse_pose_matrix(pose_obj[key])
        return None

    arr = _as_float_array(pose_obj)
    if arr is None:
        return None

    if arr.shape == (4, 4):
        mat = arr
    elif arr.shape == (3, 4):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :] = arr
    elif arr.shape == (3, 3):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = arr
    elif arr.ndim == 1 and arr.size == 16:
        mat = arr.reshape(4, 4)
    elif arr.ndim == 1 and arr.size == 12:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :] = arr.reshape(3, 4)
    else:
        return None

    if not np.all(np.isfinite(mat)):
        return None
    return mat.astype(np.float64)


def parse_pose_list(field_obj: Any) -> Optional[Sequence[np.ndarray]]:
    if not isinstance(field_obj, list):
        return None
    mats = []
    for item in field_obj:
        mat = parse_pose_matrix(item)
        if mat is None:
            return None
        mats.append(mat)
    return mats


def infer_source_idx_from_instruction(instruction: Any) -> Optional[int]:
    if not isinstance(instruction, str):
        return None
    lower = instruction.lower()
    if ("second image" in lower) or ("last image" in lower):
        return 1
    if ("first image" in lower) or ("initial image" in lower):
        return 0
    return None


def pick_source_idx(row: Dict[str, Any]) -> Optional[int]:
    for key in START_IMAGE_ID_KEYS:
        if key not in row:
            continue
        try:
            return int(row[key])
        except Exception:
            continue
    return infer_source_idx_from_instruction(row.get("instruction", ""))


def extract_pose_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    context_poses = None
    for key in CONTEXT_POSE_KEYS:
        if key in row:
            context_poses = parse_pose_list(row[key])
            if context_poses is not None:
                break

    target_pose = None
    for key in TARGET_POSE_KEYS:
        if key in row:
            target_pose = parse_pose_matrix(row[key])
            if target_pose is not None:
                break

    source_idx = pick_source_idx(row)
    if context_poses is not None and len(context_poses) > 0:
        if source_idx is None:
            source_idx = len(context_poses) - 1
        source_idx = max(0, min(int(source_idx), len(context_poses) - 1))

    return {
        "context_poses": context_poses,
        "target_pose": target_pose,
        "start_image_id": source_idx,
        "valid": bool(context_poses is not None and target_pose is not None),
    }


def _format_matrix_4x4(mat: np.ndarray, precision: int = 4) -> str:
    rows = []
    for row in mat:
        rows.append(", ".join(f"{float(v):.{precision}f}" for v in row.tolist()))
    return "[" + "; ".join(rows) + "]"


def build_pose_condition_text(
    row: Dict[str, Any],
    include_start_image_id: bool = True,
    precision: int = 4,
) -> Optional[str]:
    payload = extract_pose_payload(row)
    if not payload["valid"]:
        return None

    context_poses = payload["context_poses"]
    target_pose = payload["target_pose"]
    start_image_id = payload["start_image_id"]

    lines = ["[POSE_MATRIX_CONDITION]"]
    if include_start_image_id and start_image_id is not None:
        lines.append(f"start_image_id={int(start_image_id)}")

    for idx, pose in enumerate(context_poses):
        lines.append(f"context_pose_{idx}={_format_matrix_4x4(pose, precision=precision)}")

    lines.append(f"target_pose={_format_matrix_4x4(target_pose, precision=precision)}")
    lines.append("[/POSE_MATRIX_CONDITION]")
    return "\n".join(lines)


def compose_instruction_with_pose(
    instruction: str,
    row: Dict[str, Any],
    inject_pose_text: bool = False,
    replace_instruction: bool = False,
    include_start_image_id: bool = True,
    precision: int = 4,
    fallback_instruction: str = "Generate the target view with the provided pose matrices.",
) -> tuple[str, bool]:
    base_instruction = instruction.strip() if isinstance(instruction, str) else ""
    if not inject_pose_text:
        return base_instruction, False

    pose_text = build_pose_condition_text(
        row,
        include_start_image_id=include_start_image_id,
        precision=precision,
    )
    if pose_text is None:
        return base_instruction, False

    if replace_instruction:
        return f"{pose_text}\n{fallback_instruction}", True

    if len(base_instruction) == 0:
        return pose_text, True

    return f"{base_instruction}\n{pose_text}", True
