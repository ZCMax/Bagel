import math
from typing import Any, Optional, Sequence

import numpy as np


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
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


def so3_log_map(R: np.ndarray) -> np.ndarray:
    R = R.astype(np.float64)
    tr = float(np.trace(R))
    cos_theta = max(min((tr - 1.0) * 0.5, 1.0), -1.0)
    theta = math.acos(cos_theta)

    skew = 0.5 * (R - R.T)
    vee = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)

    if theta < 1e-6:
        return vee

    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-6:
        return vee

    return (theta / sin_theta) * vee


def relative_pose_delta_se3(source_c2w: np.ndarray, target_c2w: np.ndarray) -> Optional[np.ndarray]:
    try:
        delta = np.linalg.inv(source_c2w) @ target_c2w
    except np.linalg.LinAlgError:
        return None

    t = delta[:3, 3]
    r = so3_log_map(delta[:3, :3])
    out = np.concatenate([t, r], axis=0).astype(np.float32)
    if not np.all(np.isfinite(out)):
        return None
    return out


def infer_source_idx_from_instruction(instruction: str) -> Optional[int]:
    if not isinstance(instruction, str):
        return None
    lower = instruction.lower()
    if ("second image" in lower) or ("last image" in lower):
        return 1
    if ("first image" in lower) or ("initial image" in lower):
        return 0
    return None


def pick_source_idx(row: dict) -> Optional[int]:
    for key in ("start_image_id", "start_idx", "source_idx", "source_index", "reference_idx", "src_idx"):
        if key in row:
            try:
                idx = int(row[key])
            except Exception:
                continue
            return idx
    return infer_source_idx_from_instruction(row.get("instruction", ""))


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
