#!/bin/bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

ANN_ROOT="${ANN_ROOT:-/mnt/petrelfs/linjingli/UMM_Spatial/annotations}"
OUT_ROOT="${OUT_ROOT:-/mnt/petrelfs/linjingli/UMM_Spatial/annotations_pose}"
PREFIX="${PREFIX:-complex_2context_0310}"
SPLITS="${SPLITS:-train test}"
DATASETS="${DATASETS:-scannet matterport3d dl3dv}"

mkdir -p "${OUT_ROOT}"

for SPLIT in ${SPLITS}; do
  for DATASET in ${DATASETS}; do
    IN_JSONL="${ANN_ROOT}/${PREFIX}_${DATASET}_${SPLIT}.jsonl"
    OUT_JSONL="${OUT_ROOT}/${PREFIX}_${DATASET}_${SPLIT}_pose.jsonl"
    echo "[RUN] ${IN_JSONL} -> ${OUT_JSONL}"
    python train/pose_aux/augment_pose_jsonl.py \
      --input_jsonl "${IN_JSONL}" \
      --output_jsonl "${OUT_JSONL}" \
      --set_start_image_id infer \
      --require_valid_pose
  done
done

echo "[DONE] pose jsonl prepared under ${OUT_ROOT}"
