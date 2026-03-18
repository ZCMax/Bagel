#!/bin/bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/complex_2context_0310_3data_pose_only_0318_0030000}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data}"
ANN_ROOT="${ANN_ROOT:-/mnt/petrelfs/linjingli/UMM_Spatial/annotations_pose}"
OUT_ROOT="${OUT_ROOT:-/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs}"
MODE="${MODE:-only}"  # only | append
PREFIX="${PREFIX:-complex_2context_0310}"

MODEL_TAG="$(basename "${MODEL_PATH}")"

for DATASET in scannet dl3dv matterport3d; do
  EVAL_JSON="${ANN_ROOT}/${PREFIX}_${DATASET}_test_pose.jsonl"
  OUT_DIR="${OUT_ROOT}/${MODEL_TAG}_${MODE}_${PREFIX}_${DATASET}_test_pose"

  EXTRA_ARGS=""
  if [ "${MODE}" = "only" ]; then
    EXTRA_ARGS="--pose_text_replace_instruction"
  fi

  echo "[RUN] ${DATASET} -> ${EVAL_JSON}"
  python batch_infer.py \
    --eval-json "${EVAL_JSON}" \
    --image-root "${IMAGE_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --num -1 \
    --timestep_shift 1.0 \
    --inject_pose_text \
    --pose_text_require_valid \
    ${EXTRA_ARGS}
done
