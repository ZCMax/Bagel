#!/bin/bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

# Defaults are aligned with step_mix_refine_onlyone.sh.
# Auto-detect the actual merged-weight directory to avoid loading a wrong path.
if [ -z "${MODEL_PATH:-}" ]; then
  CANDIDATE_A="/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights/step_mix_refine_onlyone0227_0015000"
  CANDIDATE_B="/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ft_weights_old/step_mix_refine_onlyone0227_0015000"
  if [ -d "${CANDIDATE_A}" ]; then
    MODEL_PATH="${CANDIDATE_A}"
  elif [ -d "${CANDIDATE_B}" ]; then
    MODEL_PATH="${CANDIDATE_B}"
  else
    MODEL_PATH="${CANDIDATE_A}"
    echo "[WARN] default model path candidates not found:"
    echo "       - ${CANDIDATE_A}"
    echo "       - ${CANDIDATE_B}"
    echo "       Please set MODEL_PATH explicitly."
  fi
fi
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/data}"
ANN_ROOT="${ANN_ROOT:-/mnt/petrelfs/linjingli/UMM_Spatial/annotations}"
OUT_ROOT="${OUT_ROOT:-/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/outputs}"

# Optional override: set EVAL_PREFIX to evaluate another annotation family.
# Example:
#   EVAL_PREFIX=refine_step_prompt ./batch_evaluation.sh
EVAL_PREFIX="${EVAL_PREFIX:-onlyone_refine_step_prompt}"

MODEL_TAG="$(basename "${MODEL_PATH}")"

for DATASET in scannet dl3dv matterport3d; do
  EVAL_JSON="${ANN_ROOT}/${EVAL_PREFIX}_${DATASET}_test.jsonl"
  OUT_DIR="${OUT_ROOT}/${MODEL_TAG}_${EVAL_PREFIX}_${DATASET}_test"

  echo "[RUN] ${DATASET} -> ${EVAL_JSON}"
  python batch_infer.py \
    --eval-json "${EVAL_JSON}" \
    --image-root "${IMAGE_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --add_context_role_text \
    --num -1 \
    --timestep_shift 1.0
done
