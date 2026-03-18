#!/bin/bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

POSE_CONDITION_MODE="${POSE_CONDITION_MODE:-only}"  # only | append
if [ "${POSE_CONDITION_MODE}" = "append" ]; then
  DATASET_CFG="./data/configs/complex_2context_0310_3data_pose_append.yaml"
  RUN_TAG="complex_2context_0310_3data_pose_append_0318"
else
  DATASET_CFG="./data/configs/complex_2context_0310_3data_pose_only.yaml"
  RUN_TAG="complex_2context_0310_3data_pose_only_0318"
fi

MODEL_PATH="${MODEL_PATH:-/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/inspurfs/mozi_t/linjingli/bagel/models/${RUN_TAG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/train_checkpoints/${RUN_TAG}}"

srun -p efm_t --job-name=train_bagel_pose_cond --gres=gpu:8 --ntasks-per-node=1 -quotatype=reserved \
  apptainer exec --nv --bind /mnt:/mnt /mnt/inspurfs/mozi_t/linjingli/apptainer/mllm3r.sif \
  bash -c "torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29533 \
    train/pretrain_unified_navit.py \
    --dataset_config_file ${DATASET_CFG} \
    --model_path ${MODEL_PATH} \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --total_steps 30000 \
    --resume-from ${MODEL_PATH} \
    --finetune_from_hf True \
    --auto_resume True \
    --resume-model-only True \
    --finetune-from-ema True \
    --log_every 1 \
    --results_dir ${RESULTS_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --lr 2e-5 \
    --visual_und False \
    --num_workers 1 \
    --save_every 5000 \
    --expected_num_tokens 13240 \
    --max_num_tokens 26520 \
    --max_num_tokens_per_sample 13240 \
    --text_cond_dropout_prob 0.0 \
    --wandb_offline True"
