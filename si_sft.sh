#!/bin/bash
export PYTHONPATH="./:$PYTHONPATH"

# 1. 设置模型路径 (请修改为你实际的本地路径)
# 假设你之前在 Gradio 代码里用的路径是这个，如果不是请修改
MODEL_PATH="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--ByteDance-Seed-BAGEL-7B-MoT"

# 2. 设置端口 (防止冲突，可以使用随机端口或指定一个不常用的)
MASTER_PORT=29500

# 3. 启动命令
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr="127.0.0.1" \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/si_example.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --total_steps 5000 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --visual_und False \
  --num_worker 1 \
  --save_every 5000 \
  --expected_num_tokens 13240 \
  --max_num_tokens 26520 \
  --max_num_tokens_per_sample 13240 \
  --wandb_offline True 