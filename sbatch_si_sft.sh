#!/bin/bash
#SBATCH -J bagel_train
#SBATCH -p efm_t
#SBATCH -N 4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --quotatype=reserved

# --- 1. 设置环境变量 ---
export PYTHONPATH="./:$PYTHONPATH"
export MODEL_PATH="/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT"

# 获取主节点的 IP 地址 (第一个节点)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29533

# WANDB 登录（建议放在外部或脚本内）
export WANDB_API_KEY=9e37f762624801dfc332b03f2ecefbf87153ed8f

# --- 2. 启动命令 ---
# 使用 srun 在每个节点上执行一条 torchrun 命令
# 注意：直接在 srun 后面接 apptainer，torchrun 放在容器内部执行
srun apptainer exec --nv --bind /mnt:/mnt /mnt/inspurfs/efm_t/zhuchenming/apptainer/bagel.sif \
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train/pretrain_unified_navit.py \
    --dataset_config_file ./data/configs/si_example.yaml \
    --model_path $MODEL_PATH \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --total_steps 5000 \
    --warmup_steps 500 \
    --resume-from $MODEL_PATH \
    --finetune_from_hf True \
    --auto_resume True \
    --resume-model-only True \
    --finetune-from-ema True \
    --log_every 1 \
    --results_dir /mnt/inspurfs/efm_t/zhuchenming/train_checkpoints/bagel/scannetpp_si_32gpu_5000steps/logs \
    --checkpoint_dir /mnt/inspurfs/efm_t/zhuchenming/train_checkpoints/bagel/scannetpp_si_32gpu_5000steps/ckpts \
    --lr 8e-5 \
    --visual_und False \
    --num_worker 1 \
    --num_replicate $SLURM_NNODES \
    --save_every 500 \
    --wandb_runid scannetpp_si_32gpu_5000steps_v2 \
    --expected_num_tokens 13240 \
    --max_num_tokens 26520 \
    --max_num_tokens_per_sample 13240