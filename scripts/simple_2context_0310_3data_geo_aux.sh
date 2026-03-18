#!/bin/bash
export PYTHONPATH="./:$PYTHONPATH"

# 1. 设置模型路径 (请修改为你实际的本地路径)
# 假设你之前在 Gradio 代码里用的路径是这个，如果不是请修改
MODEL_PATH="/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT"

# 2. 设置端口 (防止冲突，可以使用随机端口或指定一个不常用的)
MASTER_PORT=29500

# 3. 启动命令


srun -p efm_t --job-name=train_bagel_8gpu --gres=gpu:8 --ntasks-per-node=1 -quotatype=reserved apptainer exec --nv  --bind /mnt:/mnt /mnt/inspurfs/mozi_t/linjingli/apptainer/mllm3r.sif \
    bash -c 'torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --nproc_per_node=8 \
      --master_addr="127.0.0.1" \
      --master_port=29533 \
      train/pretrain_unified_navit.py \
      --dataset_config_file ./data/configs/simple_2context_0310_3data.yaml \
      --model_path /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
      --layer_module Qwen2MoTDecoderLayer \
      --max_latent_size 64 \
      --total_steps 30000 \
      --resume-from /mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT \
      --finetune_from_hf True \
      --auto_resume True \
      --resume-model-only True \
      --finetune-from-ema True \
      --log_every 1 \
      --results_dir /mnt/inspurfs/mozi_t/linjingli/bagel/models/simple_2context_0310_3data_geo_aux_0310 \
      --checkpoint_dir /mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/train_checkpoints/simple_2context_0310_3data_geo_aux_0310 \
      --lr 2e-5 \
      --visual_und False \
      --num_workers 1 \
      --save_every 5000 \
      --expected_num_tokens 13240 \
      --max_num_tokens 26520 \
      --max_num_tokens_per_sample 13240 \
      --geo_aux_enable True \
      --geo_aux_manifest /mnt/inspurfs/efm_t/linjingli/data/geos/simple_2context_0310_3data/manifest.json \
      --geo_aux_weight 0.05 \
      --geo_pose_weight 0.05 \
      --geo_depth_weight 0.05 \
      --geo_aux_warmup_steps 3000 \
      --geo_pose_hidden_dim 512 \
      --geo_depth_hidden_dim 256 \
      --geo_aux_lr 1e-4 \
      --geo_ssim_weight 0.3 \
      --geo_depth_si_lambda 0.5 \
      --geo_depth_tol 0.05 \
      --geo_aux_head_bf16 True \
      --geo_reproj_max_samples 1 \
      --geo_decode_scale 0.5 \
      --wandb_offline True '