# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import json
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)
from train.spatial_distill import (
    SpatialDistillAdapter,
    TeacherFeatureBank,
    compute_alignment_loss,
    pool_mse_preds_by_sample,
)
from train.pose_aux import (
    PoseAuxHead,
    PoseDeltaBank,
    PoseProbeBuffer,
    compute_pose_aux_loss,
    evaluate_pose_linear_probe,
)
from train.geo_aux import (
    GeoAuxBank,
    GeoPoseHead,
    GeoDepthHead,
    split_tokens_by_hw,
    unpatchify_latent_tokens,
    se3_l1_pose_loss,
    scale_invariant_log_depth_loss,
    reprojection_consistency_loss,
)
from torchvision.utils import save_image


def _to_hashable_index(index_obj):
    if isinstance(index_obj, list):
        return tuple(_to_hashable_index(item) for item in index_obj)
    if isinstance(index_obj, tuple):
        return tuple(_to_hashable_index(item) for item in index_obj)
    if isinstance(index_obj, dict):
        return tuple(sorted((k, _to_hashable_index(v)) for k, v in index_obj.items()))
    return index_obj


def _build_geo_image_mapping(
    num_context: torch.Tensor,
    source_idx: torch.Tensor,
    patchified_vae_latent_shapes,
):
    """
    Build per-sample source/target image indices in `padded_images`:
      [context_0, ..., context_{k-1}, target]
    """
    src_img_indices = []
    tgt_img_indices = []
    tgt_hw_list = []
    offset = 0

    for nctx_t, sidx_t in zip(num_context.tolist(), source_idx.tolist()):
        nctx = int(max(nctx_t, 0))
        if nctx == 0:
            return None, None, None
        sidx = int(max(0, min(int(sidx_t), nctx - 1)))
        src_idx_global = offset + sidx
        tgt_idx_global = offset + nctx
        if tgt_idx_global >= len(patchified_vae_latent_shapes):
            return None, None, None
        src_img_indices.append(src_idx_global)
        tgt_img_indices.append(tgt_idx_global)
        tgt_hw_list.append(patchified_vae_latent_shapes[tgt_idx_global])
        offset += nctx + 1

    if offset != len(patchified_vae_latent_shapes):
        return None, None, None
    return src_img_indices, tgt_img_indices, tgt_hw_list


def _run_geo_aux_startup_selfcheck(
    train_dataset,
    geo_aux_bank: GeoAuxBank,
    logger,
    rank: int,
    max_samples_per_group: int = 256,
    strict: bool = True,
):
    if max_samples_per_group <= 0:
        logger.info("[GeoAuxSelfCheck] skip: max_samples_per_group <= 0")
        return

    train_group_names = [dataset.dataset_name for dataset in train_dataset.grouped_datasets]
    bank_group_names = set(geo_aux_bank.sample_id_to_row.keys())

    extra_bank_groups = sorted(bank_group_names - set(train_group_names))
    if len(extra_bank_groups) > 0:
        logger.info(
            "[GeoAuxSelfCheck][rank=%d] bank contains groups not used by this rank: %s",
            rank,
            ",".join(extra_bank_groups),
        )

    has_failure = False
    for dataset in train_dataset.grouped_datasets:
        dataset_name = dataset.dataset_name
        sample_id_map = geo_aux_bank.sample_id_to_row.get(dataset_name)
        if sample_id_map is None:
            logger.warning(
                "[GeoAuxSelfCheck][rank=%d] dataset=%s not found in geo bank; "
                "geo losses for this dataset will be invalid.",
                rank,
                dataset_name,
            )
            continue

        data_paths = getattr(dataset, "data_paths_per_rank", None)
        if data_paths is None:
            data_paths = getattr(dataset, "data_paths", None)
        if data_paths is None:
            logger.warning(
                "[GeoAuxSelfCheck][rank=%d] dataset=%s has no data_paths for probing.",
                rank,
                dataset_name,
            )
            continue

        checked = 0
        probed = 0
        matched = 0
        missing_id = 0
        missing_in_bank = 0
        parse_error = 0
        non_json = 0

        for item in data_paths:
            if probed >= max_samples_per_group:
                break
            probed += 1
            if not isinstance(item, tuple) or len(item) == 0 or not isinstance(item[0], str):
                non_json += 1
                continue
            line = item[0]
            try:
                row = json.loads(line)
            except Exception:
                parse_error += 1
                continue
            if not isinstance(row, dict):
                parse_error += 1
                continue

            checked += 1
            sample_id = row.get("id", row.get("sample_id", row.get("uid")))
            if sample_id is None:
                missing_id += 1
                continue
            if str(sample_id) in sample_id_map:
                matched += 1
            else:
                missing_in_bank += 1

        valid_checked = max(checked - missing_id, 0)
        match_ratio = float(matched) / float(max(valid_checked, 1))
        logger.info(
            "[GeoAuxSelfCheck][rank=%d] dataset=%s checked=%d matched=%d "
            "missing_id=%d missing_in_bank=%d parse_error=%d non_json=%d bank_size=%d "
            "match_ratio=%.4f probed=%d",
            rank,
            dataset_name,
            checked,
            matched,
            missing_id,
            missing_in_bank,
            parse_error,
            non_json,
            len(sample_id_map),
            match_ratio,
            probed,
        )

        if strict and (missing_id > 0 or missing_in_bank > 0):
            has_failure = True

    if strict and has_failure:
        raise RuntimeError(
            "geo_aux startup self-check failed: sample_id alignment mismatch detected. "
            "Please regenerate geo bank with updated precompute_geo_bank.py and ensure json rows contain stable `id`."
        )


def save_images_png(
    images: torch.Tensor,
    save_dir: str,
    prefix: str = "sample",
    value_range: tuple = (-1, 1),
):
    """
    Save a batch of images as PNG using torchvision.

    Args:
        images: Tensor (B, C, H, W)
        save_dir: directory to save png files
        prefix: filename prefix
        value_range: value range of images (default assumes VAE output)
    """
    os.makedirs(save_dir, exist_ok=True)

    images = images.detach().cpu()

    for i in range(images.shape[0]):
        save_image(
            images[i],
            os.path.join(save_dir, f"{prefix}_{i:03d}.png"),
            normalize=True,
            value_range=value_range,
        )

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )
    spatial_distill_enable: bool = field(
        default=False,
        metadata={"help": "Enable optional spatial distillation loss from precomputed teacher features."}
    )
    spatial_distill_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to feature-bank manifest.json generated by precompute_vggt_features.py."}
    )
    spatial_distill_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for the auxiliary spatial distillation loss."}
    )
    spatial_distill_loss_type: str = field(
        default="cosine",
        metadata={"help": "Distillation loss type: cosine or mse."}
    )
    spatial_distill_hidden_dim: int = field(
        default=0,
        metadata={"help": "Hidden size for the distillation adapter MLP. 0 means single linear layer."}
    )
    spatial_distill_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the distillation adapter optimizer."}
    )
    pose_aux_enable: bool = field(
        default=False,
        metadata={"help": "Enable optional pose-delta auxiliary supervision."}
    )
    pose_aux_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pose bank manifest.json generated by precompute_pose_delta_bank.py."}
    )
    pose_aux_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for pose auxiliary loss."}
    )
    pose_aux_warmup_steps: int = field(
        default=3000,
        metadata={"help": "Linear warmup steps for pose auxiliary loss weight."}
    )
    pose_aux_loss_type: str = field(
        default="smooth_l1",
        metadata={"help": "Pose auxiliary loss type: smooth_l1 or mse."}
    )
    pose_aux_hidden_dim: int = field(
        default=1024,
        metadata={"help": "Hidden dimension for pose auxiliary head."}
    )
    pose_aux_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for pose auxiliary head optimizer."}
    )
    pose_aux_trans_norm: float = field(
        default=1.0,
        metadata={"help": "Translation normalization factor for pose labels."}
    )
    pose_aux_trans_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of translation term inside pose loss."}
    )
    pose_aux_rot_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of rotation term inside pose loss."}
    )
    pose_aux_smooth_l1_beta: float = field(
        default=1.0,
        metadata={"help": "Beta for smooth_l1 pose loss."}
    )
    pose_probe_enable: bool = field(
        default=False,
        metadata={"help": "Enable periodic frozen-feature pose probe evaluation (for geometry learning diagnostics)."}
    )
    pose_probe_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Pose bank manifest for probe evaluation. If not set, reuse pose_aux_manifest."}
    )
    pose_probe_every: int = field(
        default=1000,
        metadata={"help": "Run pose probe every N optimizer steps."}
    )
    pose_probe_min_samples: int = field(
        default=512,
        metadata={"help": "Minimum buffered valid samples required before running probe."}
    )
    pose_probe_max_samples: int = field(
        default=4096,
        metadata={"help": "Max number of recent samples kept in probe buffer per rank."}
    )
    pose_probe_val_ratio: float = field(
        default=0.2,
        metadata={"help": "Validation split ratio for linear probe evaluation."}
    )
    pose_probe_ridge: float = field(
        default=1e-4,
        metadata={"help": "Ridge coefficient for closed-form linear probe."}
    )
    pose_probe_seed: int = field(
        default=3407,
        metadata={"help": "Random seed used for train/val split in pose probe."}
    )
    geo_aux_enable: bool = field(
        default=False,
        metadata={"help": "Enable geometry closed-loop auxiliary losses (reprojection + pose + depth)."}
    )
    geo_aux_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Path to geometry aux manifest generated by precompute_geo_bank.py."}
    )
    geo_aux_weight: float = field(
        default=0.05,
        metadata={"help": "Weight alpha for reprojection consistency loss L_geo."}
    )
    geo_pose_weight: float = field(
        default=0.05,
        metadata={"help": "Weight beta for pose regression loss L_pose."}
    )
    geo_depth_weight: float = field(
        default=0.05,
        metadata={"help": "Weight theta for depth multi-task loss L_depth."}
    )
    geo_aux_warmup_steps: int = field(
        default=3000,
        metadata={"help": "Linear warmup steps for geometry auxiliary weights."}
    )
    geo_pose_hidden_dim: int = field(
        default=512,
        metadata={"help": "Hidden size for geometry pose head."}
    )
    geo_depth_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden size for geometry depth head."}
    )
    geo_aux_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for geometry auxiliary heads."}
    )
    geo_ssim_weight: float = field(
        default=0.3,
        metadata={"help": "Weight of SSIM term inside reprojection photometric loss."}
    )
    geo_depth_si_lambda: float = field(
        default=0.5,
        metadata={"help": "Scale-invariant depth loss lambda term."}
    )
    geo_depth_tol: float = field(
        default=0.05,
        metadata={"help": "Depth consistency tolerance ratio for reprojection visibility filtering."}
    )
    geo_reproj_max_samples: int = field(
        default=1,
        metadata={"help": "Max number of samples per packed batch used for reprojection loss (memory control)."}
    )
    geo_decode_scale: float = field(
        default=1.0,
        metadata={"help": "Optional latent downscale factor before VAE decode for reprojection branch. <=1.0."}
    )
    geo_aux_head_bf16: bool = field(
        default=True,
        metadata={"help": "Cast geo pose/depth auxiliary heads to bfloat16 to reduce memory."}
    )
    geo_startup_selfcheck: bool = field(
        default=True,
        metadata={"help": "Run startup sample_id alignment self-check for geo_aux before training loop."}
    )
    geo_startup_selfcheck_samples: int = field(
        default=256,
        metadata={"help": "Number of rank-local json samples per grouped dataset to probe in geo startup self-check."}
    )
    geo_startup_selfcheck_strict: bool = field(
        default=True,
        metadata={"help": "If True, abort training when geo startup self-check finds missing/misaligned sample_id."}
    )
    geo_require_vae_no_dropout: bool = field(
        default=True,
        metadata={"help": "Require vae_cond_dropout_prob=0 when geo aux is enabled (recommended for stable mapping)."}
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Optional spatial distillation branch (kept separate from base model path).
    distill_feature_bank = None
    distill_adapter = None
    distill_optimizer = None
    distill_last_grad_norm = torch.tensor(0.0, device=device)
    if training_args.spatial_distill_enable:
        if not training_args.visual_gen:
            raise ValueError("spatial_distill requires visual_gen=True.")
        if training_args.spatial_distill_manifest is None:
            raise ValueError("Please set --spatial_distill_manifest when --spatial_distill_enable is True.")
        if training_args.spatial_distill_loss_type not in {"cosine", "mse"}:
            raise ValueError("--spatial_distill_loss_type must be one of: cosine, mse")

        distill_feature_bank = TeacherFeatureBank(training_args.spatial_distill_manifest)
        distill_adapter = SpatialDistillAdapter(
            student_dim=model.patch_latent_dim,
            teacher_dim=distill_feature_bank.feature_dim,
            hidden_dim=training_args.spatial_distill_hidden_dim,
        ).to(device)
        distill_adapter = DDP(
            distill_adapter,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        distill_optimizer = torch.optim.AdamW(
            distill_adapter.parameters(),
            lr=training_args.spatial_distill_lr,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.eps,
            weight_decay=0,
        )

        if resume_from is not None:
            distill_state_path = os.path.join(resume_from, "spatial_distill.pt")
            if os.path.exists(distill_state_path):
                state = torch.load(distill_state_path, map_location="cpu")
                if "adapter" in state:
                    distill_adapter.module.load_state_dict(state["adapter"], strict=True)
                if (not resume_model_only) and ("optimizer" in state) and state["optimizer"] is not None:
                    distill_optimizer.load_state_dict(state["optimizer"])
                logger.info(f"Loaded spatial distill state from: {distill_state_path}")
            else:
                logger.warning(f"Spatial distill is enabled, but no state found at: {distill_state_path}")

    pose_aux_delta_bank = None
    pose_probe_delta_bank = None
    pose_probe_buffer = None
    pose_probe_last_metrics = {}
    pose_aux_head = None
    pose_aux_optimizer = None
    pose_aux_last_grad_norm = torch.tensor(0.0, device=device)
    if training_args.pose_aux_enable:
        if not training_args.visual_gen:
            raise ValueError("pose_aux requires visual_gen=True.")
        if training_args.pose_aux_manifest is None:
            raise ValueError("Please set --pose_aux_manifest when --pose_aux_enable is True.")
        if training_args.pose_aux_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError("--pose_aux_loss_type must be one of: smooth_l1, mse")
        if training_args.pose_aux_trans_norm <= 0:
            raise ValueError("--pose_aux_trans_norm must be > 0")

        pose_aux_delta_bank = PoseDeltaBank(training_args.pose_aux_manifest)
        pose_aux_head = PoseAuxHead(
            student_dim=model.patch_latent_dim,
            hidden_dim=training_args.pose_aux_hidden_dim,
            out_dim=6,
        ).to(device)
        pose_aux_head = DDP(
            pose_aux_head,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        pose_aux_optimizer = torch.optim.AdamW(
            pose_aux_head.parameters(),
            lr=training_args.pose_aux_lr,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.eps,
            weight_decay=0,
        )

        if resume_from is not None:
            pose_state_path = os.path.join(resume_from, "pose_aux.pt")
            if os.path.exists(pose_state_path):
                state = torch.load(pose_state_path, map_location="cpu")
                if "head" in state:
                    pose_aux_head.module.load_state_dict(state["head"], strict=True)
                if (not resume_model_only) and ("optimizer" in state) and state["optimizer"] is not None:
                    pose_aux_optimizer.load_state_dict(state["optimizer"])
                logger.info(f"Loaded pose aux state from: {pose_state_path}")
            else:
                logger.warning(f"Pose aux is enabled, but no state found at: {pose_state_path}")

    if training_args.pose_probe_enable:
        if not training_args.visual_gen:
            raise ValueError("pose_probe requires visual_gen=True.")
        if training_args.pose_probe_every <= 0:
            raise ValueError("--pose_probe_every must be > 0")
        if training_args.pose_probe_min_samples <= 0:
            raise ValueError("--pose_probe_min_samples must be > 0")
        if training_args.pose_probe_max_samples <= 0:
            raise ValueError("--pose_probe_max_samples must be > 0")
        if training_args.pose_probe_ridge < 0:
            raise ValueError("--pose_probe_ridge must be >= 0")
        if not (0.0 < training_args.pose_probe_val_ratio < 1.0):
            raise ValueError("--pose_probe_val_ratio must be in (0, 1)")

        probe_manifest = training_args.pose_probe_manifest or training_args.pose_aux_manifest
        if probe_manifest is None:
            raise ValueError(
                "Please set --pose_probe_manifest, or provide --pose_aux_manifest for reuse."
            )
        pose_probe_delta_bank = PoseDeltaBank(probe_manifest)
        pose_probe_buffer = PoseProbeBuffer(max_samples=training_args.pose_probe_max_samples)
        logger.info(
            f"Pose probe enabled: manifest={probe_manifest}, "
            f"every={training_args.pose_probe_every}, "
            f"min_samples={training_args.pose_probe_min_samples}, "
            f"max_samples={training_args.pose_probe_max_samples}."
        )

    geo_aux_bank = None
    geo_pose_head = None
    geo_depth_head = None
    geo_aux_optimizer = None
    geo_aux_last_grad_norm = torch.tensor(0.0, device=device)
    if training_args.geo_aux_enable:
        if not training_args.visual_gen:
            raise ValueError("geo_aux requires visual_gen=True.")
        if training_args.geo_aux_manifest is None:
            raise ValueError("Please set --geo_aux_manifest when --geo_aux_enable is True.")
        if training_args.geo_aux_warmup_steps <= 0:
            raise ValueError("--geo_aux_warmup_steps must be > 0")
        if training_args.geo_aux_lr <= 0:
            raise ValueError("--geo_aux_lr must be > 0")
        if training_args.geo_depth_tol <= 0:
            raise ValueError("--geo_depth_tol must be > 0")
        if training_args.geo_reproj_max_samples < 0:
            raise ValueError("--geo_reproj_max_samples must be >= 0")
        if not (0.0 < training_args.geo_decode_scale <= 1.0):
            raise ValueError("--geo_decode_scale must be in (0, 1]")
        if training_args.geo_startup_selfcheck and training_args.geo_startup_selfcheck_samples <= 0:
            raise ValueError("--geo_startup_selfcheck_samples must be > 0 when geo_startup_selfcheck is enabled")
        # if training_args.geo_require_vae_no_dropout and model_args.vae_cond_dropout_prob > 0:
        #     raise ValueError(
        #         "geo_aux requires stable source/target image indexing. "
        #         "Please set --vae_cond_dropout_prob 0 (or set --geo_require_vae_no_dropout False)."
        #     )

        geo_aux_bank = GeoAuxBank(training_args.geo_aux_manifest)
        # Pose head consumes pooled source/generated latent features + their difference (3 * z_channels).
        geo_pose_in_dim = model.latent_channel * 3
        geo_head_dtype = torch.bfloat16 if training_args.geo_aux_head_bf16 else torch.float32
        geo_pose_head = GeoPoseHead(
            in_dim=geo_pose_in_dim,
            hidden_dim=training_args.geo_pose_hidden_dim,
            out_dim=6,
        ).to(device=device, dtype=geo_head_dtype)
        geo_depth_head = GeoDepthHead(
            in_dim=model.patch_latent_dim,
            hidden_dim=training_args.geo_depth_hidden_dim,
        ).to(device=device, dtype=geo_head_dtype)
        geo_pose_head = DDP(
            geo_pose_head,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        geo_depth_head = DDP(
            geo_depth_head,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        geo_aux_params = list(geo_pose_head.parameters()) + list(geo_depth_head.parameters())
        geo_aux_optimizer = torch.optim.AdamW(
            geo_aux_params,
            lr=training_args.geo_aux_lr,
            betas=(training_args.beta1, training_args.beta2),
            eps=training_args.eps,
            weight_decay=0,
        )

        if resume_from is not None:
            geo_state_path = os.path.join(resume_from, "geo_aux.pt")
            if os.path.exists(geo_state_path):
                state = torch.load(geo_state_path, map_location="cpu")
                if "pose_head" in state:
                    geo_pose_head.module.load_state_dict(state["pose_head"], strict=True)
                if "depth_head" in state:
                    geo_depth_head.module.load_state_dict(state["depth_head"], strict=True)
                if (not resume_model_only) and ("optimizer" in state) and state["optimizer"] is not None:
                    geo_aux_optimizer.load_state_dict(state["optimizer"])
                logger.info(f"Loaded geo aux state from: {geo_state_path}")
            else:
                logger.warning(f"Geo aux is enabled, but no state found at: {geo_state_path}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

        print("dataset_meta is",dataset_meta)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    if training_args.geo_aux_enable and training_args.geo_startup_selfcheck:
        _run_geo_aux_startup_selfcheck(
            train_dataset=train_dataset,
            geo_aux_bank=geo_aux_bank,
            logger=logger,
            rank=dist.get_rank(),
            max_samples_per_group=int(training_args.geo_startup_selfcheck_samples),
            strict=bool(training_args.geo_startup_selfcheck_strict),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()
    
    
    current_visualization_freq = 10000000

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    if distill_optimizer is not None:
        distill_optimizer.zero_grad()
    if pose_aux_optimizer is not None:
        pose_aux_optimizer.zero_grad()
    if geo_aux_optimizer is not None:
        geo_aux_optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    cumulative_samples = 0.0
    cumulative_tokens = 0.0

    dataset_name_order = [dataset.dataset_name for dataset in train_dataset.grouped_datasets]
    dataset_expected_local = {
        dataset.dataset_name: len(getattr(dataset, "data_paths_per_rank", []))
        for dataset in train_dataset.grouped_datasets
    }
    dataset_unique_seen = {name: set() for name in dataset_name_order}
    dataset_all_rank_pass_announced = set()

    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        cumulative_tokens += tokens_tensor.item()

        global_samples_tensor = torch.tensor(float(len(data['sample_lens'])), device=device)
        dist.all_reduce(global_samples_tensor, op=dist.ReduceOp.SUM)
        cumulative_samples += global_samples_tensor.item()

        if data_indexes is not None:
            for item in data_indexes:
                dataset_name = item['dataset_name']
                if dataset_name not in dataset_unique_seen:
                    continue
                data_index = _to_hashable_index(item['data_indexes'])
                worker_id = item['worker_id']
                dataset_unique_seen[dataset_name].add((worker_id, data_index))

        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            padded_images_for_geo = data.get("padded_images", None) if training_args.geo_aux_enable else None
            if training_args.visual_gen:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            data['return_images'] = (curr_step+1) % current_visualization_freq == 0
            data['return_mse_preds'] = bool(
                training_args.spatial_distill_enable
                or training_args.pose_aux_enable
                or training_args.pose_probe_enable
                or training_args.geo_aux_enable
            )
            data['return_mse_targets'] = bool(training_args.geo_aux_enable)
            try:
                loss_dict, images = fsdp_model(**data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e

        mse_preds = loss_dict.pop("mse_preds", None)
        mse_target_clean = loss_dict.pop("mse_target_clean", None)
        mse_target_noise = loss_dict.pop("mse_target_noise", None)
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        need_student_feats = bool(
            training_args.spatial_distill_enable
            or training_args.pose_aux_enable
            or training_args.pose_probe_enable
        )
        student_feats = None
        student_valid = None
        if need_student_feats:
            if mse_preds is None:
                raise RuntimeError(
                    "Auxiliary branch enabled but `mse_preds` was not returned by model."
                )
            if data_indexes is not None and len(data_indexes) > 0:
                student_feats, student_valid = pool_mse_preds_by_sample(
                    packed_mse_preds=mse_preds,
                    mse_loss_indexes=data['mse_loss_indexes'],
                    sample_lens=data['sample_lens'],
                )

        if training_args.spatial_distill_enable:
            distill_loss = torch.tensor(0.0, device=device)
            if student_feats is not None:
                teacher_feats, teacher_valid = distill_feature_bank.lookup_batch(
                    batch_data_indexes=data_indexes,
                    device=device,
                )
                valid_mask = student_valid & teacher_valid
                if valid_mask.any():
                    pred_teacher_feats = distill_adapter(student_feats[valid_mask].float())
                    distill_loss = compute_alignment_loss(
                        pred_teacher_features=pred_teacher_feats,
                        teacher_features=teacher_feats[valid_mask],
                        loss_type=training_args.spatial_distill_loss_type,
                    )

            loss_dict["distill"] = distill_loss.detach()
            loss = loss + distill_loss * training_args.spatial_distill_weight

        pose_aux_weight_current = 0.0
        if training_args.pose_aux_enable:
            pose_aux_loss = torch.tensor(0.0, device=device)
            if student_feats is not None:
                gt_pose_delta, pose_valid = pose_aux_delta_bank.lookup_batch(
                    batch_data_indexes=data_indexes,
                    device=device,
                )
                valid_mask = student_valid & pose_valid
                if valid_mask.any():
                    gt_pose_delta = gt_pose_delta.clone()
                    gt_pose_delta[:, :3] = gt_pose_delta[:, :3] / training_args.pose_aux_trans_norm
                    pred_pose_delta = pose_aux_head(student_feats.float())
                    pose_aux_loss = compute_pose_aux_loss(
                        pred_delta=pred_pose_delta,
                        gt_delta=gt_pose_delta,
                        valid_mask=valid_mask,
                        loss_type=training_args.pose_aux_loss_type,
                        trans_weight=training_args.pose_aux_trans_weight,
                        rot_weight=training_args.pose_aux_rot_weight,
                        smooth_l1_beta=training_args.pose_aux_smooth_l1_beta,
                    )

            warmup_steps = max(int(training_args.pose_aux_warmup_steps), 1)
            pose_aux_weight_current = training_args.pose_aux_weight * min(1.0, float(curr_step + 1) / warmup_steps)
            loss_dict["pose_aux"] = pose_aux_loss.detach()
            loss = loss + pose_aux_loss * pose_aux_weight_current

        if training_args.pose_probe_enable and student_feats is not None:
            gt_pose_delta_probe, pose_probe_valid = pose_probe_delta_bank.lookup_batch(
                batch_data_indexes=data_indexes,
                device=device,
            )
            probe_valid_mask = student_valid & pose_probe_valid
            if probe_valid_mask.any():
                pose_probe_buffer.add(
                    features=student_feats[probe_valid_mask],
                    targets=gt_pose_delta_probe[probe_valid_mask],
                )

        geo_alpha_current = 0.0
        geo_beta_current = 0.0
        geo_theta_current = 0.0
        if training_args.geo_aux_enable:
            if mse_preds is None or mse_target_clean is None or mse_target_noise is None:
                raise RuntimeError(
                    "geo_aux enabled but model did not return mse_preds/mse_target_clean/mse_target_noise."
                )
            if padded_images_for_geo is None:
                raise RuntimeError("geo_aux enabled but padded_images were not found in batch data.")

            geo_reproj_loss = torch.tensor(0.0, device=device)
            geo_pose_loss = torch.tensor(0.0, device=device)
            geo_depth_loss = torch.tensor(0.0, device=device)

            if data_indexes is not None and len(data_indexes) > 0:
                geo_batch = geo_aux_bank.lookup_batch(
                    batch_data_indexes=data_indexes,
                    device=device,
                )
                geo_valid = geo_batch["valid_mask"]

                src_img_indices, tgt_img_indices, tgt_hw_list = _build_geo_image_mapping(
                    num_context=geo_batch["num_context"],
                    source_idx=geo_batch["source_idx"],
                    patchified_vae_latent_shapes=data["patchified_vae_latent_shapes"],
                )

                if src_img_indices is not None and tgt_img_indices is not None:
                    pred_target_tokens = mse_target_clean + (mse_target_noise - mse_preds)
                    expected_tokens = sum(int(h) * int(w) for h, w in tgt_hw_list)
                    if pred_target_tokens.shape[0] == expected_tokens:
                        pred_target_token_list = split_tokens_by_hw(pred_target_tokens, tgt_hw_list)
                        downsample_factor = model_args.latent_patch_size * vae_config.downsample
                        depth_hw = geo_batch["source_depth"].shape[-2:]
                        vae_decode_dtype = next(vae_model.parameters()).dtype

                        reproj_items = []
                        pose_items = []
                        depth_items = []
                        valid_geo_indices = [b for b in range(len(tgt_hw_list)) if bool(geo_valid[b])]
                        reproj_indices = set()
                        max_reproj = int(training_args.geo_reproj_max_samples)
                        do_reproj = max_reproj > 0 and training_args.geo_aux_weight > 0
                        do_pose = training_args.geo_pose_weight > 0
                        do_depth = training_args.geo_depth_weight > 0
                        if do_reproj and len(valid_geo_indices) > 0:
                            if len(valid_geo_indices) <= max_reproj:
                                reproj_indices = set(valid_geo_indices)
                            else:
                                sampled = torch.randperm(
                                    len(valid_geo_indices),
                                    device=device,
                                )[:max_reproj].tolist()
                                reproj_indices = {valid_geo_indices[i] for i in sampled}
                        for b in range(len(tgt_hw_list)):
                            if not bool(geo_valid[b]):
                                continue

                            src_global_idx = int(src_img_indices[b])
                            src_h_tok, src_w_tok = data["patchified_vae_latent_shapes"][src_global_idx]
                            src_h = int(src_h_tok) * downsample_factor
                            src_w = int(src_w_tok) * downsample_factor
                            src_rgb = padded_images_for_geo[src_global_idx, :, :src_h, :src_w]

                            tgt_h_tok, tgt_w_tok = tgt_hw_list[b]
                            pred_tokens = pred_target_token_list[b]
                            pred_latent = unpatchify_latent_tokens(
                                tokens=pred_tokens,
                                h=tgt_h_tok,
                                w=tgt_w_tok,
                                latent_patch_size=model_args.latent_patch_size,
                                latent_channel=model.patch_latent_dim // (model_args.latent_patch_size ** 2),
                            )

                            # (b) reprojection consistency: compare generated target with reprojected source.
                            if do_reproj and b in reproj_indices:
                                decode_latent = pred_latent.unsqueeze(0)
                                if training_args.geo_decode_scale < 1.0:
                                    lat_h = decode_latent.shape[-2]
                                    lat_w = decode_latent.shape[-1]
                                    dec_h = max(1, int(round(lat_h * training_args.geo_decode_scale)))
                                    dec_w = max(1, int(round(lat_w * training_args.geo_decode_scale)))
                                    if dec_h != lat_h or dec_w != lat_w:
                                        decode_latent = F.interpolate(
                                            decode_latent,
                                            size=(dec_h, dec_w),
                                            mode="bilinear",
                                            align_corners=False,
                                        )
                                if decode_latent.dtype != vae_decode_dtype:
                                    decode_latent = decode_latent.to(dtype=vae_decode_dtype)
                                gen_rgb = vae_model.decode(decode_latent)[0]
                                src_rgb_rs = F.interpolate(
                                    src_rgb.unsqueeze(0),
                                    size=depth_hw,
                                    mode="bilinear",
                                    align_corners=False,
                                )[0]
                                gen_rgb_rs = F.interpolate(
                                    gen_rgb.unsqueeze(0),
                                    size=depth_hw,
                                    mode="bilinear",
                                    align_corners=False,
                                )[0]
                                reproj_i, valid_mask_i = reprojection_consistency_loss(
                                    generated_rgb=gen_rgb_rs,
                                    source_rgb=src_rgb_rs,
                                    source_depth=geo_batch["source_depth"][b],
                                    target_depth=geo_batch["target_depth"][b],
                                    source_k=geo_batch["source_k"][b],
                                    target_k=geo_batch["target_k"][b],
                                    source_pose_c2w=geo_batch["source_pose"][b],
                                    target_pose_c2w=geo_batch["target_pose"][b],
                                    ssim_weight=training_args.geo_ssim_weight,
                                    depth_tol=training_args.geo_depth_tol,
                                )
                                if valid_mask_i.any():
                                    reproj_items.append(reproj_i)

                            # (a) SE(3) pose regression from (I_source, I_generated).
                            if do_pose:
                                with torch.no_grad():
                                    src_latent = vae_model.encode(src_rgb.unsqueeze(0))
                                src_pool = src_latent.mean(dim=(2, 3))
                                gen_pool = pred_latent.unsqueeze(0).mean(dim=(2, 3))
                                pose_feat = torch.cat([src_pool, gen_pool, gen_pool - src_pool], dim=-1)
                                pose_head_dtype = next(geo_pose_head.parameters()).dtype
                                pred_delta = geo_pose_head(pose_feat.to(dtype=pose_head_dtype))
                                gt_delta = geo_batch["pose_delta"][b : b + 1]
                                pose_items.append(
                                    se3_l1_pose_loss(
                                        pred_delta=pred_delta,
                                        gt_delta=gt_delta,
                                        valid_mask=torch.ones((1,), device=device, dtype=torch.bool),
                                    )
                                )

                            # (c) target depth multi-task from generated latent tokens.
                            if do_depth:
                                depth_head_dtype = next(geo_depth_head.parameters()).dtype
                                pred_depth_tokens = geo_depth_head(pred_tokens.to(dtype=depth_head_dtype))
                                pred_depth_hw = F.softplus(
                                    pred_depth_tokens.reshape(int(tgt_h_tok), int(tgt_w_tok))
                                ) + 1e-6
                                gt_depth_hw = F.interpolate(
                                    geo_batch["target_depth"][b].unsqueeze(0).unsqueeze(0),
                                    size=(int(tgt_h_tok), int(tgt_w_tok)),
                                    mode="nearest",
                                ).squeeze(0).squeeze(0)
                                depth_valid = torch.isfinite(gt_depth_hw) & (gt_depth_hw > 0)
                                depth_items.append(
                                    scale_invariant_log_depth_loss(
                                        pred_depth=pred_depth_hw,
                                        gt_depth=gt_depth_hw,
                                        valid_mask=depth_valid,
                                        si_lambda=training_args.geo_depth_si_lambda,
                                    )
                                )

                        if len(reproj_items) > 0:
                            geo_reproj_loss = torch.stack(reproj_items).mean()
                        if len(pose_items) > 0:
                            geo_pose_loss = torch.stack(pose_items).mean()
                        if len(depth_items) > 0:
                            geo_depth_loss = torch.stack(depth_items).mean()

            warmup_steps = max(int(training_args.geo_aux_warmup_steps), 1)
            warm_factor = min(1.0, float(curr_step + 1) / warmup_steps)
            geo_alpha_current = training_args.geo_aux_weight * warm_factor
            geo_beta_current = training_args.geo_pose_weight * warm_factor
            geo_theta_current = training_args.geo_depth_weight * warm_factor

            loss_dict["geo_reproj"] = geo_reproj_loss.detach()
            loss_dict["geo_pose"] = geo_pose_loss.detach()
            loss_dict["geo_depth"] = geo_depth_loss.detach()
            loss = (
                loss
                + geo_reproj_loss * geo_alpha_current
                + geo_pose_loss * geo_beta_current
                + geo_depth_loss * geo_theta_current
            )

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            if distill_optimizer is not None:
                distill_last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    distill_adapter.parameters(),
                    training_args.max_grad_norm,
                )
            if pose_aux_optimizer is not None:
                pose_aux_last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    pose_aux_head.parameters(),
                    training_args.max_grad_norm,
                )
            if geo_aux_optimizer is not None:
                geo_aux_last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(geo_pose_head.parameters()) + list(geo_depth_head.parameters()),
                    training_args.max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
            if distill_optimizer is not None:
                distill_optimizer.step()
                distill_optimizer.zero_grad()
            if pose_aux_optimizer is not None:
                pose_aux_optimizer.step()
                pose_aux_optimizer.zero_grad()
            if geo_aux_optimizer is not None:
                geo_aux_optimizer.step()
                geo_aux_optimizer.zero_grad()

            if (
                training_args.pose_probe_enable
                and ((curr_step + 1) % training_args.pose_probe_every == 0)
            ):
                local_probe = pose_probe_buffer.get_tensors()
                if dist.get_rank() == 0:
                    gathered_probe = [None] * dist.get_world_size()
                else:
                    gathered_probe = None
                dist.gather_object(local_probe, gathered_probe, dst=0)

                if dist.get_rank() == 0:
                    probe_features = []
                    probe_targets = []
                    for item in gathered_probe:
                        if item is None:
                            continue
                        feats_i, targets_i = item
                        if feats_i is None or targets_i is None:
                            continue
                        if feats_i.numel() == 0 or targets_i.numel() == 0:
                            continue
                        probe_features.append(feats_i)
                        probe_targets.append(targets_i)

                    total_probe_samples = int(sum(t.shape[0] for t in probe_targets))
                    if total_probe_samples >= training_args.pose_probe_min_samples:
                        merged_features = torch.cat(probe_features, dim=0)
                        merged_targets = torch.cat(probe_targets, dim=0)
                        probe_metrics = evaluate_pose_linear_probe(
                            features=merged_features,
                            pose_delta=merged_targets,
                            val_ratio=training_args.pose_probe_val_ratio,
                            ridge=training_args.pose_probe_ridge,
                            seed=training_args.pose_probe_seed + curr_step,
                        )
                        pose_probe_last_metrics = probe_metrics
                        logger.info(
                            "[PoseProbe] "
                            f"step={curr_step}, n={int(probe_metrics.get('probe_num_samples', 0))}, "
                            f"val_rot_deg={probe_metrics.get('probe_val_rot_geodesic_deg', float('nan')):.4f}, "
                            f"val_trans_l2={probe_metrics.get('probe_val_trans_l2', float('nan')):.4f}, "
                            f"val_success={probe_metrics.get('probe_val_success', float('nan')):.4f}"
                        )
                        wandb.log(
                            {f"pose_probe/{k}": v for k, v in probe_metrics.items()},
                            step=curr_step,
                        )
                    else:
                        logger.info(
                            f"[PoseProbe] step={curr_step}, skip: "
                            f"global buffered samples={total_probe_samples} < pose_probe_min_samples="
                            f"{training_args.pose_probe_min_samples}."
                        )
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            message += f"CumSamples: {int(cumulative_samples)}, CumTokens: {int(cumulative_tokens)}, "

            coverage_chunks = []
            for dataset_name in dataset_name_order:
                expected_local = dataset_expected_local.get(dataset_name, 0)
                seen_local = len(dataset_unique_seen.get(dataset_name, []))
                if expected_local > 0:
                    coverage_ratio = seen_local / expected_local
                else:
                    coverage_ratio = 0.0
                coverage_chunks.append(
                    f"{dataset_name}:{seen_local}/{expected_local}({coverage_ratio*100:.1f}%)"
                )
                wandb_log[f"coverage_local_{dataset_name}"] = coverage_ratio

                if expected_local > 0 and dataset_name not in dataset_all_rank_pass_announced:
                    local_done = torch.tensor(
                        1 if seen_local >= expected_local else 0, device=device
                    )
                    dist.all_reduce(local_done, op=dist.ReduceOp.MIN)
                    if local_done.item() == 1:
                        dataset_all_rank_pass_announced.add(dataset_name)
                        if dist.get_rank() == 0:
                            logger.info(
                                f"[DataProgress] dataset={dataset_name} "
                                f"completed >=1 full local-shard pass on all ranks at step={curr_step}."
                            )

            if len(coverage_chunks) > 0:
                message += "LocalCoverage: " + " | ".join(coverage_chunks) + ", "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            if distill_optimizer is not None:
                wandb_log['distill_adapter_grad_norm'] = distill_last_grad_norm.item()
                wandb_log['distill_lr'] = distill_optimizer.param_groups[0]['lr']
            if pose_aux_optimizer is not None:
                wandb_log['pose_aux_head_grad_norm'] = pose_aux_last_grad_norm.item()
                wandb_log['pose_aux_lr'] = pose_aux_optimizer.param_groups[0]['lr']
                wandb_log['pose_aux_weight'] = pose_aux_weight_current
            if geo_aux_optimizer is not None:
                wandb_log['geo_aux_grad_norm'] = geo_aux_last_grad_norm.item()
                wandb_log['geo_aux_lr'] = geo_aux_optimizer.param_groups[0]['lr']
                wandb_log['geo_alpha'] = geo_alpha_current
                wandb_log['geo_beta'] = geo_beta_current
                wandb_log['geo_theta'] = geo_theta_current
            if training_args.pose_probe_enable and pose_probe_buffer is not None:
                wandb_log['pose_probe_buffer_local'] = float(pose_probe_buffer.num_samples)
                if dist.get_rank() == 0 and len(pose_probe_last_metrics) > 0:
                    for key, value in pose_probe_last_metrics.items():
                        wandb_log[f'pose_probe_last/{key}'] = value
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value
            wandb_log['cum_samples'] = cumulative_samples
            wandb_log['cum_tokens'] = cumulative_tokens

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if (curr_step+1) % current_visualization_freq == 0 and dist.get_rank() == 0: 

            if images is not None:
                with torch.no_grad():
                    draw_images = vae_model.decode(images.float())
                save_images_png(
                    draw_images,           # 只存前 4 张
                    save_dir="vis/",
                    prefix=f"step_{curr_step}",
                )
            
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            if dist.get_rank() == 0 and distill_adapter is not None:
                distill_state = {
                    "adapter": distill_adapter.module.state_dict(),
                    "optimizer": distill_optimizer.state_dict() if distill_optimizer is not None else None,
                }
                distill_state_path = os.path.join(
                    training_args.checkpoint_dir,
                    f"{curr_step:07d}",
                    "spatial_distill.pt",
                )
                torch.save(distill_state, distill_state_path)
            if dist.get_rank() == 0 and pose_aux_head is not None:
                pose_state = {
                    "head": pose_aux_head.module.state_dict(),
                    "optimizer": pose_aux_optimizer.state_dict() if pose_aux_optimizer is not None else None,
                }
                pose_state_path = os.path.join(
                    training_args.checkpoint_dir,
                    f"{curr_step:07d}",
                    "pose_aux.pt",
                )
                torch.save(pose_state, pose_state_path)
            if dist.get_rank() == 0 and geo_pose_head is not None and geo_depth_head is not None:
                geo_state = {
                    "pose_head": geo_pose_head.module.state_dict(),
                    "depth_head": geo_depth_head.module.state_dict(),
                    "optimizer": geo_aux_optimizer.state_dict() if geo_aux_optimizer is not None else None,
                }
                geo_state_path = os.path.join(
                    training_args.checkpoint_dir,
                    f"{curr_step:07d}",
                    "geo_aux.pt",
                )
                torch.save(geo_state, geo_state_path)
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list
        )
        if dist.get_rank() == 0 and distill_adapter is not None:
            distill_state = {
                "adapter": distill_adapter.module.state_dict(),
                "optimizer": distill_optimizer.state_dict() if distill_optimizer is not None else None,
            }
            distill_state_path = os.path.join(
                training_args.checkpoint_dir,
                f"{curr_step:07d}",
                "spatial_distill.pt",
            )
            torch.save(distill_state, distill_state_path)
        if dist.get_rank() == 0 and pose_aux_head is not None:
            pose_state = {
                "head": pose_aux_head.module.state_dict(),
                "optimizer": pose_aux_optimizer.state_dict() if pose_aux_optimizer is not None else None,
            }
            pose_state_path = os.path.join(
                training_args.checkpoint_dir,
                f"{curr_step:07d}",
                "pose_aux.pt",
            )
            torch.save(pose_state, pose_state_path)
        if dist.get_rank() == 0 and geo_pose_head is not None and geo_depth_head is not None:
            geo_state = {
                "pose_head": geo_pose_head.module.state_dict(),
                "depth_head": geo_depth_head.module.state_dict(),
                "optimizer": geo_aux_optimizer.state_dict() if geo_aux_optimizer is not None else None,
            }
            geo_state_path = os.path.join(
                training_args.checkpoint_dir,
                f"{curr_step:07d}",
                "geo_aux.pt",
            )
            torch.save(geo_state, geo_state_path)
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
