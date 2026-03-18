# Pose Auxiliary Loss

This module adds a pose-delta supervision branch to improve spatial geometry following.

## 1) Prepare pose bank

```bash
python train/pose_aux/precompute_pose_delta_bank.py \
  --dataset_config_file data/configs/step_mix_refine_onlyone.yaml \
  --output_dir /path/to/pose_bank_step_mix_refine_onlyone
```

Expected pose fields in each JSONL row (supports multiple formats):

- Preferred:
  - `start_image_id` or `source_idx` (0/1)
  - `context_poses` (list of 4x4 c2w)
  - `target_pose` (4x4 c2w)
- Also supported:
  - `source_pose` + `target_pose`
  - `delta_pose_se3` (6D)
  - `delta_pose` (6D or 4x4)

## 2) Enable training

Add to your existing train command:

```bash
--pose_aux_enable True \
--pose_aux_manifest /path/to/pose_bank_step_mix_refine_onlyone/manifest.json \
--pose_aux_weight 0.05 \
--pose_aux_warmup_steps 3000 \
--pose_aux_loss_type smooth_l1 \
--pose_aux_hidden_dim 1024 \
--pose_aux_lr 1e-4 \
--pose_aux_trans_norm 1.0 \
--pose_aux_trans_weight 1.0 \
--pose_aux_rot_weight 1.0
```

Notes:

- Pose aux is **off by default**.
- When enabled, a separate state file `pose_aux.pt` is saved per checkpoint.

## 3) Enable periodic pose probe (diagnostics)

The probe is not a new generation model. It is a light-weight diagnostic:

- freeze current student features (from `mse_preds`)
- fit a closed-form linear regressor to predict pose delta
- monitor validation metrics as an indirect signal of geometry learning

Add to your train command:

```bash
--pose_probe_enable True \
--pose_probe_manifest /path/to/pose_bank_step_mix_refine_onlyone/manifest.json \
--pose_probe_every 1000 \
--pose_probe_min_samples 512 \
--pose_probe_max_samples 4096 \
--pose_probe_val_ratio 0.2 \
--pose_probe_ridge 1e-4 \
--pose_probe_seed 3407
```

If `--pose_probe_manifest` is omitted, training will reuse `--pose_aux_manifest`.

Main logged metrics:

- `pose_probe/probe_val_rot_geodesic_deg`
- `pose_probe/probe_val_trans_l2`
- `pose_probe/probe_val_success`

## 4) Pose Matrix Conditioning Pipeline (training + inference)

This branch also supports directly feeding pose matrices into prompt text, so you can test:

- language-only instruction following
- instruction + pose matrix
- pose matrix only

### 4.1 Prepare pose-augmented JSONL

```bash
bash scripts/prepare_complex_2context_pose_jsonl.sh
```

This script calls:

```bash
python train/pose_aux/augment_pose_jsonl.py \
  --input_jsonl /path/to/input.jsonl \
  --output_jsonl /path/to/output_pose.jsonl \
  --set_start_image_id infer \
  --require_valid_pose
```

Each valid row gets:

- `context_poses` (list of 4x4)
- `target_pose` (4x4)
- `start_image_id` (source view index)

### 4.2 Train with pose matrix conditioning

Use one of:

- `data/configs/complex_2context_0310_3data_pose_append.yaml` (instruction + pose)
- `data/configs/complex_2context_0310_3data_pose_only.yaml` (pose-only)

Example:

```bash
POSE_CONDITION_MODE=only bash scripts/complex_2context_0310_3data_pose_condition.sh
```

### 4.3 Run inference with pose matrix conditioning

```bash
MODE=only bash scripts/batch_evaluation_pose_condition.sh
```

Or call inference directly:

```bash
python batch_infer.py \
  --eval-json /path/to/test_pose.jsonl \
  --image-root /path/to/images \
  --out-dir /path/to/out \
  --inject_pose_text \
  --pose_text_replace_instruction \
  --pose_text_require_valid
```

`bagel_inference.py` supports the same pose-text flags for understanding evaluation.
