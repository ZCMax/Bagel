# Spatial Distillation (VGGT -> BAGEL)

This module adds an optional auxiliary loss that aligns BAGEL's predicted latent features with teacher features precomputed from VGGT.

## 1) Precompute teacher feature bank

```bash
python train/spatial_distill/precompute_vggt_features.py \
  --dataset_config_file data/configs/step_mix_refine_onlyone.yaml \
  --output_dir /path/to/vggt_bank_step_mix_refine_onlyone \
  --vggt_model_path /path/to/VGGT-1B \
  --device cuda \
  --dtype bfloat16
```

This writes:

- `manifest.json`
- `<group_name>.features.pt`
- `<group_name>.valid.pt`

## 2) Enable distillation in training

Use your existing training command and add:

```bash
--spatial_distill_enable True \
--spatial_distill_manifest /path/to/vggt_bank_step_mix_refine_onlyone/manifest.json \
--spatial_distill_weight 0.05 \
--spatial_distill_loss_type cosine \
--spatial_distill_hidden_dim 0 \
--spatial_distill_lr 1e-4
```

Notes:

- Distillation is **off by default**.
- Existing training logic is unchanged when distillation is off.
- Adapter state is saved in checkpoint folders as `spatial_distill.pt`.
