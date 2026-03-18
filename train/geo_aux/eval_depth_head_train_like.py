import argparse
import json
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repo-root imports work when launching via absolute script path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.geo_aux import GeoAuxBank, GeoDepthHead, split_tokens_by_hw


def _str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


def _resolve_checkpoint(model_path: str) -> str:
    cands = [
        os.path.join(model_path, "ema_merged.safetensors"),
        os.path.join(model_path, "ema.safetensors"),
        os.path.join(model_path, "model.safetensors"),
    ]
    for path in cands:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find model checkpoint under {model_path}. "
        "Expected one of: ema_merged.safetensors / ema.safetensors / model.safetensors"
    )


def _resolve_geo_aux_ckpt(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.path.isdir(abs_path):
        cand = os.path.join(abs_path, "geo_aux.pt")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"No geo_aux.pt under directory: {abs_path}")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"geo_aux checkpoint not found: {abs_path}")
    return abs_path


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


def _infer_depth_head_dims(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "net.0.weight" in sd:
        hidden_dim, in_dim = sd["net.0.weight"].shape
        out_dim, hidden_dim_2 = sd["net.2.weight"].shape
        if int(out_dim) != 1:
            raise ValueError(f"Depth head output dim should be 1, got {out_dim}")
        if hidden_dim != hidden_dim_2:
            raise ValueError("Depth head hidden dim mismatch in checkpoint.")
        return int(in_dim), int(hidden_dim)
    out_dim, in_dim = sd["net.weight"].shape
    if int(out_dim) != 1:
        raise ValueError(f"Depth head output dim should be 1, got {out_dim}")
    return int(in_dim), 0


def _load_depth_head(geo_aux_ckpt: str, device: torch.device):
    state = torch.load(geo_aux_ckpt, map_location="cpu")
    if "depth_head" not in state:
        raise KeyError("geo_aux checkpoint must contain `depth_head`")
    depth_sd = state["depth_head"]
    depth_in_dim, depth_hidden_dim = _infer_depth_head_dims(depth_sd)
    depth_head = GeoDepthHead(in_dim=depth_in_dim, hidden_dim=depth_hidden_dim)
    depth_head.load_state_dict(depth_sd, strict=True)
    depth_head = depth_head.to(device=device)
    depth_head.eval()
    return depth_head


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _silog(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, si_lambda: float = 0.5) -> float:
    mask = valid & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
    if mask.sum().item() == 0:
        return float("nan")
    d = torch.log(pred[mask] + 1e-6) - torch.log(gt[mask] + 1e-6)
    return float(((d * d).mean() - si_lambda * (d.mean() * d.mean())).item())


def _depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, si_lambda: float = 0.5) -> Dict[str, float]:
    mask = valid & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
    if mask.sum().item() == 0:
        return {"num_valid": 0.0}

    pred_v = pred[mask]
    gt_v = gt[mask]

    abs_rel_raw = (pred_v - gt_v).abs().div(gt_v).mean()
    rmse_raw = torch.sqrt(((pred_v - gt_v) ** 2).mean())

    scale = torch.median(gt_v) / torch.clamp(torch.median(pred_v), min=1e-6)
    pred_s = pred_v * scale

    abs_rel_scaled = (pred_s - gt_v).abs().div(gt_v).mean()
    rmse_scaled = torch.sqrt(((pred_s - gt_v) ** 2).mean())

    ratio = torch.maximum(pred_s / torch.clamp(gt_v, min=1e-6), gt_v / torch.clamp(pred_s, min=1e-6))
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < (1.25 ** 2)).float().mean()
    delta3 = (ratio < (1.25 ** 3)).float().mean()

    pred_scaled_full = pred * scale
    mask_np = mask.detach().cpu().numpy().astype(bool)
    pred_np = pred_scaled_full.detach().cpu().numpy()[mask_np]
    gt_np = gt.detach().cpu().numpy()[mask_np]
    if pred_np.size >= 2:
        pearson = float(np.corrcoef(pred_np, gt_np)[0, 1])
    else:
        pearson = float("nan")

    return {
        "num_valid": float(mask.sum().item()),
        "scale_median": float(scale.item()),
        "silog_raw": _silog(pred, gt, mask, si_lambda=si_lambda),
        "silog_scaled": _silog(pred * scale, gt, mask, si_lambda=si_lambda),
        "abs_rel_raw": float(abs_rel_raw.item()),
        "rmse_raw": float(rmse_raw.item()),
        "abs_rel_scaled": float(abs_rel_scaled.item()),
        "rmse_scaled": float(rmse_scaled.item()),
        "delta1_scaled": float(delta1.item()),
        "delta2_scaled": float(delta2.item()),
        "delta3_scaled": float(delta3.item()),
        "pearson_scaled": pearson,
    }


class MetricAggregator:
    def __init__(self):
        self.sample_metrics = defaultdict(list)
        self.weighted_sum = defaultdict(float)
        self.weighted_count = 0.0
        self.num_samples = 0

    def add(self, metrics: Dict[str, float]):
        n = float(metrics.get("num_valid", 0.0))
        if n <= 0:
            return
        self.num_samples += 1
        self.weighted_count += n
        for k, v in metrics.items():
            if k == "num_valid":
                continue
            if not np.isfinite(v):
                continue
            self.sample_metrics[k].append(float(v))
            self.weighted_sum[k] += float(v) * n

    def summary(self) -> Dict[str, float]:
        out = {
            "num_samples": float(self.num_samples),
            "total_valid_pixels": float(self.weighted_count),
        }
        if self.num_samples == 0:
            return out
        for k, vals in self.sample_metrics.items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                continue
            out[f"{k}_mean"] = float(arr.mean())
            out[f"{k}_median"] = float(np.median(arr))
            if self.weighted_count > 0:
                out[f"{k}_weighted_mean"] = float(self.weighted_sum[k] / self.weighted_count)
        return out


def _load_model_and_tokenizer(args, device: torch.device):
    from data.data_utils import add_special_tokens
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.layer_module = args.layer_module
    llm_config.qk_norm = args.llm_qk_norm
    llm_config.tie_word_embeddings = args.tie_word_embeddings

    language_model = Qwen2ForCausalLM(llm_config)
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config,
        vit_config=None,
        vae_config=vae_config,
        latent_patch_size=args.latent_patch_size,
        max_latent_size=args.max_latent_size,
        connector_act=args.connector_act,
        interpolate_pos=args.interpolate_pos,
        timestep_shift=args.timestep_shift,
    )
    model = Bagel(language_model, None, config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model_ckpt = _resolve_checkpoint(args.model_path)
    state = load_file(model_ckpt, device="cpu")
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    model_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.model_dtype]

    model = model.to(device=device, dtype=model_dtype).eval()
    vae_model = vae_model.to(device=device).eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in vae_model.parameters():
        p.requires_grad = False

    print(f"[LoadModel] checkpoint={model_ckpt}")
    print(
        f"[LoadModel] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}, "
        f"model_dtype={model_dtype}, vae_dtype={next(vae_model.parameters()).dtype}"
    )
    if len(missing) > 0:
        print(f"[LoadModel] first missing keys: {missing[:8]}")
    if len(unexpected) > 0:
        print(f"[LoadModel] first unexpected keys: {unexpected[:8]}")

    return model, vae_model, vae_config, tokenizer, new_token_ids


def _build_eval_loader(args, tokenizer, new_token_ids, vae_config):
    from data.dataset_base import DataConfig, PackedDataset, collate_wrapper

    with open(args.dataset_config_file, "r", encoding="utf-8") as f:
        dataset_meta = yaml.safe_load(f)

    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    vae_image_downsample = args.latent_patch_size * vae_config.downsample
    dataset_config.vae_image_downsample = vae_image_downsample
    dataset_config.max_latent_size = args.max_latent_size
    if args.disable_dropout:
        dataset_config.text_cond_dropout_prob = 0.0
        dataset_config.vit_cond_dropout_prob = 0.0
        dataset_config.vae_cond_dropout_prob = 0.0
    else:
        dataset_config.text_cond_dropout_prob = args.text_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = args.vit_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = args.vae_cond_dropout_prob

    eval_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=0,
        world_size=1,
        num_workers=args.num_workers,
        expected_num_tokens=args.expected_num_tokens,
        max_num_tokens_per_sample=args.max_num_tokens_per_sample,
        max_num_tokens=args.max_num_tokens,
        max_buffer_size=args.max_buffer_size,
        prefer_buffer_before=args.prefer_buffer_before,
        interpolate_pos=args.interpolate_pos,
        use_flex=args.use_flex,
        data_status=None,
    )
    eval_dataset.set_epoch(args.data_seed)

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "collate_fn": collate_wrapper(),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    eval_loader = DataLoader(eval_dataset, **loader_kwargs)
    return eval_loader


def _prepare_forward_data(batch, device: torch.device):
    if device.type == "cuda":
        data = batch.cuda(device.index if device.index is not None else 0).to_dict()
    else:
        data = batch.to_dict()
    data_indexes = data.pop("batch_data_indexes", None)
    data.pop("ce_loss_weights", None)
    return data, data_indexes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate geo depth head in train-like token path (without modifying training code)."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to BAGEL model dir.")
    parser.add_argument("--geo_aux_ckpt", type=str, required=True, help="Path to geo_aux.pt or its parent dir.")
    parser.add_argument("--geo_aux_manifest", type=str, required=True, help="Path to geo bank manifest.json.")
    parser.add_argument("--dataset_config_file", type=str, required=True, help="Training dataset yaml.")

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--model_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--autocast", type=_str2bool, default=True, help="Enable autocast during model forward.")

    parser.add_argument("--layer_module", type=str, default="Qwen2MoTDecoderLayer")
    parser.add_argument("--llm_qk_norm", type=_str2bool, default=True)
    parser.add_argument("--tie_word_embeddings", type=_str2bool, default=False)
    parser.add_argument("--latent_patch_size", type=int, default=2)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--connector_act", type=str, default="gelu_pytorch_tanh")
    parser.add_argument("--interpolate_pos", type=_str2bool, default=False)
    parser.add_argument("--timestep_shift", type=float, default=1.0)
    parser.add_argument(
        "--token_formula",
        type=str,
        default="train",
        choices=["train", "plus"],
        help="`train`: clean + (noise - pred), `plus`: clean + pred",
    )

    parser.add_argument("--max_samples", type=int, default=256, help="Max valid geo samples to evaluate.")
    parser.add_argument("--max_batches", type=int, default=-1, help="Optional cap on iterated packed batches.")
    parser.add_argument("--si_lambda", type=float, default=0.5, help="SILog lambda.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output json path.")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--expected_num_tokens", type=int, default=13240)
    parser.add_argument("--max_num_tokens_per_sample", type=int, default=13240)
    parser.add_argument("--max_num_tokens", type=int, default=26520)
    parser.add_argument("--prefer_buffer_before", type=int, default=16384)
    parser.add_argument("--max_buffer_size", type=int, default=50)
    parser.add_argument("--use_flex", type=_str2bool, default=False)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--global_seed", type=int, default=4396)

    parser.add_argument("--disable_dropout", type=_str2bool, default=True)
    parser.add_argument("--text_cond_dropout_prob", type=float, default=0.1)
    parser.add_argument("--vit_cond_dropout_prob", type=float, default=0.3)
    parser.add_argument("--vae_cond_dropout_prob", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    _set_seed(args.global_seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] cuda requested but unavailable, fallback to cpu.")
        args.device = "cpu"
    device = torch.device(args.device)

    geo_aux_ckpt = _resolve_geo_aux_ckpt(args.geo_aux_ckpt)
    depth_head = _load_depth_head(geo_aux_ckpt, device=device)
    depth_head_dtype = next(depth_head.parameters()).dtype

    model, vae_model, vae_config, tokenizer, new_token_ids = _load_model_and_tokenizer(args, device=device)
    geo_bank = GeoAuxBank(args.geo_aux_manifest)
    eval_loader = _build_eval_loader(args, tokenizer, new_token_ids, vae_config)

    model_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.model_dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=args.autocast, dtype=model_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    pred_agg = MetricAggregator()
    clean_agg = MetricAggregator()
    const_agg = MetricAggregator()

    skipped = defaultdict(int)
    evaluated = 0
    iter_batches = 0
    pbar = tqdm(total=args.max_samples, desc="[DepthEval]", dynamic_ncols=True)

    for batch in eval_loader:
        if args.max_batches > 0 and iter_batches >= args.max_batches:
            break
        iter_batches += 1
        if evaluated >= args.max_samples:
            break

        data, data_indexes = _prepare_forward_data(batch, device=device)
        if data_indexes is None or len(data_indexes) == 0:
            skipped["empty_data_indexes"] += 1
            continue
        if "padded_images" not in data:
            skipped["no_padded_images"] += 1
            continue

        geo_batch = geo_bank.lookup_batch(batch_data_indexes=data_indexes, device=device)
        geo_valid = geo_batch["valid_mask"]

        src_img_indices, tgt_img_indices, tgt_hw_list = _build_geo_image_mapping(
            num_context=geo_batch["num_context"],
            source_idx=geo_batch["source_idx"],
            patchified_vae_latent_shapes=data["patchified_vae_latent_shapes"],
        )
        if src_img_indices is None or tgt_img_indices is None:
            skipped["mapping_invalid"] += 1
            continue

        with torch.no_grad():
            with autocast_ctx:
                padded_images = data.pop("padded_images")
                data["padded_latent"] = vae_model.encode(padded_images)
                data["return_images"] = False
                data["return_mse_preds"] = True
                data["return_mse_targets"] = True
                loss_dict, _ = model(**data)

        mse_preds = loss_dict.get("mse_preds", None)
        mse_target_clean = loss_dict.get("mse_target_clean", None)
        mse_target_noise = loss_dict.get("mse_target_noise", None)
        if mse_preds is None or mse_target_clean is None or mse_target_noise is None:
            skipped["missing_mse_tensors"] += 1
            continue

        expected_tokens = sum(int(h) * int(w) for h, w in tgt_hw_list)
        if (
            mse_preds.shape[0] != expected_tokens
            or mse_target_clean.shape[0] != expected_tokens
            or mse_target_noise.shape[0] != expected_tokens
        ):
            skipped["token_shape_mismatch"] += 1
            continue

        if args.token_formula == "train":
            pred_target_tokens = mse_target_clean + (mse_target_noise - mse_preds)
        else:
            pred_target_tokens = mse_target_clean + mse_preds

        pred_target_token_list = split_tokens_by_hw(pred_target_tokens, tgt_hw_list)
        clean_target_token_list = split_tokens_by_hw(mse_target_clean, tgt_hw_list)

        for b in range(len(tgt_hw_list)):
            if evaluated >= args.max_samples:
                break
            if not bool(geo_valid[b]):
                continue

            tgt_h_tok, tgt_w_tok = tgt_hw_list[b]
            tgt_h_tok = int(tgt_h_tok)
            tgt_w_tok = int(tgt_w_tok)
            if tgt_h_tok <= 0 or tgt_w_tok <= 0:
                skipped["invalid_hw"] += 1
                continue

            gt_depth_hw = F.interpolate(
                geo_batch["target_depth"][b].unsqueeze(0).unsqueeze(0),
                size=(tgt_h_tok, tgt_w_tok),
                mode="nearest",
            ).squeeze(0).squeeze(0)
            depth_valid = torch.isfinite(gt_depth_hw) & (gt_depth_hw > 0)
            if depth_valid.sum().item() == 0:
                skipped["zero_valid_depth"] += 1
                continue

            with torch.no_grad():
                pred_depth_tokens = depth_head(pred_target_token_list[b].to(dtype=depth_head_dtype))
                pred_depth_hw = F.softplus(pred_depth_tokens.float().reshape(tgt_h_tok, tgt_w_tok)) + 1e-6

                clean_depth_tokens = depth_head(clean_target_token_list[b].to(dtype=depth_head_dtype))
                clean_depth_hw = F.softplus(clean_depth_tokens.float().reshape(tgt_h_tok, tgt_w_tok)) + 1e-6

            pred_metrics = _depth_metrics(
                pred=pred_depth_hw,
                gt=gt_depth_hw,
                valid=depth_valid,
                si_lambda=args.si_lambda,
            )
            clean_metrics = _depth_metrics(
                pred=clean_depth_hw,
                gt=gt_depth_hw,
                valid=depth_valid,
                si_lambda=args.si_lambda,
            )

            const_depth = torch.full_like(gt_depth_hw, torch.median(gt_depth_hw[depth_valid]))
            const_metrics = _depth_metrics(
                pred=const_depth,
                gt=gt_depth_hw,
                valid=depth_valid,
                si_lambda=args.si_lambda,
            )

            pred_agg.add(pred_metrics)
            clean_agg.add(clean_metrics)
            const_agg.add(const_metrics)
            evaluated += 1
            pbar.update(1)

    pbar.close()

    out = {
        "config": {
            "model_path": os.path.abspath(args.model_path),
            "geo_aux_ckpt": geo_aux_ckpt,
            "geo_aux_manifest": os.path.abspath(args.geo_aux_manifest),
            "dataset_config_file": os.path.abspath(args.dataset_config_file),
            "device": str(device),
            "model_dtype": args.model_dtype,
            "autocast": args.autocast,
            "token_formula": args.token_formula,
            "disable_dropout": args.disable_dropout,
            "max_samples": args.max_samples,
            "max_batches": args.max_batches,
        },
        "progress": {
            "evaluated_samples": evaluated,
            "iterated_batches": iter_batches,
            "skipped": dict(skipped),
        },
        "pred_token_depth": pred_agg.summary(),
        "clean_token_depth": clean_agg.summary(),
        "const_baseline_depth": const_agg.summary(),
    }

    # Quick deltas for easier reading.
    pred_abs_rel = out["pred_token_depth"].get("abs_rel_scaled_mean")
    clean_abs_rel = out["clean_token_depth"].get("abs_rel_scaled_mean")
    const_abs_rel = out["const_baseline_depth"].get("abs_rel_scaled_mean")
    if pred_abs_rel is not None and const_abs_rel is not None:
        out["delta_pred_vs_const_abs_rel_mean"] = float(pred_abs_rel - const_abs_rel)
    if pred_abs_rel is not None and clean_abs_rel is not None:
        out["delta_pred_vs_clean_abs_rel_mean"] = float(pred_abs_rel - clean_abs_rel)

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        save_path = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
