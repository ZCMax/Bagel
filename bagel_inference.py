import argparse
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.pose_condition import compose_instruction_with_pose


def get_args():
    parser = argparse.ArgumentParser(
        description="BAGEL image+text understanding inference (batch jsonl -> text jsonl)"
    )

    # model
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--ByteDance-Seed-BAGEL-7B-MoT",
        help="Path to BAGEL checkpoint directory",
    )
    parser.add_argument("--mode", type=int, default=1, help="1: bf16, 2: nf4, 3: int8")

    # input / output
    parser.add_argument("--eval-json", type=str, required=True, help="Input jsonl file")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--out-name",
        type=str,
        default="predictions.jsonl",
        help="Output jsonl filename",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Optional root path for relative image paths in jsonl",
    )
    parser.add_argument("--num", type=int, default=-1, help="Number of samples to run. <=0 means all.")
    parser.add_argument("--idx", type=int, default=-1, help="Run only one sample by index")
    parser.add_argument(
        "--random_subset",
        action="store_true",
        help="When --num > 0, sample randomly instead of taking first N",
    )

    # generation
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--text_temperature", type=float, default=0.3)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--show_thinking", action="store_true", help="Enable think-style output")
    parser.add_argument(
        "--add_context_role_text",
        action="store_true",
        help="Insert role text like 'This is the first image.' before each context image",
    )
    parser.add_argument(
        "--inject_pose_text",
        action="store_true",
        help="Append pose-matrix condition text from json fields (context_poses/target_pose/start_image_id).",
    )
    parser.add_argument(
        "--pose_text_replace_instruction",
        action="store_true",
        help="Replace natural-language instruction with pose-matrix condition text.",
    )
    parser.add_argument(
        "--pose_text_require_valid",
        action="store_true",
        help="Mark sample as failed if pose fields are missing when pose text is enabled.",
    )
    parser.add_argument(
        "--pose_text_precision",
        type=int,
        default=4,
        help="Decimal precision used to serialize pose matrices.",
    )
    parser.add_argument(
        "--no_pose_text_start_image_id",
        action="store_true",
        help="Do not include start_image_id in the serialized pose condition text.",
    )
    parser.add_argument(
        "--use_vit_cond",
        dest="use_vit_cond",
        action="store_true",
        help="Use ViT branch for understanding (recommended).",
    )
    parser.add_argument(
        "--no-use_vit_cond",
        dest="use_vit_cond",
        action="store_false",
        help="Disable ViT branch for understanding.",
    )
    parser.set_defaults(use_vit_cond=True)

    return parser.parse_args()


def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_interleaved_inputs(input_images, instruction, add_context_role_text=False):
    if not add_context_role_text:
        return [*input_images, instruction]

    role_texts = [
        "This is the first image.",
        "This is the second image.",
    ]
    input_terms = []
    for idx, image in enumerate(input_images):
        if idx < len(role_texts):
            input_terms.append(role_texts[idx])
        else:
            input_terms.append(f"This is context image {idx + 1}.")
        input_terms.append(image)
    input_terms.append(instruction)
    return input_terms


def report_reference_stats(samples):
    first_re = re.compile(r"\b(first|initial)\s+image\b", re.I)
    second_re = re.compile(r"\b(second|last)\s+image\b", re.I)
    first_cnt = 0
    second_cnt = 0
    neither_cnt = 0
    total = 0
    for item in samples:
        prompt = resolve_prompt(item, raise_if_missing=False)
        if prompt is None:
            continue
        total += 1
        has_first = bool(first_re.search(prompt))
        has_second = bool(second_re.search(prompt))
        if has_first:
            first_cnt += 1
        if has_second:
            second_cnt += 1
        if (not has_first) and (not has_second):
            neither_cnt += 1
    print(
        f"[INFO] reference stats -> first:{first_cnt}, second/last:{second_cnt}, "
        f"neither:{neither_cnt}, total_with_prompt:{total}"
    )


def resolve_prompt(item, raise_if_missing=True):
    for key in ("instruction", "prompt", "question", "text", "query"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if raise_if_missing:
        raise KeyError("Missing prompt text key. Expected one of: instruction/prompt/question/text/query")
    return None


def resolve_context_paths(item, image_root):
    if "context" in item:
        paths = item["context"]
    elif "images" in item:
        paths = item["images"]
    elif "image" in item:
        paths = [item["image"]]
    elif "img" in item:
        paths = [item["img"]]
    else:
        raise KeyError("Missing image field. Expected one of: context/images/image/img")

    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, (list, tuple)):
        raise TypeError(f"Expected image paths to be list/tuple/str, but got {type(paths)}")

    output_paths = []
    for p in paths:
        if not isinstance(p, str):
            raise TypeError(f"Each image path must be a string, got {type(p)}")
        if os.path.isabs(p) or not image_root:
            output_paths.append(p)
        else:
            output_paths.append(os.path.join(image_root, p))
    return output_paths


def load_images(paths):
    images = []
    for p in paths:
        images.append(Image.open(p).convert("RGB"))
    return images


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    try:
        from accelerate.big_modeling import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    except Exception as e:
        raise RuntimeError(
            "Failed to import `accelerate`. Please ensure accelerate/bitsandbytes/triton runtime is installed."
        ) from e

    print(f"Loading model from: {args.model_path}")
    model_path = args.model_path

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    print(f"Mode: {args.mode} (1=bf16, 2=nf4, 3=int8)")
    if args.mode == 1:
        try:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(model_path, "ema_merged.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                offload_folder="offload",
                dtype=torch.bfloat16,
                force_hooks=True,
            ).eval()
        except Exception:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(model_path, "ema.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                offload_folder="offload",
                dtype=torch.bfloat16,
                force_hooks=True,
            ).eval()
    elif args.mode == 2:
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

        bnb_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, "ema.safetensors"),
            bnb_quantization_config=bnb_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif args.mode == 3:
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

        bnb_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, "ema.safetensors"),
            bnb_quantization_config=bnb_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"Unsupported mode: {args.mode}")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    val_datas = read_jsonl(args.eval_json)
    if args.idx > -1:
        if args.idx >= len(val_datas):
            raise IndexError(f"--idx={args.idx} out of range for {len(val_datas)} samples")
        repeat_n = args.num if args.num > 0 else 1
        val_datas = [val_datas[args.idx]] * repeat_n
    elif args.num > 0 and args.num < len(val_datas):
        if args.random_subset:
            val_datas = random.sample(val_datas, args.num)
        else:
            val_datas = val_datas[:args.num]

    report_reference_stats(val_datas)

    print(
        f"Start inference: total={len(val_datas)}, show_thinking={args.show_thinking}, "
        f"use_vit_cond={args.use_vit_cond}, output={out_path}"
    )

    success_n = 0
    error_n = 0
    with open(out_path, "w", encoding="utf-8") as writer:
        for sample_idx, item in enumerate(tqdm(val_datas)):
            sample_id = item.get("id", sample_idx)
            images = []
            try:
                prompt = resolve_prompt(item)
                prompt, has_pose_text = compose_instruction_with_pose(
                    instruction=prompt,
                    row=item,
                    inject_pose_text=args.inject_pose_text,
                    replace_instruction=args.pose_text_replace_instruction,
                    include_start_image_id=not args.no_pose_text_start_image_id,
                    precision=args.pose_text_precision,
                )
                if args.inject_pose_text and args.pose_text_require_valid and (not has_pose_text):
                    raise ValueError("pose text requested but pose fields are missing/invalid.")
                context_paths = resolve_context_paths(item, args.image_root)
                images = load_images(context_paths)

                input_terms = build_interleaved_inputs(
                    images,
                    prompt,
                    add_context_role_text=args.add_context_role_text,
                )
                output_list = inferencer.interleave_inference(
                    input_lists=input_terms,
                    think=args.show_thinking,
                    understanding_output=True,
                    use_vit_cond=args.use_vit_cond,
                    max_think_token_n=args.max_new_tokens,
                    do_sample=args.do_sample,
                    text_temperature=args.text_temperature,
                )

                pred_text = ""
                for out in output_list:
                    if isinstance(out, str):
                        pred_text = out

                if not pred_text:
                    raise RuntimeError("No text output returned by inferencer.")

                record = {
                    "id": sample_id,
                    "prediction": pred_text,
                    "instruction": prompt,
                    "context": item.get("context", item.get("images", item.get("image", item.get("img")))),
                    "used_pose_text": bool(has_pose_text),
                }
                if "answer" in item:
                    record["answer"] = item["answer"]
                success_n += 1
            except Exception as e:
                record = {
                    "id": sample_id,
                    "prediction": "",
                    "error": str(e),
                }
                error_n += 1
                print(f"[WARN] failed on id={sample_id}: {e}")
            finally:
                for im in images:
                    try:
                        im.close()
                    except Exception:
                        pass

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Finished. success={success_n}, error={error_n}, saved={out_path}")


if __name__ == "__main__":
    main()
