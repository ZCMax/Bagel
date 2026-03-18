import os
import torch
import numpy as np
import argparse
import random
import re
from PIL import Image
from tqdm import tqdm
import json
# 移除 gradio
# import gradio as gr 
from data.pose_condition import compose_instruction_with_pose

def get_args():
    parser = argparse.ArgumentParser(description="BAGEL Local Inference with 2 Images")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--ByteDance-Seed-BAGEL-7B-MoT", help="Path to the model")
    parser.add_argument("--mode", type=int, default=1, help="1: bf16, 2: nf4, 3: int8")
    parser.add_argument("--num", type=int, default=-1, help="Number of samples to run. <=0 means all.")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--random_subset", action="store_true", help="When --num > 0, sample randomly instead of taking the first N.")
    parser.add_argument("--image-root", type=str, default="/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv")
    # 输入输出相关参数
    parser.add_argument("--eval-json", type=str, required=True, help="Path to evaluation json file")
    parser.add_argument("--out-dir", type=str, required=True, help="Path to evaluation json file")
    # 推理超参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=3.0, help="Override diffusion timestep_shift. Default uses model config.")
    parser.add_argument("--cfg_interval_start", type=float, default=0.0)
    parser.add_argument("--cfg_interval_end", type=float, default=1.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel", choices=["global", "channel", "text_channel"])
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--text_temperature", type=float, default=0.3)
    parser.add_argument("--show_thinking", action="store_true", help="Enable thinking process output")
    parser.add_argument(
        "--use_vit_cond",
        action="store_true",
        help="Enable ViT condition branch. Keep this OFF if training used --visual_und False.",
    )
    parser.add_argument(
        "--add_context_role_text",
        action="store_true",
        help="Insert explicit role text ('first image', 'second image') before each context image.",
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
        help="Skip a sample if pose fields are missing or invalid when pose text is enabled.",
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
    
    args = parser.parse_args()
    return args

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

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir,exist_ok=True)

    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
        SiglipVisionConfig, SiglipVisionModel
    )
    from modeling.qwen2 import Qwen2Tokenizer

    try:
        from accelerate.big_modeling import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    except Exception as e:
        raise RuntimeError(
            "Failed to import `accelerate`. Please ensure accelerate/bitsandbytes/triton runtime is installed correctly."
        ) from e

    print(f"Loading model from: {args.model_path}")
    
    # --- 1. 模型初始化 (保持原逻辑不变) ---
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
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)

    # --- 2. 设备分配与加载 (保持原逻辑不变) ---
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map: device_map[k] = first_device

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
        except:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(model_path, "ema.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                offload_folder="offload",
                dtype=torch.bfloat16,
                force_hooks=True,
            ).eval()
    elif args.mode == 2: # NF4
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
        bnb_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif args.mode == 3: # INT8
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
        raise NotImplementedError

    # 初始化推理器
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

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

    def read_jsonl(file_path):
        """逐行读取JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data.append(json.loads(line))
        return data

    def report_reference_stats(samples):
        first_re = re.compile(r"\b(first|initial)\s+image\b", re.I)
        second_re = re.compile(r"\b(second|last)\s+image\b", re.I)
        first_cnt = 0
        second_cnt = 0
        neither_cnt = 0
        for item in samples:
            text = item.get("instruction", "")
            has_first = bool(first_re.search(text))
            has_second = bool(second_re.search(text))
            if has_first:
                first_cnt += 1
            if has_second:
                second_cnt += 1
            if (not has_first) and (not has_second):
                neither_cnt += 1
        print(
            f"[INFO] reference stats -> first:{first_cnt}, second/last:{second_cnt}, "
            f"neither:{neither_cnt}, total:{len(samples)}"
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

        
    
    for item in tqdm(val_datas):
    
        input_images = [Image.open(os.path.join(args.image_root,image_path)).convert("RGB") for image_path in item['context']]
        instruction = item.get("instruction", "")
        instruction, has_pose_text = compose_instruction_with_pose(
            instruction=instruction,
            row=item,
            inject_pose_text=args.inject_pose_text,
            replace_instruction=args.pose_text_replace_instruction,
            include_start_image_id=not args.no_pose_text_start_image_id,
            precision=args.pose_text_precision,
        )
        if args.inject_pose_text and args.pose_text_require_valid and (not has_pose_text):
            print(f"[WARN] skip id={item.get('id', 'unknown')}: pose text requested but pose fields are invalid")
            continue
        
        # 3.2 准备参数
        # 这些参数参考了原代码中 edit_image 的默认值
        inference_hyper = dict(
            max_think_token_n=1024 if args.show_thinking else 1024,
            do_sample=args.show_thinking, # 如果开启thinking才sample，否则默认False
            text_temperature=args.text_temperature,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=[args.cfg_interval_start, args.cfg_interval_end],  
            timestep_shift=3.0,
            num_timesteps=args.num_timesteps,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
        )


        # 3.3 运行推理
        input_terms = build_interleaved_inputs(
            input_images,
            instruction,
            add_context_role_text=args.add_context_role_text,
        )
        output_list = inferencer.interleave_inference(
            input_lists=input_terms,
            think=args.show_thinking,
            use_vit_cond=args.use_vit_cond,
            **inference_hyper,
        )

        # --- 4. 保存结果 ---
        output_img = None
        text_output = ""
        for out in output_list:
            if isinstance(out, Image.Image):
                output_img = out
            elif isinstance(out, str):
                text_output = out

        if output_img is None:
            print(f"[WARN] generation failed on id={item.get('id', 'unknown')}")
            continue
        output_img.save(os.path.join(args.out_dir,str(item['id'])+'.png'))

if __name__ == "__main__":
    main()
