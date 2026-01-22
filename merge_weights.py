import torch
from safetensors.torch import load_file, save_file
import argparse
import os
from tqdm import tqdm

def merge_models(base_path, ft_path, output_path):
    print(f"🔄 开始合并流程...")
    print(f"1. 加载官方底座权重 (Base): {base_path}")
    # 加载到 CPU 以节省显存，通常系统内存足够处理 7B 模型
    base_state_dict = load_file(base_path, device="cpu")
    
    print(f"2. 加载微调权重 (Fine-tuned): {ft_path}")
    ft_state_dict = load_file(ft_path, device="cpu")

    # 准备合并
    merged_state_dict = base_state_dict  # 浅拷贝，直接在 base 上修改
    
    updated_keys = 0
    missing_in_base = []

    print("3. 开始合并与精度转换 (Float32 -> BFloat16)...")
    
    # 遍历微调模型的每一个参数
    for key, value in tqdm(ft_state_dict.items(), desc="Updating weights"):
        # 1. 处理 DDP 训练可能产生的 'module.' 前缀
        clean_key = key.replace("module.", "")
        
        # 2. 检查这个 key 是否在底座模型中存在
        if clean_key in merged_state_dict:
            # 3. 核心步骤：
            #    a. 取出微调的权重 (value)
            #    b. 转换为 bfloat16 (匹配官方格式)
            #    c. 覆盖底座模型的权重
            merged_state_dict[clean_key] = value.to(dtype=torch.bfloat16)
            updated_keys += 1
        else:
            # 这种情况极少发生，除非你训练时修改了模型结构
            missing_in_base.append(clean_key)
            # 依然加入字典，以防万一
            merged_state_dict[clean_key] = value.to(dtype=torch.bfloat16)

    print(f"\n✅ 合并完成统计:")
    print(f"   - 总共更新了 {updated_keys} 个参数键值。")
    print(f"   - Base 中保留了 {len(base_state_dict) - updated_keys} 个原有参数 (包括 ViT/VAE 等冻结层)。")
    
    if missing_in_base:
        print(f"   - ⚠️ 警告: 有 {len(missing_in_base)} 个键在 Base 中未找到 (可能是新层): {missing_in_base[:5]}...")

    print(f"4. 保存合并后的模型到: {output_path}")
    # 保存时添加 metadata 标记为 pt 格式，防止读取时的 warning
    save_file(merged_state_dict, output_path, metadata={"format": "pt"})
    print("🎉 所有工作完成！现在可以用这个文件进行推理了。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge finetuned partial weights with official base weights.")
    
    # 你的官方 ema.safetensors 路径 (也就是你下载的那个完整模型)
    parser.add_argument("--base_model", type=str, required=True, help="Path to the official base ema.safetensors")
    
    # 你训练出来的 checkpoints/.../ema.safetensors
    parser.add_argument("--ft_model", type=str, required=True, help="Path to your finetuned ema.safetensors")
    
    # 输出路径
    parser.add_argument("--output", type=str, default="ema_merged.safetensors", help="Path to save the merged model")
    
    args = parser.parse_args()
    
    merge_models(args.base_model, args.ft_model, args.output)