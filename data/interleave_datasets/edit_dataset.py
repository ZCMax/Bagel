# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import random
import os
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset, JsonStandardIterableDataset
from ..data_utils import pil_img2rgb

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # randomly choose start and end, return [0, 1] when only two images
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data, 
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False, 
            need_vae=True, 
            need_vit=True, 
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data, 
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True, 
                need_vae=False, 
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=True, 
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data, 
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, 
                        need_vae=False, 
                        need_vit=False,
                    )
        return data


class UnifiedSIIterableDataset(InterleavedBaseIterableDataset, JsonStandardIterableDataset):
    
    def parse_row(self, row, image_dir):
            """
            data 结构示例:
            {
                "context": ["images/39f.../image_color/1010.jpg", "images/39f.../4530.jpg"],
                "target": "images/39f.../image_color/1150.jpg",
                "instruction": "..."
            }
            image_dir 示例:

            """
            data = self._init_data()

            # ===============================================
            # 1. 处理 Context Images (原 Source Images)
            # ===============================================
            context_list = row.get("context", [])
            
            # 假设 context 里的图片都是 Condition (不计算 loss)
            for rel_path in context_list:
                # 拼接完整路径 (self.base_path 是解压后的根目录)
                full_path = os.path.join(image_dir, rel_path) # 或者根据你的逻辑拼接
                
                try:
                    img = Image.open(full_path).convert('RGB')
                    data = self._add_image(
                        data, img, 
                        need_loss=False, 
                        need_vae=True, 
                        need_vit=True
                    )
                except Exception as e:
                    print(f"Error loading context image {rel_path}: {e}")
                    return {} # 任何一张图坏了，整条数据作废

            # ===============================================
            # 2. 处理 Instruction
            # ===============================================
            instruction = row.get("instruction", "")
            data = self._add_text(data, instruction, need_loss=False)

            # ===============================================
            # 3. 处理 Target Image
            # ===============================================
            target_rel = row.get("target")
            if target_rel:
                full_path = os.path.join(image_dir, target_rel)
                try:
                    tgt_img = Image.open(full_path).convert('RGB')
                    data = self._add_image(
                        data, tgt_img, 
                        need_loss=True,  # 只有 Target 需要 Loss
                        need_vae=False,  # Target 通常不需要 VAE (取决于你的模型结构)
                        need_vit=False
                    )
                except Exception as e:
                    print(f"Error loading target image {target_rel}: {e}")
                    return {}
            
            return data