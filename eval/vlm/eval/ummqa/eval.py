path1 = '/mnt/petrelfs/linjingli/st_new/spatio-temporal-benchmark/data/UMM_QA/umm_qa_samples_balanced_scannet_cam2obj.json'


# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import os
import re
from PIL import Image
from tqdm import tqdm
import torch
import json
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
import random
random.seed(137)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='eval/vlm/eval/mme/Your_Results')
    parser.add_argument('--out_json', type=str, default='/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ummqa/bagel_ummqa.json')
    parser.add_argument('--model-path', type=str, default='/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT')
    args = parser.parse_args()

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    
    annno_path = '/mnt/petrelfs/linjingli/st_new/spatio-temporal-benchmark/data/UMM_QA/umm_qa_samples_balanced_scannet_cam2obj.json'
    
    anno_data = json.load(open(annno_path))
    print(len(anno_data))
    anno_data = random.sample(anno_data,1000)
    results = []
    for sample in tqdm(anno_data):
        question = sample['Prompt']
        answer = sample['Answer']
        images = [Image.open(img_path).convert('RGB') for img_path in sample['Images']]
        post_prompt = ("Answer with the option's letter from the given choices directly. "
                       "Enclose the option's letter within ``.")
        prompt = f'{question}\n{post_prompt}'
        images, conversation = process_conversation(images, prompt)

        response = model.chat(
            tokenizer, 
            new_token_ids,
            image_transform,
            images=images,
            prompt=conversation,
            max_length=20,
        )

        print(question, answer, response)
        sample['pred'] = response
        results.append(sample)
    with open(args.out_json,'w') as f:
        json.dump(results,f,indent=4)


