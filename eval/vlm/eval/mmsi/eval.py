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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='eval/vlm/eval/mme/Your_Results')
    parser.add_argument('--out_json', type=str, default='results/bagel_mmsi.json')
    parser.add_argument('--model-path', type=str, default='/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT')
    args = parser.parse_args()

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    
    mmsi_image_path = '/mnt/petrelfs/linjingli/st_new/spatio-temporal-benchmark/VLMEvalKit_/to_transfer/data/images/mmsi_bench'
    mmsi_anno_path = '/mnt/petrelfs/linjingli/st_new/spatio-temporal-benchmark/VLMEvalKit_/to_transfer/data/annotations/mmsi_bench.json'

    image_data_group = {}
    all_image_name = sorted([img for img in os.listdir(mmsi_image_path) if img.split('.')[-1]=='jpg'])
    for image_name in all_image_name:
        question_index = int(image_name.split('_')[0])
        if question_index not in image_data_group:
            image_data_group[question_index] = []
        image_data_group[question_index].append(image_name)
    print(image_data_group[7])
    
    ost_anno = json.load(open(mmsi_anno_path))
    results = []
    for sample in tqdm(ost_anno):
        question = sample['question']
        answer = sample['answer']
        category = sample['category']
        index = sample['index']
        images = [Image.open(os.path.join(mmsi_image_path,img_name)).convert('RGB') for img_name in image_data_group[index]]
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

        print(index, question, answer, response)
        sample['pred'] = response
        results.append(sample)
    with open(args.out_json,'w') as f:
        json.dump(results,f,indent=4)


