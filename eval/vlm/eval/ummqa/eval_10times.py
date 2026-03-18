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
import copy
import json
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import torch

# Ensure `python eval/vlm/eval/ummqa/eval_10times.py` works from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ANNO_PATH = (
    "/mnt/petrelfs/linjingli/st_new/spatio-temporal-benchmark/data/UMM_QA/"
    "umm_qa_samples_balanced_scannet_cam2obj.json"
)
DEFAULT_POST_PROMPT = (
    "Answer with the option's letter from the given choices directly. "
    "Enclose the option's letter within ``."
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_single_choice(pred):
    if pred is None:
        return None
    try:
        text = str(pred).strip()
    except Exception:
        return None

    patterns = [
        r"``\s*([A-Da-d])\s*``",
        r"`\s*([A-Da-d])\s*`",
        r"option\s*([A-Da-d])",
        r"answer\s+is\s*([A-Da-d])",
        r"\b([A-Da-d])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def judge_difficulty(correct_10, total_10, correct_100, total_100):
    if total_10 >= 10 and correct_10 == 10:
        return "too_easy"
    if total_100 >= 100 and correct_100 == 0:
        return "too_hard"
    return "normal"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="eval/vlm/eval/mme/Your_Results")
    parser.add_argument(
        "--out_json",
        type=str,
        default="/mnt/petrelfs/linjingli/UMM_Spatial/bagel/results/ummqa/bagel_ummqa_10times.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/inspurfs/mozi_t/linjingli/bagel/BAGEL-7B-MoT",
    )
    parser.add_argument("--anno_path", type=str, default=DEFAULT_ANNO_PATH)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument(
        "--fixed_index",
        type=int,
        default=-1,
        help="Only evaluate one sample index from annotation list when >= 0.",
    )
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--easy_trials", type=int, default=10)
    parser.add_argument("--hard_trials", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from eval.vlm.utils import build_transform, load_model_and_tokenizer, process_conversation

    if args.easy_trials != 10:
        print(f"[warn] easy_trials={args.easy_trials}, rule '10/10 correct => too_easy' becomes {args.easy_trials}/{args.easy_trials}.")
    if args.hard_trials != 100:
        print(f"[warn] hard_trials={args.hard_trials}, rule '100/100 wrong => too_hard' becomes {args.hard_trials}/{args.hard_trials}.")
    if args.hard_trials < args.easy_trials:
        raise ValueError("hard_trials must be >= easy_trials.")

    set_seed(args.seed)

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[test] total_params: {total_params}B")

    with open(args.anno_path, "r") as f:
        anno_data = json.load(f)
    print(f"[test] annotation size: {len(anno_data)}")

    if args.fixed_index >= 0:
        if args.fixed_index >= len(anno_data):
            raise IndexError(f"fixed_index={args.fixed_index} out of range, len={len(anno_data)}")
        eval_data = [anno_data[args.fixed_index]]
    else:
        if args.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        sample_n = min(args.num_samples, len(anno_data))
        eval_data = random.sample(anno_data, sample_n)
    print(f"[test] evaluating {len(eval_data)} sample(s).")

    results = []
    global_trials = 0
    global_correct = 0
    too_easy_count = 0
    too_hard_count = 0
    normal_count = 0

    for sample_idx, sample in enumerate(tqdm(eval_data)):
        question = sample["Prompt"]
        answer = extract_single_choice(sample.get("Answer"))
        if answer is None:
            answer = str(sample.get("Answer", "")).strip().upper()

        pil_images = []
        for img_path in sample["Images"]:
            with Image.open(img_path) as img:
                pil_images.append(img.convert("RGB"))
        prompt = f"{question}\n{DEFAULT_POST_PROMPT}"
        images, conversation = process_conversation(pil_images, prompt)

        trial_results = []
        correct_10 = 0
        correct_100 = 0
        total_10 = 0
        total_100 = 0

        # First stage: run 10 (easy_trials) times.

        for trial_idx in range(args.easy_trials):
            try:
                trial_seed = args.seed + sample_idx * 1000 + trial_idx
                set_seed(trial_seed)
                response = model.chat(
                    tokenizer,
                    new_token_ids,
                    image_transform,
                    images=images,
                    prompt=conversation,
                    max_length=args.max_length,
                    do_sample=True,
                    temperature=args.temperature,
                )
                pred_letter = extract_single_choice(response)
                is_correct = pred_letter == answer

                total_10 += 1
                total_100 += 1
                global_trials += 1
                if is_correct:
                    correct_10 += 1
                    correct_100 += 1
                    global_correct += 1

                trial_results.append(
                    {
                        "trial": trial_idx + 1,
                        "seed": trial_seed,
                        "response": response,
                        "pred_letter": pred_letter,
                        "is_correct": is_correct,
                    }
                )

                if args.verbose:
                    print(
                        f"[sample {sample_idx}][trial {trial_idx + 1}] "
                        f"gt={answer} pred={pred_letter} correct={is_correct}"
                    )
            except:
                continue

        # Second stage: only when first easy_trials are all wrong.
        if correct_10 == 0 and args.hard_trials > args.easy_trials:
            for trial_idx in range(args.easy_trials, args.hard_trials):
                trial_seed = args.seed + sample_idx * 1000 + trial_idx
                set_seed(trial_seed)
                response = model.chat(
                    tokenizer,
                    new_token_ids,
                    image_transform,
                    images=images,
                    prompt=conversation,
                    max_length=args.max_length,
                    do_sample=True,
                    temperature=args.temperature,
                )
                pred_letter = extract_single_choice(response)
                is_correct = pred_letter == answer

                total_100 += 1
                global_trials += 1
                if is_correct:
                    correct_100 += 1
                    global_correct += 1

                trial_results.append(
                    {
                        "trial": trial_idx + 1,
                        "seed": trial_seed,
                        "response": response,
                        "pred_letter": pred_letter,
                        "is_correct": is_correct,
                    }
                )

                if args.verbose:
                    print(
                        f"[sample {sample_idx}][trial {trial_idx + 1}] "
                        f"gt={answer} pred={pred_letter} correct={is_correct}"
                    )

        difficulty = judge_difficulty(correct_10, total_10, correct_100, total_100)
        if difficulty == "too_easy":
            too_easy_count += 1
        elif difficulty == "too_hard":
            too_hard_count += 1
        else:
            normal_count += 1

        sample_result = copy.deepcopy(sample)
        sample_result["gt_letter"] = answer
        sample_result["trials_run"] = len(trial_results)
        sample_result["correct_10"] = correct_10
        sample_result["total_10"] = total_10
        sample_result["acc_10"] = (correct_10 / total_10) if total_10 > 0 else 0.0
        sample_result["correct_100"] = correct_100
        sample_result["total_100"] = total_100
        sample_result["acc_100"] = (correct_100 / total_100) if total_100 > 0 else 0.0
        sample_result["difficulty"] = difficulty
        sample_result["trial_results"] = trial_results
        results.append(sample_result)

    summary = {
        "num_samples": len(results),
        "temperature": args.temperature,
        "do_sample": True,
        "easy_trials": args.easy_trials,
        "hard_trials": args.hard_trials,
        "global_trials": global_trials,
        "global_correct": global_correct,
        "global_acc": (global_correct / global_trials) if global_trials > 0 else 0.0,
        "too_easy_count": too_easy_count,
        "too_hard_count": too_hard_count,
        "normal_count": normal_count,
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_data = {
        "summary": summary,
        "results": results,
    }
    with open(args.out_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print("[done] summary:", json.dumps(summary, indent=2))
    print(f"[done] saved: {args.out_json}")
