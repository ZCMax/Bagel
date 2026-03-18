import json
import argparse
import os
import re
parser = argparse.ArgumentParser()
# parser.add_argument('--results_dir', default='./LaVIN', type=str)
parser.add_argument('--out-json',type=str)
def extract_single_choice_with_word_boundary(pred):
    """
    从预测文本中提取选项，并与正确答案比较。
    返回提取到的选项，如果没有找到则返回None。
    """
    if pred is None:
        return None

    # 确保pred是字符串类型
    try:
        pred = str(pred)
    except Exception:
        return None

    pattern_1 = r'``([^`]*)``'
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)  # 提取反引号之间的内容

    pattern_2 = r'`([^`]*)`'
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)  # 提取双反引号之间的内容

    pattern_3 = r'\b[A-D]\b(?!\s[a-zA-Z])'
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()  # 提取孤立的大写字母（排除"A bike"，不定冠词+空格+单词的情况）
    else:
        return None  # 如果没有匹配，返回 None

    return pred

if __name__ == '__main__':

    args = parser.parse_args()
    results = json.load(open(args.out_json))
    print(results)
    category_acc = {}
    correct = 0
    total = 0
    for sample in results:
        question = sample['question']
        answer = sample['answer']
        category = sample['category']
        index = sample['index']
        pred = sample['pred']
        extracted_pred = extract_single_choice_with_word_boundary(pred)
        print(extracted_pred)
        
        if category not in category_acc:
            category_acc[category] = []

            # 如果提取到了有效选项，进行得分计算
        if extracted_pred is not None:
            answer = answer.lower().replace("\n", " ").strip()
            predict = extracted_pred.lower().replace("\n", " ").strip()
            try:
                if answer == predict[0]:
                    sample['score'] = 1.0
                    correct += 1
                elif predict[0] == "(" and answer == predict[1]:
                    sample['score'] = 1.0
                    correct += 1
                elif predict[0:7] == "option " and answer == predict[7]:
                    sample['score'] = 1.0
                    correct += 1
                elif predict[0:14] == "the answer is " and answer == predict[14]:
                    sample['score'] = 1.0
                    correct += 1
            except Exception:
                pass
        if 'score' in sample:
            category_acc[category].append(sample['score'])
        else:
            category_acc[category].append(0.0)
        total += 1

    accuracy = correct / total if total > 0 else 0
    print("MMSI_Bench 评测结果：")
    print(f"总样本数: {total}")
    print(f"正确样本数: {correct}")
    print(f"准确率: {accuracy:.2%}")
    category_acc = {key:sum(category_acc[key])/len(category_acc[key]) for key in sorted(list(category_acc.keys()))}
    for key in category_acc:
        print(key+f'类型得分：{category_acc[key]:.2%}')
    