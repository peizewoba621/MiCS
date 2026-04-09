import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, default="llama2-7b", help='llama2-7b or llama2-13b or llama3-8b')
parser.add_argument(
    '--dataset',
    type=str,
    default="hotpot",
    help='ragtruth, dolly'
)
args = parser.parse_args()
if args.dataset == "hotpot":
    response_path = "./log/test_llama2_7B/llama2_7B_response_vhp_1000.json"


def calculate_loss(selected_indices, true_indices):
    # 方法1: 计算重叠度
    overlap = len(set(selected_indices) & set(true_indices))
    total = len(set(selected_indices) | set(true_indices))
    overlap_ratio = overlap / total if total > 0 else 0

    # 方法2: 计算排序损失（如果支持事实有优先级）
    # 这里可以根据具体需求定义损失函数

    return {
        'overlap_count': overlap,
        'overlap_ratio': overlap_ratio,
        'selected_indices': selected_indices,
        'true_indices': true_indices
    }


def calculate_paragraph_selection_loss(top2_paragraphs, real2_paragraphs, top2_scores, real2_scores):
    """
    计算段落选择损失 - 比较预测的Top2段落与真实supporting facts段落

    Args:
        top2_paragraphs: 预测的top2段落 [(idx1, score1), (idx2, score2)]
        real2_paragraphs: 真实的段落 [(idx1, score1), (idx2, score2)]
        top2_scores: 预测段落的分数 [0.8, 0.6]
        real2_scores: 真实段落的分数 [0.7, 0.9]

    Returns:
        dict: 包含各种损失指标的字典
    """
    # 提取段落索引
    top2_indices = [idx for idx, _ in top2_paragraphs]
    real2_indices = [idx for idx, _ in real2_paragraphs]

    # 1. 计算重叠度
    overlap = len(set(top2_indices) & set(real2_indices))
    overlap_accuracy = overlap / 2.0  # 总共2个段落

    # 2. 计算精确率和F1分数
    precision = overlap / len(top2_indices) if len(top2_indices) > 0 else 0
    recall = overlap / len(real2_indices) if len(real2_indices) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 3. 计算分数差异损失
    score_loss = 0
    if len(top2_scores) > 0 and len(real2_scores) > 0:
        # 确保长度一致
        min_len = min(len(top2_scores), len(real2_scores))
        top2_array = np.array(top2_scores[:min_len])
        real2_array = np.array(real2_scores[:min_len])
        score_loss = np.mean(np.abs(top2_array - real2_array))

    # 4. 综合损失 (重叠损失 + 分数损失)
    overlap_loss = 1 - overlap_accuracy  # 重叠度越高，损失越小
    total_loss = overlap_loss * 0.7 + score_loss * 0.3  # 重叠度权重更高

    return {
        'overlap_count': overlap,
        'overlap_accuracy': overlap_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'score_loss': score_loss,
        'overlap_loss': overlap_loss,
        'total_loss': total_loss
    }


def calculate_ranking_loss(item, supporting_indices):
    """
    计算排序损失 - 真值段落在所有段落中的排名情况

    Args:
        item: 数据项，包含所有段落的外部相似度分数
        supporting_indices: 真值段落的索引列表

    Returns:
        dict: 包含排序损失信息的字典
    """
    # 获取所有10个段落的分数
    all_scores = []
    for k in range(10):
        key = f"external_similarity_avg{k}"
        if key in item:
            all_scores.append((k, item[key]))
        else:
            all_scores.append((k, 0.0))

    # 按分数降序排序
    all_scores_sorted = sorted(all_scores, key=lambda x: x[1], reverse=True)

    # 找到真值段落的排名
    true_paragraph_ranks = []
    true_paragraph_scores = []

    for true_idx in supporting_indices:
        # 找到真值段落在排序中的位置（排名从1开始）
        rank = next(i + 1 for i, (idx, score) in enumerate(all_scores_sorted) if idx == true_idx)
        true_paragraph_ranks.append(rank)
        true_paragraph_scores.append(item.get(f"external_similarity_avg{true_idx}", 0.0))

    # 计算排名损失
    # 理想情况下，真值段落应该排在前2位
    ideal_ranks = [1, 2]  # 理想排名
    ranking_losses = []

    for i, actual_rank in enumerate(true_paragraph_ranks):
        ideal_rank = ideal_ranks[i] if i < len(ideal_ranks) else i + 1
        # 排名损失 = (实际排名 - 理想排名) / 总段落数
        loss = (actual_rank - ideal_rank) / 10.0
        ranking_losses.append(loss)

    avg_ranking_loss = np.mean(ranking_losses) if ranking_losses else 0.0
    best_rank = min(true_paragraph_ranks) if true_paragraph_ranks else 10
    worst_rank = max(true_paragraph_ranks) if true_paragraph_ranks else 10

    return {
        'true_paragraph_ranks': true_paragraph_ranks,
        'true_paragraph_scores': true_paragraph_scores,
        'avg_ranking_loss': avg_ranking_loss,
        'best_rank': best_rank,
        'worst_rank': worst_rank,
        'all_ranks_sorted': [(idx, score, rank) for rank, (idx, score) in enumerate(all_scores_sorted, 1)]
    }


total_samples = 0
first_overlap_count = 0
second_overlap_count = 0
both_overlap_count = 0
neither_overlap_count = 0

# 损失统计
total_overlap_accuracy = 0
total_precision = 0
total_f1_score = 0
total_score_loss = 0
total_combined_loss = 0

# 排序损失统计
total_ranking_loss = 0
total_best_rank = 0
total_worst_rank = 0

response = []
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response.append(data)


def comprehensive_evaluation(predicted_indices, true_indices, predicted_scores=None):
    """
    综合评估模型预测的段落选择效果

    Args:
        predicted_indices: 模型预测的段落索引列表 [0, 5]
        true_indices: 真实的支持事实段落索引列表 [2, 7]
        predicted_scores: 所有段落的外部分数列表（可选）

    Returns:
        dict: 包含各种评估指标的字典
    """
    predicted_set = set(predicted_indices)
    true_set = set(true_indices)

    # 1. 完全匹配准确率
    exact_match = 1.0 if predicted_set == true_set else 0.0

    # 2. 重叠准确率（召回率）
    overlap_count = len(predicted_set & true_set)
    overlap_accuracy = overlap_count / len(true_set) if len(true_set) > 0 else 0.0

    # 3. 精确率
    precision = overlap_count / len(predicted_set) if len(predicted_set) > 0 else 0.0

    # 4. F1分数
    if precision + overlap_accuracy == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * overlap_accuracy / (precision + overlap_accuracy)

    # 5. 排序准确率（如果有分数）
    ranking_accuracy = None
    if predicted_scores is not None:
        true_scores = [predicted_scores[i] for i in true_indices if i < len(predicted_scores)]
        false_scores = [predicted_scores[i] for i in range(len(predicted_scores)) if i not in true_indices]

        correct_pairs = 0
        total_pairs = len(true_scores) * len(false_scores)

        for true_score in true_scores:
            for false_score in false_scores:
                if true_score > false_score:
                    correct_pairs += 1

        ranking_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0

    # 6. 详细分类
    first_correct = 1 if predicted_indices[0] in true_indices else 0
    second_correct = 1 if predicted_indices[1] in true_indices else 0
    both_correct = 1 if first_correct and second_correct else 0
    neither_correct = 1 if not first_correct and not second_correct else 0

    return {
        'exact_match': exact_match,
        'overlap_accuracy': overlap_accuracy,
        'precision': precision,
        'f1_score': f1_score,
        'ranking_accuracy': ranking_accuracy,
        'overlap_count': overlap_count,
        'first_correct': first_correct,
        'second_correct': second_correct,
        'both_correct': both_correct,
        'neither_correct': neither_correct,
        'predicted_indices': predicted_indices,
        'true_indices': true_indices
    }


def print_evaluation_results(evaluation, sample_id=None):
    """打印评估结果"""
    if sample_id:
        print(f"\n=== 样本 {sample_id} 评估结果 ===")
    else:
        print(f"\n=== 评估结果 ===")

    print(f"预测段落: {evaluation['predicted_indices']}")
    print(f"真实段落: {evaluation['true_indices']}")
    print(f"重叠数量: {evaluation['overlap_count']}/2")
    print(f"完全匹配: {evaluation['exact_match']:.4f}")
    print(f"重叠准确率: {evaluation['overlap_accuracy']:.4f}")
    print(f"精确率: {evaluation['precision']:.4f}")
    print(f"F1分数: {evaluation['f1_score']:.4f}")
    if evaluation['ranking_accuracy'] is not None:
        print(f"排序准确率: {evaluation['ranking_accuracy']:.4f}")

    print(f"第一个正确: {evaluation['first_correct']}")
    print(f"第二个正确: {evaluation['second_correct']}")
    print(f"两个都正确: {evaluation['both_correct']}")
    print(f"两个都错误: {evaluation['neither_correct']}")


total_samples = 0
total_exact_match = 0
total_overlap_accuracy = 0
total_f1_score = 0
total_both_correct = 0
total_neither_correct = 0
total_precision = 0
total_ranking_accuracy = 0
for i in tqdm(range(len(response))):
    for item in response[i]:
        # 获取top2段落
        scores_with_k = []
        for k in range(10):
            key = f"external_similarity_wei{k}"  # avg,max,mid,last5,wei
            if key in item:
                scores_with_k.append((k, item[key]))

        top_2 = sorted(scores_with_k, key=lambda x: x[1], reverse=True)[:2]
        top_2_indices = [k for k, score in top_2]

        print(top_2)
        # 获取supporting facts的段落索引
        supporting_facts = item.get('supporting_facts', [])
        supporting_titles = [fact[0] for fact in supporting_facts]

        # 找到supporting facts对应的段落索引
        supporting_indices = []
        for k in range(10):
            context_key = f"context{k}"
            if context_key in item:
                context_content = item[context_key]
                title = context_content.split(":")[0] if ":" in context_content else ""
                if title in supporting_titles:
                    supporting_indices.append(k)
        # print(supporting_indices)

        # 获取real2段落
        scores_with_m = []
        for k in supporting_indices:
            key = f"external_similarity_avg{k}"  # avg,max,mid,last5,wei
            if key in item:
                scores_with_m.append((k, item[key]))

        real_2 = sorted(scores_with_m, key=lambda x: x[1], reverse=True)[:2]
        print(real_2)

        # 计算损失 - 比较预测的Top2段落与真实supporting facts段落
        # 获取预测段落的分数
        top2_scores = [score for _, score in top_2]
        # 获取真实supporting facts段落的分数
        real2_scores = [score for _, score in real_2]

        # 计算段落选择损失
        loss_result = calculate_paragraph_selection_loss(top_2, real_2, top2_scores, real2_scores)

        # 计算排序损失 - 真值段落在所有段落中的排名
        ranking_loss = calculate_ranking_loss(item, supporting_indices)

        print(f"=== 段落选择损失分析 ===")
        print(f"预测Top2段落: {[idx for idx, _ in top_2]}")
        print(f"真实段落: {[idx for idx, _ in real_2]}")
        print(f"预测分数: {top2_scores}")
        print(f"真实分数: {real2_scores}")
        print(f"重叠数量: {loss_result['overlap_count']}/2")
        print(f"重叠准确率: {loss_result['overlap_accuracy']:.4f}")
        print(f"精确率: {loss_result['precision']:.4f}")
        print(f"F1分数: {loss_result['f1_score']:.4f}")
        print(f"分数差异损失: {loss_result['score_loss']:.4f}")
        print(f"综合损失: {loss_result['total_loss']:.4f}")

        print(f"\n=== 排序损失分析 ===")
        print(f"真值段落排名: {ranking_loss['true_paragraph_ranks']}")
        print(f"真值段落分数: {ranking_loss['true_paragraph_scores']}")
        print(f"平均排名损失: {ranking_loss['avg_ranking_loss']:.4f}")
        print(f"最高排名: {ranking_loss['best_rank']}")
        print(f"最差排名: {ranking_loss['worst_rank']}")
        print("-" * 50)

        # 累计统计
        total_samples += 1
        total_overlap_accuracy += loss_result['overlap_accuracy']
        total_precision += loss_result['precision']
        total_f1_score += loss_result['f1_score']
        total_score_loss += loss_result['score_loss']
        total_combined_loss += loss_result['total_loss']

        # 累计排序损失统计
        total_ranking_loss += ranking_loss['avg_ranking_loss']
        total_best_rank += ranking_loss['best_rank']
        total_worst_rank += ranking_loss['worst_rank']

# 输出总体统计
print(f"\n=== 总体段落选择统计 ===")
print(f"总样本数: {total_samples}")
print(f"平均重叠准确率: {total_overlap_accuracy / total_samples:.4f}")
print(f"平均精确率: {total_precision / total_samples:.4f}")
print(f"平均F1分数: {total_f1_score / total_samples:.4f}")
print(f"平均分数差异损失: {total_score_loss / total_samples:.4f}")
print(f"平均综合损失: {total_combined_loss / total_samples:.4f}")

print(f"\n=== 总体排序损失统计 ===")
print(f"平均排名损失: {total_ranking_loss / total_samples:.4f}")
print(f"平均最高排名: {total_best_rank / total_samples:.2f}")
print(f"平均最差排名: {total_worst_rank / total_samples:.2f}")

