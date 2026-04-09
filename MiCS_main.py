
import sys

sys.path.insert(0, '../transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
import pdb
import pickle
import argparse
import os
import gc
import statistics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")

parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, default="llama3-8b", help='llama2-7b or llama2-13b or llama3-8b')
parser.add_argument(
    '--dataset',
    type=str,
    default="mu",#hotpot wiki mu
    help='ragtruth, dolly'
)
args = parser.parse_args()
if args.dataset == "hotpot":
    if args.model_name == "llama2-7b":
        response_path = "./log/test_llama2_7B/hotpot_converted_10.jsonl"
    elif args.model_name == "llama2-13b":
        response_path = "./log/test_llama2_7B/hotpot_converted_10.jsonl"
    elif args.model_name == "llama3-8b":
        response_path = "./log/test_llama2_7B/hotpot_converted_10.jsonl"
if args.dataset == "wiki":
    if args.model_name == "llama2-7b":
        response_path = "./log/test_llama2_7B/wk_converted_10.jsonl"
    elif args.model_name == "llama2-13b":
        response_path = "./log/test_llama2_7B/wk_converted_10.jsonl"
    elif args.model_name == "llama3-8b":
        response_path = "./log/test_llama2_7B/wk_converted_10.jsonl"
if args.dataset == "mu":
    if args.model_name == "llama2-7b":
        response_path = "./log/test_llama2_7B/mu_converted_10.jsonl"
    elif args.model_name == "llama2-13b":
        response_path = "./log/test_llama2_7B/mu_converted_10.jsonl"
    elif args.model_name == "llama3-8b":
        response_path = "./log/test_llama2_7B/mu_converted_10.jsonl"
response = []
i=0
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # if data['source_id'] == '5a8b57f25542995d1e6f1371':
        #     print('12345')
        response.append(data)
        i+=1
        if i>=1000:
            break;
if args.dataset == "hotpot":
    if args.model_name == "llama2-7b":
        source_info_path = "./log/test_llama2_7B/hotpot_converted_10.jsonl"
    elif args.model_name == "llama2-13b":
        source_info_path = "./log/test_llama2_7B/hotpot_converted_10.jsonl"
    elif args.model_name == "llama3-8b":
        source_info_path = "./log/test_llama2_7B/hotpot_converted_10.jsonl"
if args.dataset == "wiki":
    if args.model_name == "llama2-7b":
        source_info_path = "./log/test_llama2_7B/wk_converted_10.jsonl"
    elif args.model_name == "llama2-13b":
        source_info_path = "./log/test_llama2_7B/wk_converted_10.jsonl"
    elif args.model_name == "llama3-8b":
        source_info_path = "./log/test_llama2_7B/wk_converted_10.jsonl"
if args.dataset == "mu":
    if args.model_name == "llama2-7b":
        source_info_path = "./log/test_llama2_7B/mu_converted_10.jsonl"
    elif args.model_name == "llama2-13b":
        source_info_path = "./log/test_llama2_7B/mu_converted_10.jsonl"
    elif args.model_name == "llama3-8b":
        source_info_path = "./log/test_llama2_7B/mu_converted_10.jsonl"

source_info_dict = {}
i=0
with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        # if data['source_id'] == '5a8b57f25542995d1e6f1371':
        #     print('54321')
        source_info_dict[data['source_id']] = data
        i+=1
        if i>=1000:
            break;

if args.model_name == "llama2-7b":
    model_name = "llama2/shakechen/Llama-2-7b-chat-hf"
elif args.model_name == "llama2-13b":
    model_name = "llama2/Llama-2-13b-chat-hf/ydyajyA/Llama-2-13b-chat-hf"
elif args.model_name == "llama3-8b":
    model_name = "llama2/LLM-Research/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    f"/hd/pengzewei/pro/sunhao_dai/PLMs/{model_name}",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(f"/hd/pengzewei/pro/sunhao_dai/PLMs/{model_name}")

device = torch.device("cuda:0")  # 注意这里要改为 0
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()
if args.model_name == "llama2-13bb":
    tokenizer_for_temp = AutoTokenizer.from_pretrained("/hd/pengzewei/pro/sunhao_dai/PLMs/llama2/llama-2-7b-chat-hf")
else:
    tokenizer_for_temp = tokenizer

if args.model_name == "llama2-7b":
    topk_head_path = "./log/test_llama2_7B/topk_heads.json"
elif args.model_name == "llama2-13b":
    topk_head_path = "./log/test_llama2_13B/topk_heads.json"
elif args.model_name == "llama3-8b":
    topk_head_path = "./log/test_llama3_8B/topk_heads.json"
else:
    print("model name error")
    exit(-1)

with open(topk_head_path, 'r') as f:  # 复制头
    # [(layer, head)...]
    copy_heads = json.load(f)


def summarize(obj, name="obj", depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{name}: ...")
        return
    if torch.is_tensor(obj):
        print(f"{indent}{name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{name}: {type(obj).__name__} len={len(obj)}")
        for i, x in enumerate(obj):
            summarize(x, f"[{i}]", depth + 1, max_depth)
    elif isinstance(obj, dict):
        print(f"{indent}{name}: dict keys={list(obj.keys())[:5]}{'...' if len(obj) > 5 else ''}")
        for k, v in list(obj.items())[:5]:
            summarize(v, f"[{k}]", depth + 1, max_depth)
    else:
        print(f"{indent}{name}: {type(obj).__name__}")


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):  # 计算两个分布之间的JSD
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)
    # # Fix bug: https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2 but for stable calculation, we maintain the original implementation of JSD.
    # kl1 = F.kl_div(M.log(), softmax_mature.unsqueeze(0),  reduction='none').mean(-1)
    # kl2 = F.kl_div(M.log(), softmax_anchor,  reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2)

    return js_divs.cpu().item() * 10e5


def calculate_ma_dist(sep_vocabulary_dist, sep_attention_dist):  # 计算两个分布之间的 曼哈顿距离
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    # 取绝对值
    abs_diff = torch.abs(dist_diff)

    # 计算 Manhattan 距离
    manhattan_distance = torch.sum(abs_diff)

    return manhattan_distance.cpu().item()


def is_hallucination_token(token_id, hallucination_spans):  # 检查token是否是幻觉
    for span in hallucination_spans:
        if token_id >= span[0] and token_id <= span[1]:
            return True
    return False


def calculate_hallucination_spans(response, text, response_rag, tokenizer, prefix_len):
    hallucination_span = []
    if "dolly" in source_info_path:
        return hallucination_span
    for item in response:
        start_id = item['start']
        end_id = item['end']
        start_text = text + response_rag[:start_id]
        end_text = text + response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        hallucination_span.append([start_id, end_id])
    return hallucination_span




select_response = []
data_type = "hotpot-qa"  # 无异议 但必须这么写，无论什么数据集

prompt_1 = '\nBriefly answer the following question:\n'
prompt_3 = '\nBear in mind that your response should be strictly based on the given context:\n'
prompt_4 = '\n\nIn case the passages do not contain the necessary information to answer the question, please reply with: \"Unable to answer based on given passages.\"\n'

# 最多处理 1000 个样本，为了减轻显存压力，内部自动按 500 一段分两批跑
total_loaded = len(response)
max_samples = min(1000, total_loaded)
chunk_size = 500
processed = 0

print(f"Total loaded samples: {total_loaded}, will process up to {max_samples}.")

for chunk_start in range(0, max_samples, chunk_size):
    chunk_end = min(chunk_start + chunk_size, max_samples)
    print(f"Processing samples from {chunk_start} to {chunk_end} / {max_samples}")

    for i in tqdm(range(chunk_start, chunk_end)):
        if response[i]['model'] == data_type and response[i]["split"] == "test":
            response_rag = response[i]['response']
            source_id = response[i]['source_id']
            temperature = response[i]['temperature']
            max_para = response[i]['max_para'] + 1

            # 对每一个 context 单独计算外部相似度
            for k in range(max_para):
                torch.cuda.empty_cache()

                context_key = f"context{k}"
                prompt = prompt_1 + response[i]['question'] + '\n' + prompt_3 + response[i][context_key] + prompt_4
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt[:12000]}
                ]
                text = tokenizer_for_temp.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                print(text)
                # 编码/解码示例
                s = text
                ids = tokenizer(s, return_tensors="pt").input_ids[0].tolist()

                input_text = text + response_rag
                print("all_text_len:", len(input_text))
                print("prompt_len", len(prompt))
                print("respond_len", len(response_rag))
                input_ids = tokenizer([input_text], return_tensors="pt").input_ids
                prefix_ids = tokenizer([text], return_tensors="pt").input_ids

                # 统一限制序列最大长度，避免不同模型爆显存
                # 为保持可比性，对所有模型都使用同一个上限
                max_seq_len = 1024
                if input_ids.shape[-1] > max_seq_len:
                    # 仅保留最后 max_seq_len 个 token，同时截断 prefix_ids，保证相对位置一致
                    input_ids = input_ids[:, -max_seq_len:]
                    if prefix_ids.shape[-1] > max_seq_len:
                        prefix_ids = prefix_ids[:, -max_seq_len:]
                if not input_ids.is_cuda:
                    input_ids = input_ids.to(device)
                if not prefix_ids.is_cuda:
                    prefix_ids = prefix_ids.to(device)
                continue_ids = input_ids[0, prefix_ids.shape[-1]:]  # todo 这边要改成幻觉 token 的起止位置
                # print(input_ids)
                # print(prefix_ids)
                # print(continue_ids)
                if "labels" in response[i].keys():
                    # 情况1：存在人工标注的幻觉信息
                    hallucination_spans = calculate_hallucination_spans(
                        response[i]['labels'], text, response_rag, tokenizer, prefix_ids.shape[-1]
                    )
                else:
                    # 情况2：没有人工标注的幻觉信息
                    hallucination_spans = []

                start_p, end_p = None, None
                if args.model_name == "llama2-7b":
                    start = 0
                    number = 32
                elif args.model_name == "llama3-8b":
                    start = 0
                    number = 32
                elif args.model_name == "llama2-13b":
                    start = 0
                    number = 40
                else:
                    print("model name error")

                with torch.no_grad():
                    logits_dict, outputs = model(
                        input_ids=input_ids,
                        return_dict=True,
                        output_attentions=True,
                        # 必须开启 hidden_states，模型内部依赖 outputs.hidden_states[knowledge_layer+1]
                        output_hidden_states=True,
                        knowledge_layers=list(range(start, number))
                    )

                # print('打印past_key_values，：', outputs.past_key_values)
                # print('打印hidden_states，：', outputs.hidden_states)
                # print('打印attentions，：', outputs.attentions)
                logits_dict = {key: [value[0].to(device), value[1].to(device)] for key, value in logits_dict.items()}

                # print('打印，logits_dict2：', logits_dict)
                # skip tokens without hallucination
                # 仅使用最后一层的 hidden_state
                last_hidden_states = outputs.hidden_states[-1][0, :, :]  # [seq_len, hidden_size]
                # todo 修改成 筛选 teacher focusing 的 token 和 model generate token 是否在 top_10内
                # probs = outputs['logits'][range(outputs["logits"].shape[0]), continue_ids].sum().item()
                # # ---------------------------------------------------------------------------------------------------------------
                external_similarity = []  # 这个用来存储生成的 token embedding 和 copy head 关注的 token embedding 的相似度得分
                parameter_knowledge_difference = []
                hallucination_label = []
                # 计算一下输入的 context 里面有没有 hallucination 词，如果有的话 copy 的时候把他们的 pointer weight 调小
                # input: input_ids, corr token vocab distribution
                # output: hallucination score for the input_ids or hallucination mask
                # outputs.attentions is a tuple, taking the last layer's attentions
                attentions_list = []
                for attentions_layer_id in range(len(outputs.attentions)):
                    for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):
                        if [attentions_layer_id, head_id] not in copy_heads:
                            continue
                        attentions_list.append({
                            "layer_head": (attentions_layer_id, head_id),
                            "attention_score": outputs.attentions[attentions_layer_id][:, head_id, :, :]
                        })

                # Step 1: Average the attention across the number of heads
                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):  # 便利模型相应（从提示词结束位置开始

                    # Step 2: Extract the non-zero values from the last row/column
                    # Now we gather the attention scores for the last token of each sequence
                    pointer_scores_list = [
                        attention_dict["attention_score"][:, seq_i, :] for attention_dict in attentions_list
                    ]  # shape: (batch_size, sequence_length)

                    # Step 3: Perform a softmax over the modified attention scores
                    # pointer_probs = nn.F.softmax(pointer_scores, dim=-1)  # shape: (batch_size, sequence_length)
                    if start_p is not None and end_p is not None:
                        pointer_probs_list = torch.cat(
                            [pointer_scores[:, start_p:end_p] for pointer_scores in pointer_scores_list], dim=0
                        )
                    else:
                        pointer_probs_list = torch.cat(
                            [pointer_scores[:, :prefix_ids.shape[-1]] for pointer_scores in pointer_scores_list],
                            dim=0
                        )  # shape: (batch_size, prefix_sequence_length) 截取这一步还是只让模型关注文本内容

                    # Step 4: select the top attented token
                    # Create an extended attention mask that masks out special tokens
                    # hyperparameter: token rate

                    # pointer_probs_list 是每个位置对应的大小(head_num, seq_len)，last_hidden_states shape (seq_len, hidden_state)是每个位置对应的 value，请取出 top 10% input_ids_cp 的 last_hidden_states，最终输出为(head_num, top10_len, hidden_state)
                    # 获取top 10%的索引
                    top_k = int(pointer_probs_list.shape[-1] * 0.1)  # 10% of sequence length
                    # 获取排序后的索引，按照概率从大到小排序
                    sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)
                    # 选择前top_k个索引
                    top_k_indices = sorted_indices[:, :top_k]
                    # 我们需要将 top_k_indices 展平，以便用于索引 last_hidden_states
                    flattened_indices = top_k_indices.flatten()  # shape (head_num * k,)
                    # 使用展平的索引在 last_hidden_states 中查找相应的 hidden_state
                    selected_hidden_states = last_hidden_states[flattened_indices]  # shape (head_num * k, hidden_state)
                    # 重新 reshape 成 (head_num, k, hidden_state)
                    top_k_hidden_states = selected_hidden_states.view(
                        top_k_indices.shape[0], top_k_indices.shape[1], -1
                    )
                    # 将其隐藏状态取mean-pooling均值
                    attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1)  # (head_num, hidden_state)
                    # Step 5: 计算最后一个token和前面文本的相似度Calculate the similarity between the last token and the attentioned prefix text
                    current_hidden_state = last_hidden_states[seq_i, :]  # shape (hidden_state,)

                    # 扩展 current_hidden_state 的形状以匹配 pointer_probs_list
                    current_hidden_state = current_hidden_state.unsqueeze(0).expand(attend_token_hidden_state.shape)

                    # 计算余弦相似度
                    cosine_similarity = F.cosine_similarity(
                        attend_token_hidden_state.to(device),
                        current_hidden_state.to(device),
                        dim=1
                    )
                    # 标记当前token是否为幻觉
                    if is_hallucination_token(seq_i, hallucination_spans):
                        hallucination_label.append(1)
                    else:
                        hallucination_label.append(0)
                    external_similarity.append(cosine_similarity.cpu().tolist())
                    # parameter_knowledge_difference.append(
                    #     [calculate_dist(value[0][0, seq_i, :], value[1][0, seq_i, :]) for value in logits_dict.values()])
                    torch.cuda.empty_cache()

                torch.cuda.empty_cache()
                ext_key = f"external_similarity{k}"
                ext_key_ave = f"external_similarity_avg{k}"
                max_key = f"external_similarity_max{k}"
                mid_key = f"external_similarity_mid{k}"
                last5_key = f"external_similarity_last5{k}"
                wei_key = f"external_similarity_wei{k}"

                response[i][ext_key] = external_similarity  # 外部评分E

                # 展平所有分数
                all_scores = []
                for token_scores in external_similarity:
                    all_scores.extend(token_scores)

                # 计算平均
                if len(all_scores) >= 5:
                    response[i][last5_key] = sum(all_scores[-5:]) / 5
                else:
                    response[i][last5_key] = sum(all_scores) / max(len(all_scores), 1)

                response[i][ext_key_ave] = sum(all_scores) / max(len(all_scores), 1)
                # 给深层更高的权重，长度与得分对齐
                weights = [idx + 1 for idx in range(len(all_scores))] if all_scores else [1]
                response[i][wei_key] = sum(x * w for x, w in zip(all_scores, weights)) / max(sum(weights), 1)
                response[i][max_key] = max(all_scores) if all_scores else 0.0
                response[i][mid_key] = statistics.median(all_scores) if all_scores else 0.0
                # 释放当前 context 相关的大张量，防止显存在循环中不断累积
                del outputs, last_hidden_states, logits_dict, attentions_list
                torch.cuda.empty_cache()
        # response[i]["parameter_knowledge_difference"] = parameter_knowledge_difference  # 参数评分P
        # response[i]["hallucination_label"] = hallucination_label  # 幻觉标签
        scores_with_k = []
        for k in range(max_para):
            key = f"external_similarity_avg{k}"
            if key in response[i]:
                scores_with_k.append((k, response[i][key]))

        # 按分数排序，取前2个
        top_2 = sorted(scores_with_k, key=lambda x: x[1], reverse=True)[:2]

        print("前2个最大平均分:")
        for rank, (k, score) in enumerate(top_2):
            context_key = f"context{k}"
            context_content = response[i].get(context_key, "")
            # 提取标题（假设格式是"标题: 内容"）
            title = context_content.split(":")[0] if ":" in context_content else f"段落{k}"
            print(f"第{rank + 1}名: context{k} ({title}), 平均分: {score:.4f}")
            top=f'top{rank+1}'
            response[i][top] =f"第{rank + 1}名: context{k} ({title}), 平均分: {score:.4f}"

        # 取所有层中的最大值

        select_response.append(response[i])
        processed += 1
        if processed >= max_samples:
            print(f"已经处理 {processed} 个样本，提前结束循环。")
            break

    if processed >= max_samples:
        break

# with open("./data/llama2_7B_response.json", "w") as f:
#     json.dump(select_response, f, indent=4, ensure_ascii=False)

if args.model_name == "llama2-7b":
    if args.dataset == "hotpot":
        save_path = "./log/test_llama2_7B/llama2_7B_response_vhp_1000.json"
    if args.dataset == "wiki":
        save_path = "./log/test_llama2_7B/llama2_7B_response_vwk_1000.json"
    if args.dataset == "mu":
        save_path = "./log/test_llama2_7B/llama2_7B_response_mu_1000.json"
elif args.model_name == "llama2-13b":
    if args.dataset == "hotpot":
        save_path = "./log/test_llama2_13B/llama2_13B_response_vhp_1000.json"
    if args.dataset == "wiki":
        save_path = "./log/test_llama2_13B/llama2_13B_response_vwk_1000.json"
    if args.dataset == "mu":
        save_path = "./log/test_llama2_13B/llama2_13B_response_mu_1000.json"
elif args.model_name == "llama3-8b":
    if args.dataset == "hotpot":
        save_path = "./log/test_llama3_8B/llama3_8B_response_vhp_1000.json"
    if args.dataset == "wiki":
        save_path = "./log/test_llama3_8B/llama3_8B_response_vwk_1000.json"
    if args.dataset == "mu":
        save_path = "./log/test_llama3_8B/llama3_8B_response_mu_1000.json"
else:
    print("model name error")
    exit(-1)

with open(save_path, "w") as f:
    json.dump(select_response, f, ensure_ascii=False)


