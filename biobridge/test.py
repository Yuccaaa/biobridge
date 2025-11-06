# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # 1. 定义模型名称
# #model_name = "meta-llama/Llama-2-7b-chat-hf" # 如果要用Llama2，确保已登录Hugging Face
# model_name = "/nas/shared/kilab/wangyujia/ProtT3/llm_model" # 推荐这个模型，加载更快

# # 如果需要使用授权模型（如Llama2），请确保已登录Hugging Face
# # from huggingface_hub import login
# # login() # 会提示你输入Hugging Face Token

# print(f"正在加载模型: {model_name}...")

# # 2. 加载分词器 (Tokenizer)
# # trust_remote_code=True 参数对于某些自定义模型（如Qwen）可能是必需的
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# # 3. 加载大型语言模型 (LLM)
# # device_map="auto" 会自动将模型加载到可用的GPU上，如果没有GPU则加载到CPU
# # torch_dtype=torch.bfloat16 (或 torch.float16) 可以节省显存，并加速推理
# # quantization_config (bitsandbytes) 用于进行4bit/8bit量化，进一步节省显存
# try:
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16, # 或者 torch.float16
    #     trust_remote_code=True,
    #     # 如果你想进行量化，可以取消下面几行的注释
    #     # load_in_4bit=True, # 加载为4bit量化模型，需要安装 bitsandbytes 和 accelerate
    #     # quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    # )
#     print("模型加载成功！")
# except Exception as e:
#     print(f"模型加载失败: {e}")
#     print("请检查模型名称是否正确，以及是否有权限访问模型。")
#     print("如果是Llama模型，请确认已在Hugging Face上接受协议并登录。")
#     exit()

# # 4. 设置模型为评估模式 (inference mode)
# model.eval()

# # 5. 定义一个输入提示
# # 对于 instruct 模型，通常需要遵循特定的对话格式（如 ChatML, Llama-2 Chat等）
# # 这里使用 tokenizer.apply_chat_template 来自动格式化输入
# messages = [
#     {"role": "user", "content": "According to the protein information provided below and the protein name [Q9H400], predict the most likely subcellular localization from the following options:\nOptions: 0. \"Nucleus, U\" \n 1. \"Cytoplasm, S\"  \n 2. \"Extracellular, S\"  \n 3. \"Mitochondrion, U\"  \n 4. \"Cell membrane, M\"  \n 5. \"Endoplasmic reticulum, M\"  \n 6. \"Plastid, S\"  \n 7. \"Golgi apparatus, M\"  \n  8. \"Lysosome/Vacuole, M\"  \n9. \"Peroxisome, U\"\n\nPlease directly provide the text of the most likely localization option.You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER> MGLPVSWAPPALWVLGCCALLLSLWALCTACRRPEDAVAPRKRARRQRARLQGSATAAEASLLRRTHLCSLSKSDTRLHELHRGPRSSRALRPASMDLLRPHWLEVSRDITGPQAAPSAFPHQELPRALPAAAATAGCAGLEATYSNVGLAALPGVSLAASPVVAEYARVQKRKGTHRSPQEPQQGKTEVTPAAQVDVLYSRVCKPKRRDPGPTTDPLDPKGQGAILALAGDLAYQTLPLRALDVDSGPLENVYESIRELGDPAGRSSTCGAGTPPASSCPSLGRGWRPLPASLP"},
# ]

# # 将消息列表转换为模型输入的 token ID
# # add_special_tokens=False 通常用于多轮对话，避免重复添加
# # tokenize_config.add_bos_token = True 也可以尝试
# input_ids = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True, # 对于指令模型，这会添加生成所需的特定提示
#     return_tensors="pt"
# ).to(model.device) # 将输入移动到模型所在的设备 (GPU/CPU)

# print("\n--- 开始生成文本 ---")
# # 6. 生成文本
# # max_new_tokens: 生成的最大新token数量
# # num_beams: Beam search 的束宽，>1 表示使用 beam search
# # do_sample: 是否使用采样（如果为True，则num_beams通常为1，并结合temperature, top_p等）
# # top_k, top_p, temperature: 采样策略参数
# # pad_token_id, eos_token_id: 确保分词器有正确的填充和结束标志
# # 通常 model.config.pad_token_id 和 model.config.eos_token_id 是好的默认值

# # 确保 pad_token_id 和 eos_token_id 设置正确
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id # 常用做法，如果没有pad_token_id就用eos_token_id

# with torch.no_grad(): # 推理时不需要计算梯度
#     output_ids = model.generate(
#         input_ids,
#         max_new_tokens=200, # 最大生成200个新token
#         num_beams=1, # 禁用 beam search，使用贪婪解码或采样
#         do_sample=True, # 启用采样，让生成内容更丰富
#         temperature=0.7, # 采样温度
#         top_p=0.9, # Nucleus sampling
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

# # 7. 解码生成的 token ID 为文本
# # skip_special_tokens=True 会跳过像 <bos>, <eos>, <pad> 这样的特殊 token
# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# print("\n--- 生成结果 ---")
# print(generated_text)

# print("\n--- 测试完成 ---")

# import json

# def compute_accuracy_from_file(filepath):
#     correct = 0
#     total = 0
    
#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             data = json.loads(line)
#             # 提取第一个数字（假设都是整数）
#             try:
#                 target = int(str(data["targets"]).strip().splitlines()[0])
#                 pred   = int(str(data["predictions"]).strip().splitlines()[0])
#             except (KeyError, ValueError, IndexError):
#                 continue
            
#             total += 1
#             if pred == target:
#                 correct += 1

#     if total == 0:
#         return 0.0
#     return correct / total

# if __name__ == "__main__":
#     filepath = "/nas/shared/kilab/wangyujia/ProtT3/results/metallonbinding_07260343_predictions.txt"  # 替换为你的文件路径
#     acc = compute_accuracy_from_file(filepath)
#     print(f"Accuracy: {acc*100:.2f}% ({int(acc*100)}%)")

# ##accuracy
# import json
# import re

# def compute_accuracy_from_file(filepath):
#     total, correct = 0, 0
#     pattern = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE)

#     with open(filepath, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             data = json.loads(line)

#             tgt_match = pattern.search(data.get("targets", ""))
#             pred_match = pattern.search(data.get("predictions", ""))

#             if not tgt_match or not pred_match:
#                 continue
            
#             try:
#                 tgt_val = int(tgt_match.group(1).strip())
#                 pred_val = int(pred_match.group(1).strip())

#             except ValueError:
#                 continue  # 如果无法转换为int，跳过此条


#             total += 1
#             if pred_val == tgt_val:
#                 correct += 1

#     if total == 0:
#         return 0.0, 0
#     return correct / total, total

# if __name__ == "__main__":
#     filepath = "/nas/shared/kilab/wangyujia/ProtT3/results/ablation_material_predictions.txt"
#     acc, count = compute_accuracy_from_file(filepath)
#     print(f"Checked {count} items. Accuracy: {acc*100:.3f}%")


##spearman
import json
import re
from scipy.stats import spearmanr

# 文件路径
file_path = '/nas/shared/kilab/wangyujia/ProtT3/results/ablation_gb1_predictions.txt'  # 替换为你的文件路径

# 提取 <answer>数值</answer> 的正则
pattern = re.compile(r"<answer>(.*?)</answer>")

y_true = []
y_pred = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # 提取 targets
            target_match = pattern.search(data["targets"])
            pred_match = pattern.search(data["predictions"])
            if target_match and pred_match:
                y_true.append(float(target_match.group(1)))
                y_pred.append(float(pred_match.group(1)))
        except Exception as e:
            print(f"Error processing line: {e}")
            continue

# 计算 Spearman 相关系数
if len(y_true) > 1:
    rho, p_value = spearmanr(y_true, y_pred)
    print(f"Spearman correlation coefficient: {rho:.5f}, p-value: {p_value:.4e}")
else:
    print("Not enough data to compute Spearman correlation.")


##f1
# import json

# def parse_answer(ans_str):
#     ans_str = ans_str.strip()
#     # 提取<answer>标签中的内容
#     if ans_str.startswith("<answer>") and ans_str.endswith("</answer>"):
#         ans_str = ans_str[len("<answer>"): -len("</answer>")].strip()
#     if ans_str == "":
#         return set()
#     return set(ans_str.split(','))

# # 统计指标
# TP_total, FP_total, FN_total = 0, 0, 0
# f1_list = []

# with open('/nas/shared/kilab/wangyujia/ProtT3/results/ec_07262004_predictions.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line.strip())
#         target_set = parse_answer(data['targets'])
#         pred_set = parse_answer(data['predictions'])

#         TP = len(target_set & pred_set)
#         FP = len(pred_set - target_set)
#         FN = len(target_set - pred_set)

#         # 累计总数（用于micro F1）
#         TP_total += TP
#         FP_total += FP
#         FN_total += FN

#         # 计算每条数据的 Precision, Recall, F1
#         precision = TP / (TP + FP) if TP + FP > 0 else 0
#         recall = TP / (TP + FN) if TP + FN > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
#         if f1==1:
#             print(f1)
#         f1_list.append(f1)

# # Macro-F1: 所有样本F1的平均
# macro_f1 = sum(f1_list) / len(f1_list)

# # Micro-F1: 总体TP/FP/FN累加后计算
# precision_micro = TP_total / (TP_total + FP_total) if TP_total + FP_total > 0 else 0
# recall_micro = TP_total / (TP_total + FN_total) if TP_total + FN_total > 0 else 0
# micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if precision_micro + recall_micro > 0 else 0

# print(f"Macro F1-score: {macro_f1:.4f}")
# print(f"Micro F1-score: {micro_f1:.4f}")
