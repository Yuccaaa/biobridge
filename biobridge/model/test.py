from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
llm_tokenizer = AutoTokenizer.from_pretrained("/oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300", use_fast=False, padding_side='right')
llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})

llm_model = AutoModelForCausalLM.from_pretrained("/oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300", torch_dtype=torch.bfloat16)
llm_model.resize_token_embeddings(len(llm_tokenizer))

text = "You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER> Based on the following protein\'s amino acid sequence, is the protein located on the membrane? Swiss-Prot description for P86987,Options:\n0.Yes\n1.No"

# Step 1: 编码成 input_ids 和 attention_mask（注意要 tensor 类型）
inputs = llm_tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]             # shape: [1, L]
attention_mask = inputs["attention_mask"]   # shape: [1, L]

# Step 2: 通过模型 embedding 层获取 inputs_embeds
with torch.no_grad():
    inputs_embeds = llm_model.get_input_embeddings()(input_ids)


outputs = llm_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
   
    max_length=128,
    min_length=1,
   
    use_cache=True,
    cache_implementation="hybrid"
)
output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
output_text = [text.strip() for text in output_text]
print(output_text)