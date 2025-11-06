#!/bin/bash

# 直接运行stage1.py的简化脚本

# 检查必要的文件是否存在
if [ ! -f "stage1.py" ]; then
    echo "错误: stage1.py 文件不存在"
    exit 1
fi

# 设置环境变量（可选）
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像加速模型下载

# 定义参数
DEVICES="'0,1,2,3,4,5,6,7'"
MODE="train"
FILENAME="stage1_ckpt"
NUM_QUERY_TOKEN=8
PLM_NAME="/nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m"
BERT_NAME="/nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft"
SAVE_EVERY=10
BATCH_SIZE=32
PRECISION="bf16-mixed"
MIX_DATASET="True"
NUM_WORKERS=8
STRATEGY="ddp"

# 运行命令
python stage1.py \
    --devices $DEVICES \
    --mode $MODE \
    --filename $FILENAME \
    --num_query_token $NUM_QUERY_TOKEN \
    --plm_name $PLM_NAME \
    --bert_name $BERT_NAME \
    --save_every_n_epochs $SAVE_EVERY \
    --batch_size $BATCH_SIZE \
    --precision $PRECISION \
    --mix_dataset \
    --num_workers $NUM_WORKERS \
    --strategy $STRATEGY \
    --use_wandb_logger

#python stage1.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage1_ckpt --num_query_token 8 --plm_name  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft   --save_every_n_epochs 5 --max_epochs 20 --batch_size 32 --precision 'bf16-mixed' --mix_dataset  --num_workers 8 --use_wandb_logger

python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_test --num_query_token 8  --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300  --llm_tune mid_lora --enable_flash --mix_dataset --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_06290009_deepspeed/epoch=19.ckpt/converted.ckpt --use_wandb_logger
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_test --num_query_token 8  --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300  --llm_tune mid_lora  --mix_dataset --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_06290009_deepspeed/epoch=19.ckpt/converted.ckpt --use_wandb_logger
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_test --num_query_token 8  --save_every_n_epochs 5 --max_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300  --llm_tune mid_lora  --mix_dataset --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_06290009_deepspeed/epoch=19.ckpt/converted.ckpt --use_wandb_logger
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_continue_deeplocmulti --num_query_token 8  --save_every_n_epochs 5 --max_epochs 25 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  --llm_tune mid_lora   --init_checkpoint /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07070513_2datasets_construct/epoch=09.ckpt/converted.ckpt  --use_wandb_logger
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_continue_deeplocmulti_07141239 --num_query_token 8  --save_every_n_epochs 1 --max_epochs 3 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  --llm_tune mid_lora   --init_checkpoint /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07070513_2datasets_construct/epoch=09.ckpt/converted.ckpt  --use_wandb_logger --caption_eval_epoch 3 
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_continue_ablation_fluorescence --num_query_token 8  --save_every_n_epochs 10 --max_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /nas/shared/kilab/hf-hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28  --llm_tune mid_lora   --init_checkpoint /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_08011616_2datasets_withoutpretrain/epoch=09.ckpt/converted.ckpt  --caption_eval_epoch 10 --dataset fluorescence
export WANDB_BASE_URL=https://api.bandw.top

python stage3.py --devices '0,1,2,3,4,5,6,7'   --filename prot_qa --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset deeplocbinary  --prompt " " --inference_batch 32 --max_inference_len 36  --checkpoint_name  /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_06301657/last.ckpt/converted.ckpt
python stage3.py --devices '0,1,2,3,4,5,6,7'  --filename stage3  --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset deeplocbinary  --prompt " " --inference_batch 32 --max_inference_len 36  --checkpoint_name  /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_06301657/last.ckpt/converted.ckpt --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300
python stage3.py --devices '0,1,2,3,4,5,6,7'  --filename stage3  --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset swissprot  --prompt "You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"  --inference_batch 32 --max_inference_len 128  --checkpoint_name  /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07021249/epoch=09.ckpt/converted.ckpt  --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300
accelerate launch stage3.py --devices '0,1,2,3,4,5,6,7'  --filename stage3  --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset deeplocbinary  --prompt "You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"  --inference_batch 32 --max_inference_len 128  --checkpoint_name  /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_06290009_deepspeed/epoch=19.ckpt/converted.ckpt  --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300
python stage3.py --devices '0,1,2,3,4,5,6,7'  --filename stage3  --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset deeplocbinary  --prompt "You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"  --inference_batch 32 --max_inference_len 128  --checkpoint_name  /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_06290009_deepspeed/epoch=19.ckpt/converted.ckpt  --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300
python stage3.py --devices '0,1,2,3,4,5,6,7'  --filename stage3  --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset deeplocbinary  --prompt "You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"  --inference_batch 32 --max_inference_len 512  --checkpoint_name  /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07041521/epoch=14.ckpt/converted.ckpt  --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/pretrain_output/qwen2.5-7b-instruct-bio/bio_all/save1epoch/checkpoint-1300 --text_max_len 4096 --prot_max_len 4096
python stage3.py --devices '0,1,2,3,4,5,6,7'  --filename stage3  --num_query_token 8   --num_workers 8  --precision 'bf16-mixed'  --dataset empty  --inference_batch 32 --max_inference_len 512  --checkpoint_name /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07070513_2datasets_construct/epoch=09.ckpt/converted.ckpt   --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged

python convert.py --input /path/to/stage1/ckpt/address


python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_continue_empty _07141239 --num_query_token 8  --save_every_n_epochs 1 --max_epochs 3 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  --llm_tune mid_lora   --init_checkpoint /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07070513_2datasets_construct/epoch=09.ckpt/converted.ckpt  --caption_eval_epoch 3 

训练第二阶段：
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_07301646_2datasets_construct --num_query_token 8  --save_every_n_epochs 2 --max_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  --llm_tune mid_lora  --mix_dataset --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_07041727_2dataset/epoch=29.ckpt/converted.ckpt  --use_wandb_logger
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_07301646_2datasets_construct --num_query_token 8  --save_every_n_epochs 2 --max_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  --llm_tune mid_lora  --mix_dataset --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_07041727_2dataset/epoch=29.ckpt/converted.ckpt  --use_wandb_logger
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_07301646_2datasets_construct --num_query_token 8  --save_every_n_epochs 2 --max_epochs 10 --batch_size 4 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  --llm_tune mid_lora  --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_07041727_2dataset/epoch=29.ckpt/converted.ckpt  --use_wandb_logger --dataset swiss-prot
python stage2.py --devices '0,1,2,3,4,5,6,7' --mode train --filename stage2_08011616_2datasets_qweninstruct --num_query_token 8  --save_every_n_epochs 2 --max_epochs 10 --batch_size 4 --precision 'bf16-mixed' --num_workers 8 --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft --llm_name /nas/shared/kilab/hf-hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28  --llm_tune mid_lora  --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_07041727_2dataset/epoch=29.ckpt/converted.ckpt  --use_wandb_logger --dataset swiss-prot