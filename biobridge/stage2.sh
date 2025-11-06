#!/bin/bash

# 执行 stage2.py 训练脚本
export WANDB_BASE_URL=https://api.bandw.top
python stage2.py \
    --devices '0,1,2,3,4,5,6,7' \
    --mode 'train' \
    --filename 'stage2_continue_ablation_gb1.0' \
    --num_query_token 8 \
    --save_every_n_epochs 1 \
    --max_epochs 3 \
    --batch_size 32 \
    --precision 'bf16-mixed' \
    --num_workers 8 \
    --plm_model /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m \
    --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft \
    --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged \
    --llm_tune mid_lora \
    --init_checkpoint /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage2_07070513_2datasets_construct/epoch=09.ckpt/converted.ckpt \
    --use_wandb_logger \
    --caption_eval_epoch 3 \
    --dataset gb1
