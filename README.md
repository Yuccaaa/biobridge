# BioBridge: Bridging Proteins and Language for Enhanced Biological Reasoning with LLMs

## Paper Introduction
BioBridge is a domain-adaptive continual pretraining framework designed to fuse the advantages of Protein Language Models (PLMs) and general-purpose Large Language Models (LLMs). It addresses two core challenges in biological reasoning:  
1. The **biological knowledge barrier** of general LLMs (lack of domain-specific protein understanding).  
2. The **poor generalization** of specialized PLMs (limited adaptability to multi-task scenarios).  

Key innovations of BioBridge include:  
- **Domain-Incremental Continual Pre-training (DICP)**: Infuses biomedical knowledge into LLMs via specialized corpora (e.g., biology textbooks, PubMed articles) while mitigating catastrophic forgetting of general language capabilities.  
- **PLM-Projector Module**: Uses ESM2 (a state-of-the-art PLM) as a protein encoder and a cross-modal projector to map protein sequence embeddings into the LLM’s semantic space, enabling effective protein-text alignment.  
- **End-to-End Optimization**: Unifies pre-training and alignment stages to support multi-task biological reasoning (e.g., protein property prediction, knowledge question-answering) without task-specific retraining.  

Extensive experiments validate that BioBridge performs comparably to mainstream PLMs (e.g., ESM2) on protein benchmarks (PFMBench) and maintains strong general language capabilities on datasets like MMLU and RACE, showcasing its unique value in balancing domain adaptability and general reasoning competency.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Yuccaaa/biobridge.git
   cd biobridge
   ```

2. Install dependencies:
python=3.10
   ```bash
   pip install flash_attn-2.5.8-cp310-cp310-linux_x86_64.whl  # 替换为实际文件名
   ```bash
Install Lavis: pip install rouge_score nltk salesforce-lavis
Install others: pip install -U transformers pytorch-lightning



## Data
The training data for BioBridge integrates multiple sources to ensure comprehensive biomedical coverage and general reasoning retention. For detailed data collection, preprocessing pipelines, and format specifications, refer to the **Materials and Methods** section of the original paper. Key data sources include:  
- Biomedical corpora: Biology textbooks, PubMed Central articles/abstracts, sequence-augmented sentences (via BERN2 named entity recognition).  
- Protein-text pairs: 90K Swiss-Prot entries, 422K OntoProtein pairs (covering molecular functions, biological processes).  
- General reasoning data: Mixture of Thoughts (MoT) corpus (93K math, 83K code, 173K scientific problems) to prevent catastrophic forgetting.  


## Usage
Modify experimental settings (model hyperparameters, data paths, training configurations) in `configure.py` before running the code. The framework supports three core training stages, as outlined below:

### 1. Domain-Incremental Continual Pre-training (DICP)
This stage adapts the base LLM (Qwen2.5-7B-Instruct) to biomedical data while preserving general language capabilities.  
The pre-training implementation for this project is based on ModelScope's SWIFT (Scalable Lightweight Framework for Tuning) framework.
SWIFT Framework Repository: https://github.com/modelscope/swift


### 2. PLM-Projector Cross-Modal Alignment
Uses ESM2 (frozen protein encoder) and Q-Former to align protein embeddings with the LLM’s semantic space via contrastive learning.  
- Run command:
  ```bash
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

  ```


### 3. End-to-End Fine-Tuning
Unifies the pre-trained LLM and alignment module to enable multi-task biological reasoning (no downstream task-specific data required).  
- Run command:
  ```bash
 python stage2.py 
   --devices '0,1,2,3,4,5,6,7' 
   --mode train 
   --filename stage2_07301646_2datasets_construct 
   --num_query_token 8  
   --save_every_n_epochs 2 
   --max_epochs 10 
   --batch_size 4 
   --precision 'bf16-mixed' 
   --num_workers 8 
   --plm_model  /nas/shared/kilab/wangyujia/ProtT3/plm_model/esm2-150m 
   --bert_name /nas/shared/kilab/wangyujia/ProtT3/plm_model/microsoft 
   --llm_name /oss/wangyujia/BIO/construction_finetuning/alpaca/v1-20250609-141541/checkpoint-50-merged  
   --llm_tune mid_lora  
   --stage1_path /nas/shared/kilab/wangyujia/ProtT3/all_checkpoints/stage1_07041727_2dataset/epoch=29.ckpt/converted.ckpt  
   --use_wandb_logger 
   --dataset swiss-prot
  ```


### Model Weights
Pretrained and fine-tuned model weights are available for download at:  
https://huggingface.co/yuccaaa/biobridge

