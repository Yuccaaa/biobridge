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
   ```bash
   pip install -r requirements.txt
   ```


## Data
The training data for BioBridge integrates multiple sources to ensure comprehensive biomedical coverage and general reasoning retention. For detailed data collection, preprocessing pipelines, and format specifications, refer to the **Materials and Methods** section of the original paper. Key data sources include:  
- Biomedical corpora: Biology textbooks, PubMed Central articles/abstracts, sequence-augmented sentences (via BERN2 named entity recognition).  
- Protein-text pairs: 90K Swiss-Prot entries, 422K OntoProtein pairs (covering molecular functions, biological processes).  
- General reasoning data: Mixture of Thoughts (MoT) corpus (93K math, 83K code, 173K scientific problems) to prevent catastrophic forgetting.  

Data preprocessing details (e.g., token sequence truncation/concatenation to 4096 tokens) are provided in the paper’s Section II.C.


## Usage
Modify experimental settings (model hyperparameters, data paths, training configurations) in `configure.py` before running the code. The framework supports three core training stages, as outlined below:

### 1. Domain-Incremental Continual Pre-training (DICP)
This stage adapts the base LLM (Qwen2.5-7B-Instruct) to biomedical data while preserving general language capabilities.  
- Reference implementation: `pretrain/dicp_pretrain.py`  
- Run command:
  ```bash
  python pretrain/dicp_pretrain.py \
    --config configure.py \
    --batch_size 32 \
    --epochs 1 \
    --lr 1e-5 \
    --data_dir ./data/dicp_corpus
  ```


### 2. PLM-Projector Cross-Modal Alignment
Uses ESM2 (frozen protein encoder) and Q-Former to align protein embeddings with the LLM’s semantic space via contrastive learning.  
- Run command:
  ```bash
  python train/plm_projector_alignment.py \
    --config configure.py \
    --dataset_path ./data/onto_protein_swissProt \
    --epochs 30 \
    --num_gpus 8 \
    --contrastive_tau 0.07
  ```


### 3. End-to-End Fine-Tuning
Unifies the pre-trained LLM and alignment module to enable multi-task biological reasoning (no downstream task-specific data required).  
- Run command:
  ```bash
  python train/end2end_finetune.py \
    --config configure.py \
    --train_data ./data/swissProt_pairs \
    --output_dir ./checkpoints \
    --batch_size 16 \
    --epochs 5
  ```


### Model Weights
Pretrained and fine-tuned model weights are available for download at:  
https://huggingface.co/yuccaaa/biobridge

