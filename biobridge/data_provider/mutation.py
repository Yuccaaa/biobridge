import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
import pandas as pd

class TAPE_Stability(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(TAPE_Stability, self).__init__()
        self.data_path = data_path
        self.user_prompt = prompt
        self.return_prompt = return_prompt
        
        self.data_list = self._load_and_preprocess(self.data_path)
        self.text2id = self._build_text_vocab()

    def _load_and_preprocess(self, data_path):
        data_list = []
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            try:
                
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = """
【Task】Predict the thermostability score of the given protein sequence, which reflects its ability to maintain proper folding above a concentration threshold.
【Background】Protein stability is an important biophysical property indicating a protein’s resistance to denaturation or unfolding under thermal or chemical stress. In this task, each protein is evaluated by a numerical stability score, where higher values indicate greater ability to remain folded under extreme conditions. This score serves as a proxy for the protein’s intrinsic stability.
【Question】What is the predicted stability score for this sequence?
【Output Format】You must return only the score number, wrapped in <answer></answer> tags.  
"""
                if self.user_prompt:
                    prompt += self.user_prompt

                # extra可以返回原始feather字符串，也可以返回feather_vals
                 # 或 feather_raw
                data_list.append((prot_seq, text_seq, prompt))
            except Exception as e:
                print(f"警告: 跳过有问题的行: {row}，原因: {e}")
        return data_list

    def _build_text_vocab(self):
        text2id = {}
        for _, text_seq, _ in self.data_list:
            if text_seq not in text2id:
                text2id[text_seq] = len(text2id)
        return text2id

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq, prompt = self.data_list[index]
        if self.return_prompt:
            return prot_seq, prompt, text_seq,index
        return prot_seq, text_seq, index

class TAPE_Fluorescence(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(TAPE_Fluorescence, self).__init__()
        self.data_path = data_path
        self.user_prompt = prompt
        self.return_prompt = return_prompt
        
        self.data_list = self._load_and_preprocess(self.data_path)
        self.text2id = self._build_text_vocab()

    def _load_and_preprocess(self, data_path):
        data_list = []
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            try:
                
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = """
【Task】Predict the log fluorescence intensity of the given protein sequence.
【Output Format】You must return only the numerical value, wrapped in <answer></answer> tags.  
"""
# 【Background】Fluorescence intensity reflects how strongly a protein emits light when excited by a specific wavelength. It is commonly measured in protein variants such as GFP (Green Fluorescent Protein) mutants. The log-transformed fluorescence value quantifies the brightness on a logarithmic scale. Mutations in the sequence can increase or decrease fluorescence intensity.
# 【Question】What is the predicted log fluorescence intensity for this sequence?
                if self.user_prompt:
                    prompt += self.user_prompt

                # extra可以返回原始feather字符串，也可以返回feather_vals
                 # 或 feather_raw
                data_list.append((prot_seq, text_seq, prompt))
            except Exception as e:
                print(f"警告: 跳过有问题的行: {row}，原因: {e}")
        return data_list

    def _build_text_vocab(self):
        text2id = {}
        for _, text_seq, _ in self.data_list:
            if text_seq not in text2id:
                text2id[text_seq] = len(text2id)
        return text2id

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq, prompt = self.data_list[index]
        if self.return_prompt:
            return prot_seq, prompt, text_seq,index
        return prot_seq, text_seq, index