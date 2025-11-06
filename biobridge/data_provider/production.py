
import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
import pandas as pd
class Antibiotic_Resistance(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(Antibiotic_Resistance, self).__init__()
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
 
                prompt = f"""
【Task】Predict the antibiotic resistance class of the given protein.
【Background】Antibiotic resistance refers to the ability of bacteria or other microbes to resist the effects of antibiotics that were once effective against them. Each protein is associated with resistance to exactly one of 19 antibiotic classes.

【Prediction Goal】Based on the provided protein sequence, determine which single antibiotic class (from 1 to 19) this protein confers resistance to.

【Output Format】Return only one predicted resistance class (a number from 1 to 19), wrapped in <answer> </answer> tags.  
Example: <answer>7</answer>
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

class Thermostability(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(Thermostability, self).__init__()
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
                #name = str(row['name']).strip()
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()
                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
【Task】Predict the thermostability value of the given protein.
【Background】Thermostability refers to the ability of a molecule to resist irreversible chemical or physical changes at high temperatures, such as decomposition or aggregation.
【Output Format】Provide the predicted thermostability as a numeric value (e.g., melting temperature in °C). Wrap your answer in <answer></answer> tags.  

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



class Material(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(Material, self).__init__()
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
                #name = str(row['name']).strip()
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()
                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
【Task】Determine whether the given material can be successfully produced.
【Background】In materials science, certain chemical compounds or materials may or may not be synthesizable (i.e., producible) under realistic experimental conditions. This task requires classifying whether the input material composition and structure allow for successful production. This is a binary classification problem.
【Question】Can this material be successfully produced?
【Output Format】Respond with either "1" or "0", and wrap your answer in <answer></answer> tags.  

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



class Clone(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(Clone, self).__init__()
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
                #name = str(row['name']).strip()
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()
                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""

【Task】Determine whether the given protein sequence can be successfully cloned.
【Background】In molecular biology, cloning refers to the process of creating copies of a DNA or protein sequence. Some sequences can be challenging to clone due to their length, GC-content, secondary structures, or toxicity to the host. This task requires predicting whether the given protein sequence is likely to be successfully cloned. This is a binary classification problem.
【Question】Can this protein sequence be successfully cloned?
【Output Format】Respond with either "1" or "0", and wrap your answer in <answer></answer> tags.  
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
