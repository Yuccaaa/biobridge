import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
import pandas as pd

class GO_BP(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(GO_BP, self).__init__()
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
                prot_seq = str(row['question']).strip()
                result = str(row['answer']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
【Task】Predict the biological processes involving the given protein.
【Background】Each process is represented by a GO-BP term (e.g., GO:0008150), describing a series of molecular events relevant to protein function.
【Output Format】List the predicted GO-BP terms, separated by commas, and wrap them in <answer> </answer> tags.  
Example: <answer>GO:0008150, GO:0009987, GO:0050896</answer>
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


class GO_CC(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(GO_CC, self).__init__()
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
                prot_seq = str(row['question']).strip()
                result = str(row['answer']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
【Task】Predict the cellular components associated with this protein.
【Background】Each location is represented by a GO-CC term (e.g., GO:0005737), indicating where the protein functions within the cell.
【Output Format】List the predicted GO-CC terms, separated by commas, and wrap them in <answer> </answer> tags.  
Example: <answer>GO:0005737, GO:0005829, GO:0005886</answer>
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


class GO_MF(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(GO_MF, self).__init__()
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
                prot_seq = str(row['question']).strip()
                result = str(row['answer']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
【Task】Predict the molecular functions performed by this protein.
【Background】Each function is represented by a GO-MF term (e.g., GO:0003677), describing specific biochemical activities of the protein.
【Output Format】List the predicted GO-MF terms, separated by commas, and wrap them in <answer> </answer> tags.  
Example: <answer>GO:0003677, GO:0005524, GO:0016787</answer>
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


class EC(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(EC, self).__init__()
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
                name = str(row['name']).strip()
                # 先按“-”分割，得到结构信息和 UniProt ID
                structure_part, uniprot_id = name.split('-')  # '3r7t_A', 'Q9PMG4'

                # 再按“_”分割结构信息，得到 PDB ID 和链ID
                pdb_id, chain_id = structure_part.split('_')  # '3r7t', 'A'
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
The information provided above is protein information, one of the chains of its crystal structure {pdb_id}, named {chain_id}, and numbered {uniprot_id} in the Uniprot sequence database.
Based on this information, the possible enzyme activity is inferred and the corresponding EC number is predicted.
【Output Format】List predicted EC numbers, separated by commas, wrapped in <answer> </answer> tags. 
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