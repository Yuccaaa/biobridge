import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
import pandas as pd

class BindingDB(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(BindingDB, self).__init__()
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
                ligand_smiles = str(row['ligand']).strip()
                prot_seq = str(row['protein']).strip()
                result = str(row['ic50']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
                【Protein sequence (1-letter amino acid codes)】；{ligand_smiles}【Ligand structure (SMILES)】
                Task: Evaluate the inhibitory effect of the ligand on the given protein.
                Note: IC50 (half maximal inhibitory concentration) is the concentration of a substance required to inhibit 50% of the protein's activity. Lower IC50 values indicate stronger inhibition.
                Based on the provided protein and ligand, predict the inhibitory strength by classifying the IC50 level:  
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