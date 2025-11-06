import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
import pandas as pd

class MetallonBinding(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(MetallonBinding, self).__init__()
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
                prot_seq = str(row['aa_seq']).strip()
                result = str(int(row['label'])).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"""
Task: Determine whether this protein is a metalloprotein based on the provided sequence and protein name {name}.
Background: Metalloproteins are proteins that bind metal ions, often through specific amino acid residues such as histidine (H), cysteine (C), aspartate (D), or glutamate (E).
Question: Does this protein bind metal ions? Please choose one of the following options:  
0: Non-metalloprotein — This protein does **not** bind to any metal ions.   
1: Metalloprotein — This protein **binds** to one or more metal ions.
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