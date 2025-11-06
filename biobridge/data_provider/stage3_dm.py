# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
import pandas as pd


class Stage2Collater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        
    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, _ = zip(*batch)
        prot_tokens = self.prot_tokenizer(prot_seqs,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.prot_max_len,
                                          return_tensors="pt",
                                          return_attention_mask=True, 
                                          return_token_type_ids=False)
        if False:
            self.tokenizer.padding_side = 'left'
            prompt_tokens = self.tokenizer(prompt_seqs,
                                        truncation=True,
                                        padding='longest',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True, 
                                        return_token_type_ids=False)
            self.tokenizer.padding_side = 'right'
            text_tokens = self.tokenizer(text_seqs,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True, 
                                        return_token_type_ids=False)
        else:
            self.tokenizer.padding_side = 'left'
            prompt_tokens = self.tokenizer(prompt_seqs,
                                           truncation=True,
                                           padding='longest',
                                           add_special_tokens=True,
                                           max_length=self.text_max_len,
                                           return_tensors='pt',
                                           return_attention_mask=True, 
                                           return_token_type_ids=False)
            max_prompt_len = int(prompt_tokens.attention_mask.sum(dim=1).max())
            input_pair = [[p, t] for p, t in zip(prompt_seqs, text_seqs)]
            input_tokens = self.tokenizer(input_pair,
                                          truncation=True,
                                          padding='max_length',
                                          add_special_tokens=True,
                                          max_length=self.text_max_len + max_prompt_len,
                                          return_tensors='pt',
                                          return_attention_mask=True,
                                          return_token_type_ids=True)
        return prot_tokens, input_tokens


class InferenceCollater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        
    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, indices = zip(*batch)
        self.tokenizer.padding_side = 'right'
        prompt_tokens = self.tokenizer(prompt_seqs,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=True,
                                       max_length=self.text_max_len,
                                       return_tensors='pt',
                                       return_attention_mask=True, 
                                       return_token_type_ids=False)
        prot_tokens = self.prot_tokenizer(prot_seqs,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.prot_max_len,
                                          return_tensors="pt",
                                          return_attention_mask=True, 
                                          return_token_type_ids=False)
        target_dict = {'targets': text_seqs, 'indices': indices}
        return prot_tokens, prompt_tokens, target_dict


class Stage3DM(LightningDataModule):
    def __init__(
        self,
        dataset: str = 'deeplocbinary',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        self.prompt = args.prompt

        if dataset=='deeplocbinary':
            self.train_dataset = DeepLocBinaryDataset('/oss/wangyujia/pretrain-bench/locate/deeplocbinary/train.csv', prompt=self.prompt, return_prompt=True)
            self.val_dataset = DeepLocBinaryDataset('/oss/wangyujia/pretrain-bench/locate/deeplocbinary/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = DeepLocBinaryDataset('/oss/wangyujia/pretrain-bench/locate/deeplocbinary/test.csv', prompt=self.prompt, return_prompt=True)
        elif dataset=='deeplocmulti':
            self.train_dataset = DeepLocMultiDataset('/oss/wangyujia/pretrain-bench/locate/deeplocmulti/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = DeepLocMultiDataset('/oss/wangyujia/pretrain-bench/locate/deeplocmulti/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = DeepLocMultiDataset('/oss/wangyujia/pretrain-bench/locate/deeplocmulti/test.csv', prompt=self.prompt, return_prompt=True)
        elif dataset=='swissprot':
            self.train_dataset = SwissProtDataset('/nas/shared/kilab/wangyujia/ProtT3/data/SwissProtV3/train_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
            self.val_dataset = SwissProtDataset('/nas/shared/kilab/wangyujia/ProtT3/data/SwissProtV3/test_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
            self.test_dataset = SwissProtDataset('/nas/shared/kilab/wangyujia/ProtT3/data/SwissProtV3/valid_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
        elif dataset=='molinstruction':
            self.train_dataset = MolInstructionDataset('/oss/wangyujia/pretrain-bench/mol-instruction/train.jsonl', prompt='', return_prompt=True)
            self.val_dataset = MolInstructionDataset('/oss/wangyujia/pretrain-bench/mol-instruction/train.jsonl', prompt='', return_prompt=True)
            self.test_dataset = MolInstructionDataset('/oss/wangyujia/pretrain-bench/mol-instruction/train.jsonl', prompt='', return_prompt=True)

        else:
            self.train_dataset = Empty( prompt=self.prompt, return_prompt=True)
            self.val_dataset = Empty( prompt=self.prompt, return_prompt=True)
            self.test_dataset = Empty(prompt=self.prompt, return_prompt=True)

            #raise NotImplementedError

        self.tokenizer = None
        self.prot_tokenizer = None
    
    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=Stage2Collater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        return [val_loader, test_loader]
        
    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(
                self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len
            ),
        )
        return test_loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data')
        parser.add_argument('--dataset', type=str, default='data')
        parser.add_argument('--text_max_len', type=int, default=4096)
        parser.add_argument('--q_max_len', type=int, default=29)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prot_max_len', type=int, default=4096)
        parser.add_argument('--prompt', type=str, default='')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser



class DeepLocBinaryDataset(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(DeepLocBinaryDataset, self).__init__()
        self.data_path = data_path
        self.user_prompt = prompt  # 用户传入的 prompt 前缀
        self.return_prompt = return_prompt
        
        # 加载并预处理数据
        self.data_list = self._load_and_preprocess(self.data_path)
        self.text2id = self._build_text_vocab()

    def _load_and_preprocess(self, data_path):
        data_list = []
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            try:
                name = str(row['name'])
                prot_seq = str(row['aa_seq'])
                
                raw = str(row['label']).strip()
                text_seq = f"<answer>{raw}</answer>\n"
                prompt = f"According to the protein information provided below and the protein name {name}, is the protein localized to the membrane?  Options:\n0.\"Yes\"\n1.\"No\" \n You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
                if self.user_prompt:
                    prompt = prompt + self.user_prompt
                    
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
            #print(prompt)
            return prot_seq, prompt, text_seq, index
        return prot_seq, text_seq, index

    # def get_protein_sequence(self, index):
    #     return self.data_list[index][0]

    # def get_text_description(self, index):
    #     return self.data_list[index][1]

    # def get_text_id(self, text_seq):
    #     return self.text2id.get(text_seq, -1)

class DeepLocMultiDataset(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(DeepLocMultiDataset, self).__init__()
        self.data_path = data_path
        self.user_prompt = prompt  # 用户传入的 prompt 前缀
        self.return_prompt = return_prompt
        
        # 加载并预处理数据
        self.data_list = self._load_and_preprocess(self.data_path)
        self.text2id = self._build_text_vocab()

    def _load_and_preprocess(self, data_path):
        data_list = []
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            try:
                name = str(row['name'])
                prot_seq = str(row['aa_seq'])
                #text_seq = str(row['label']).strip() + '\n'
                raw = str(row['label']).strip()
                text_seq = f"<answer>{raw}</answer>\n"

                prompt=f"According to the protein information provided below and the protein name {name}, predict the most likely subcellular localization from the following options:\nOptions: 0. \"Nucleus, U\" \n 1. \"Cytoplasm, S\"  \n 2. \"Extracellular, S\"  \n 3. \"Mitochondrion, U\"  \n 4. \"Cell membrane, M\"  \n 5. \"Endoplasmic reticulum, M\"  \n 6. \"Plastid, S\"  \n 7. \"Golgi apparatus, M\"  \n  8. \"Lysosome/Vacuole, M\"  \n9. \"Peroxisome, U\"\n\nPlease directly provide the text of the most likely localization option.You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER> "
                if self.user_prompt:
                    prompt = prompt + self.user_prompt
                
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
            #print(prompt)
            return prot_seq, prompt, text_seq, index
        return prot_seq, text_seq, index

    def get_protein_sequence(self, index):
        return self.data_list[index][0]

    def get_text_description(self, index):
        return self.data_list[index][1]

    def get_text_id(self, text_seq):
        return self.text2id.get(text_seq, -1)

class Deepsol(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(Deepsol, self).__init__()
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
                prot_seq = str(row['prot_seq']).strip()
                result = str(row['result']).strip()
                text_seq = f"<answer>{result}</answer>\n"
                feature = str(row['feather']).strip()
                prompt = f"According to the protein information provided before, as well as its quantitative features—including sequence length, molecular weight, absolute charge, aliphatic index (AI), average hydrophobicity (GRAVY), proportions of turn-forming residues, predicted secondary structure composition (3-state and 8-state SS), and the fraction of exposed residues (FER) at various relative solvent accessibility (RSA) thresholds:{feature},Is the protein soluble? options:\n0.\"No, the protein is not soluble\" \n 1.\"Yes, the protein is soluble\" \n Please directly provide the text of the most likely solubility option. You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
                #prompt = f"According to the protein information provided before, predict whether the protein is soluble from the following options:\n0.\"No, the protein is not soluble\" \n 1.\"Yes, the protein is soluble\" \n Please directly provide the text of the most likely solubility option. You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
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

class DeepsoluE(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(DeepsoluE, self).__init__()
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
                prot_seq = str(row['prot_seq']).strip()
                result = str(row['label']).strip()
                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"Given the protein information provided before, Is the protein soluble? Options:\n0.\"No, the protein is not soluble\" \n 1.\"Yes, the protein is soluble\" \n Please directly provide the text of the most likely solubility option. You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
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

class Protsolm(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(Protsolm, self).__init__()
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
                name = str(row['name'])
                prot_seq = str(row['aa_seq']).strip()
                result = str(row['label']).strip()

                text_seq = f"<answer>{result}</answer>\n"
 
                prompt = f"Given the protein information provided before and the protein name {name}, Is the protein soluble? Options:\n0.\"No, the protein is not soluble\" \n 1.\"Yes, the protein is soluble\" \n Please directly provide the text of the most likely solubility option. You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
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


class FLIP_GB1(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(FLIP_GB1, self).__init__()
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
 
                prompt = f"Given a protein sequence from the GB1 domain before, predict its binding affinity to the Fc region of a human IgG antibody. The GB1 domain is a small protein that binds to the Fc region, which includes the CH2 and CH3 domains of the IgG heavy chains. Mutations in the GB1 sequence can enhance or reduce its binding affinity. The predicted binding strength should be expressed as a fitness score ranging from 0 (no binding) to 1 (wild-type level binding).Question: What is the predicted binding fitness score  for this sequence? You can only give the score number in the answer, and write it like this: <answer>score number</answer>"
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


class FLIP_AAV(Dataset):
    def __init__(self, data_path, prompt='', return_prompt=False):
        super(FLIP_AAV, self).__init__()
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
 
                prompt = f"Adeno-associated virus (AAV) capsid proteins are essential for delivering DNA payloads into target cells, a property widely exploited in gene therapy. Mutations in a specific 28-amino acid region (positions 561–588 of VP-1, UniProt ID: P03135) can alter the capsid’s ability to package DNA and infect cells. The fitness of a given AAV variant reflects its ability to successfully assemble and deliver its genetic cargo. In this task, given a full AAV capsid protein sequence with mutations restricted to positions 561–588, predict the fitness score of the variant. The fitness score ranges from 0 (non-functional) to 1 (wild-type level functional performance).What is the predicted fitness score for this AAV variant, representing its ability to package and deliver DNA? <answer>fitness score</answer>"
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

class AlpacaDataset(Dataset):
    def __init__(self, jsonl_path, prompt='', return_prompt=False):
        super(AlpacaDataset, self).__init__()
        self.jsonl_path = jsonl_path
        self.user_prompt = prompt  # 可选的附加 prompt 内容
        self.return_prompt = return_prompt

        # 加载数据
        self.data_list = self._load_and_preprocess(self.jsonl_path)
        self.text2id = self._build_text_vocab()

    def _load_and_preprocess(self, jsonl_path):
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    prot_seq = ''  # 始终为空
                    text_seq = obj['result'].strip() + '\n'
                    prompt = obj['text'].strip()
                    if self.user_prompt:
                        prompt = prompt + self.user_prompt
                    data_list.append((prot_seq, text_seq, prompt))
                except Exception as e:
                    print(f"警告: 跳过有问题的行: {line}，原因: {e}")
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
            return prot_seq, prompt, text_seq, index
        return prot_seq, text_seq, index

    def get_protein_sequence(self, index):
        return self.data_list[index][0]

    def get_text_description(self, index):
        return self.data_list[index][1]

    def get_text_id(self, text_seq):
        return self.text2id.get(text_seq, -1)


class MolInstructionDataset(Dataset):
    def __init__(self, jsonl_path, prompt='',return_prompt=False):
        super(MolInstructionDataset, self).__init__()
        self.jsonl_path = jsonl_path
        self.return_prompt = return_prompt

        self.data_list = self._load_and_preprocess(self.jsonl_path)
        self.text2id = self._build_text_vocab()
        self.user_prompt = prompt  # 用户传入的 prompt 前缀
        
        # 加载并预处理数据

    def _extract_prot_from_input(self, input_str):
        # 提取两个 \n 之间的内容作为 prot
        parts = input_str.split('\n')
        if len(parts) >= 3:
            return parts[1].strip()  # 取第2部分（索引1）
        else:
            return ''  # 不满足格式返回空字符串

    def _load_and_preprocess(self, jsonl_path):
        data_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    #prot_seq = " "
                    prot_seq = self._extract_prot_from_input(obj.get('input', ''))
                    prompt = obj.get('instruction', '').strip()
                    text_seq = obj.get('output', '').strip() + '\n'
                    data_list.append((prot_seq, prompt, text_seq))
                except Exception as e:
                    print(f"警告: 跳过有问题的行: {line}，原因: {e}")
        return data_list

    def _build_text_vocab(self):
        text2id = {}
        for _, _, text_seq in self.data_list:
            if text_seq not in text2id:
                text2id[text_seq] = len(text2id)
        return text2id

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, prompt, text_seq = self.data_list[index]
        if self.return_prompt:
            return prot_seq, prompt, text_seq, index
        return prot_seq, text_seq, index

    def get_protein_sequence(self, index):
        return self.data_list[index][0]

    def get_prompt(self, index):
        return self.data_list[index][1]

    def get_text_description(self, index):
        return self.data_list[index][2]

    def get_text_id(self, text_seq):
        return self.text2id.get(text_seq, -1)


class Empty(Dataset):
    def __init__(self, prompt='', return_prompt=False):
        super(Empty, self).__init__()
        
        self.user_prompt = prompt  # 用户传入的 prompt 前缀
        self.return_prompt = return_prompt
        
        # 加载并预处理数据
        self.data_list = self._load_and_preprocess()
        self.text2id = self._build_text_vocab()

    def _load_and_preprocess(self):
        data_list = []
        prot_seq = "MKILSVLLLALIICSIVGWSEAQFTDVSCTTSKECWSVCQRLHNTSIGKCMNKKCRCYS"
        text_seq = "FUNCTION: Binds to actin and affects the structure of the cytoskeleton. At high concentrations, profilin prevents the polymerization of actin, whereas it enhances it at low concentrations (By similarity). SUBCELLULAR LOCATION: Cytoplasm, cytoskeleton. SIMILARITY: Belongs to the profilin family."
        #prompt = self.user_prompt
        #prompt = "According to the protein information provided below and the protein name [O82291]), is the protein localized to the membrane? Options:\n0.\"Yes\"\n1.\"No\" You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
        #prompt = "According to the protein information provided below and the protein name [Q9H400], predict the most likely subcellular localization from the options below.\n Options: 0. \"Nucleus, U\" \n 1. \"Cytoplasm, S\"  \n 2. \"Extracellular, S\"  \n 3. \"Mitochondrion, U\"  \n 4. \"Cell membrane, M\"  \n 5. \"Endoplasmic reticulum, M\"  \n 6. \"Plastid, S\"  \n 7. \"Golgi apparatus, M\"  \n  8. \"Lysosome/Vacuole, M\"  \n9. \"Peroxisome, U\"\n You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>\n "   
        prompt = 'Swiss-Prot description:{}, According to the protein information provided below and the protein name {name}, is the protein localized to the membrane?  Please solve the following problem step by step. Think carefully through each step before answering. First, analyze the problem, then identify relevant information, perform any necessary calculations or reasoning, and finally provide a clear and concise answer at the end.'
        #prompt = "what's the weather today"
        data_list.append((prot_seq, text_seq, prompt))
        # prot_seq = "AGFPEQEPEPKFWNDWAQKTLDKALSLQTLNKNKAQNLILFLGDGMGVPTVTAARILKGQLRGQPGEEGQLEMDKFPFVALSKTYNTNAQVADSAGTATAYLCGVKANEGTVGVSAAAVRSQANTTQGNEVTSILRWAKDAGKSIGIVTTTRVNHATPSAAYAHCVDRDWYSDNEMPADAVEAGCKDIARQLFENIPDIDVIMGGGRKYMYPKNTTDVEYPGQPKHSGTRKDGRNLVKEWVDRNTEKKGHYVWNKKDLLSLNPTKVDYLLGLFEPADLPYDLERNKETDPSLSEMVEVAIKILRRNPNGFYLLVEGGRIDHGHHEGKDKQAIHEAVEMDRAIGRADLMTSTSDTLTVVTADHSHLFSFGGYTPRGNEIFGLAAFISDVDQKPFTAILYGNGPGYKLVNGARENVSTVDYQDNSYLAQAAVPLSSETHGGEDVAVFAKGPMAHLLHGVHEQNYIPHAMAYAACIGQNR"
        # text_seq = ""
        # #prompt = self.user_prompt
        # # prompt = "According to the protein information provided below and the protein name [O82291]), is the protein localized to the membrane? Options:\n0.\"Yes\"\n1.\"No\" You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
        # prompt ="According to the protein information provided below and the protein name [Q9H400], predict the most likely subcellular localization from the following options:\nOptions: 0. \"Nucleus, U\" \n 1. \"Cytoplasm, S\"  \n 2. \"Extracellular, S\"  \n 3. \"Mitochondrion, U\"  \n 4. \"Cell membrane, M\"  \n 5. \"Endoplasmic reticulum, M\"  \n 6. \"Plastid, S\"  \n 7. \"Golgi apparatus, M\"  \n  8. \"Lysosome/Vacuole, M\"  \n9. \"Peroxisome, U\"\n\nPlease directly provide the text of the most likely localization option."
        # #prompt = "Using the protein information above and the protein name [O82291]), is the protein localized to the membrane?"
        # #prompt = "what's the weather today"
        # data_list.append((prot_seq, text_seq, prompt))
        # prot_seq = "MEDEAVLDRGASFLKHVCDEEEVEGHHTIYIGVHVPKSYRRRRRHKRKTGHREKKEKERISENYSDKSDVENADESSSSILKPLISPAAERIRFILGEEDDSPAPPQLFTELDELLAVDGQEMEWKETARWIKFEEKVEQGGERWSKPHVATLSLHSLFELRTCMEKGSIMLDREASSLPQLVEMIVDHQIETGLLKPDLKDKVTYTLLRKHRHQTKKSNLRSLADIGKTVSSASRMFTNPDNGSPAMTHRNLTSSSLNDISDKPEKDQLKNKFMKKLPRDAEASNVLVGEVDFLDSPFIAFVRLQQAVMLGALTEVPVPTRFLFILLGPKGKAKSYHEIGRAIATLMSDEVFHDIAYKAKDRQDLIAGIDEFLDEVIVLPPGEWDPAIRIEPPKSLPSSDKRKNMYSGGENVQMNGDTPPDGGHGGGGHADCEELQRTGRFCGGLIKDIKRKAPFFASDFYDALNIQALSAILFIYLATVTNAITFGGLLGDATDNMQGVLESFLGTAVSGAIFCLFAGQPLTILSSTGPVLVFERLLFNFSKDHNFDYLEFRLWIGLWSAFLCLILVATDASFLVQYFTRFTEEGFSSLISFIFIYDAFKKMIKLADYYPINSNFKVGYNTQFSCVCMPPDPVNISVSNDTTLAPEDLPTISSSNMYHNATFDWAFLTTKECLKYGGKLVGNNCGFVPDITLMSFILFLGTYTSSMALKKFKTSPYFPTTARKLISDFAIILPILIFCVIDALVGVDTPKLIVPSEFKPTSPNRGWFVAPFGGNPWWVYLAAAIPALLVTILIFMDQQITAVIVNRKEHKLKKGAGYHLDLFWVAILMVVCSFMALPWYVAATVISIAHIDSLKMETETSAPGEQPKFLGVREQRVTGTLVFILTGLSVFMAPILKFIPMPVLYGVFLYMGVASLNGVQFMDRLKLLLMPLKHQPDFIYLRHVPLRRVHLFTFLQVLCLALLWILKSTVAAIIFPVMILALVAVRKGMDYLFSQHDLSFLDDVIPEKDKKKKEDEKKKKKKKGSVDSDNDDSDCPYSEKVPSIKIPMDIMEQQPFLSDSKPSDRERSPTFLERHTSC"
        # text_seq = ""
        # #prompt = self.user_prompt
        # # prompt = "According to the protein information provided below and the protein name [O82291]), is the protein localized to the membrane? Options:\n0.\"Yes\"\n1.\"No\" You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
        # prompt = "According to the protein information provided below and the protein name [Q9H400], predict the most likely subcellular localization from the following options:\nOptions: 0. \"Nucleus, U\" \n 1. \"Cytoplasm, S\"  \n 2. \"Extracellular, S\"  \n 3. \"Mitochondrion, U\"  \n 4. \"Cell membrane, M\"  \n 5. \"Endoplasmic reticulum, M\"  \n 6. \"Plastid, S\"  \n 7. \"Golgi apparatus, M\"  \n  8. \"Lysosome/Vacuole, M\"  \n9. \"Peroxisome, U\"\n\nPlease directly provide the text of the most likely localization option."
        # #prompt = "Using the protein information above and the protein name [O82291]), is the protein localized to the membrane?"
        # #prompt = "what's the weather today"
        # data_list.append((prot_seq, text_seq, prompt))
        # prot_seq = "MHSRLKFLAYLHFICASSIFWPEFSSAQQQQQTVSLTEKIPLGAIFEQGTDDVQSAFKYAMLNHNLNVSSRRFELQAYVDVINTADAFKLSRLICNQFSRGVYSMLGAVSPDSFDTLHSYSNTFQMPFVTPWFPEKVLAPSSGLLDFAISMRPDYHQAIIDTIQYYGWQSIIYLYDSHDGLLRLQQIYQELKPGNETFRVQMVKRIANVTMAIEFLHTLEDLGRFSKKRIVLDCPAEMAKEIIVQHVRDIKLGRRTYHYLLSGLVMDNHWPSDVVEFGAINITGFRIVDSNRRAVRDFHDSRKRLEPSGQSQSQNAGGPNSLPAISAQAALMYDAVFVLVEAFNRILRKKPDQFRSNHLQRRSHGGSSSSSATGTNESSALLDCNTSKGWVTPWEQGEKISRVLRKVEIDGLSGEIRFDEDGRRINYTLHVVEMSVNSTLQQVAEWRDDAGLLPLHSHNYASSSRSASASTGDYDRNHTYIVSSLLEEPYLSLKQYTYGESLVGNDRFEGYCKDLADMLAAQLGIKYEIRLVQDGNYGAENQYAPGGWDGMVGELIRKEADIAISAMTITAERERVIDFSKPFMTLGISIMIKKPVKQTPGVFSFLNPLSQEIWISVILSYVGVSFVLYFVTRFPPYEWRIVRRPQADSTAQQPPGIIGGATLSEPQAHVPPVPPNEFTMLNSFWYSLAAFMQQGCDITPPSIAGRIAAAVWWFFTIILISSYTANLAAFLTVERMVAPIKTPEDLTMQTDVNYGTLLYGSTWEFFRRSQIGLHNKMWEYMNANQHHSVHTYDEGIRRVRQSKGKYALLVESPKNEYVNARPPCDTMKVGRNIDTKGFGVATPIGSPLRKRLNEAVLTLKENGELLRIRNKWWFDKTECNLDQETSTPNELSLSNVAGIYYILIGGLLLAVIVAIMEFFCRNKTPQLKSPGSNGSAGGVPGMLASSTYQRDSLSDAIMHSQAKLAMQASSEYDERLVGVELASNVRYQYSM"
        # text_seq = ""
        # #prompt = self.user_prompt
        # # prompt = "According to the protein information provided below and the protein name [O82291]), is the protein localized to the membrane? Options:\n0.\"Yes\"\n1.\"No\" You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"
        # prompt = "According to the protein information provided below and the protein name [Q9H400], predict the most likely subcellular localization from the following options:\nOptions: 0. \"Nucleus, U\" \n 1. \"Cytoplasm, S\"  \n 2. \"Extracellular, S\"  \n 3. \"Mitochondrion, U\"  \n 4. \"Cell membrane, M\"  \n 5. \"Endoplasmic reticulum, M\"  \n 6. \"Plastid, S\"  \n 7. \"Golgi apparatus, M\"  \n  8. \"Lysosome/Vacuole, M\"  \n9. \"Peroxisome, U\"\n\nPlease directly provide the text of the most likely localization option. "
        # #prompt = "Using the protein information above and the protein name [O82291]), is the protein localized to the membrane?"
        # #prompt = "what's the weather today"
        # data_list.append((prot_seq, text_seq, prompt))
        # prot_seq = "MAGATSSIIRENDFEDELAESMQSYNRETADKLALTRTESVKPEPEITAPPHSRFSRSFKTVLIAQCAFTGFFSTIAGAIYYPVLSVIERKFDIDEELVNVTVVVYFVFQGLAPTFMGGFADSLGRRPVVLVAIVIYFGACIGLACAQTYAQIIVLRCLQAAGISPVIAINSGIMGDVTTRAERGGYVGYVAGFQVLGSAFGALIGAGLSSRWGWRAIFWFLAIGSGICFLASFLILPETKRNISGNGSVTPKSYLNRAPILVLPTVRKSLHLDNPDYETLELPTQLNLLAPFKILKAYEICILMLVAGLQFAMYTTHLTALSTALSKQYHLTVAKVGLCYLPSGICTLCSIVIAGRYLNWNYRRRLKYYQNWLGKKRSKLLEEHDNDLNLVQRIIENDPKYTFNIFKARLQPAFVTLLLSSSGFCAYGWCITVKAPLAAVLCMSGFASLFSNCILTFSTTLIVDLFPTKTSTATGCLNLFRCILSAVFIAALSKMVEKMKFGGVFTFLGALTSSSSILLFILLRKGKELAFKRKKQELGVN"
        # text_seq = ""
        # #prompt = self.user_prompt
        # prompt = "According to the protein information provided below and the protein name [O82291]), is the protein localized to the membrane? Options:\n0.\"Yes\"\n1.\"No\" You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"   
        # #prompt = "Using the protein information above and the protein name [O82291]), is the protein localized to the membrane?"
        # #prompt = "what's the weather today"
        # data_list.append((prot_seq, text_seq, prompt))
        # prot_seq = "MSVADFYGSNVEVLLNNDSKARGVITNFDSSNSILQLRLANDSTKSIVTKDIKDLRILPKNEIMPKNGTKSPSTNSTKLKSAETYSSKNKWSMDCDEEFDFAANLEKFDKKQVFAEFREKDKKDPAKLLVSHNKSPNRNYHHKQNVLGPSVKDEFVDLPSAGSQINGIDAVLSSSSNGHVTPGSKKGSRETLKKKPFVDENIPAELHTTTGDILKPITPEQLSQGIALAIAKTSTDIVVENAAQLLSQFVFSVLGGHKRLSSRNHNSQPLVCILVGSHDHASAAVAAGRRLCAIGIKVVLRLLTPFNVDNRQLLMFQAAGGYIPTENFDQFLNKLTSPIELVVDVLTGFHPSIDKNSHALIQWANDLNVLILSVDIPSGYTVQKKNTAILPKWTLALGAVTTTLAQAALVKQAAGVSVFVGNLGTGSQTWAELGILESQVTGQYLAQISCTSTN"
        # text_seq = ""
        # #prompt = self.user_prompt
        # prompt = "According to the protein information provided below and the protein name [O82291]), is the protein localized to the membrane? Options:\n0.\"Yes\"\n1.\"No\" You need to answer the following question directly, which means you can only give the number of the option in the answer. For example: <ANSWER>option number</ANSWER>"   
        # #prompt = "Using the protein information above and the protein name [O82291]), is the protein localized to the membrane?"
        # #prompt = "what's the weather today"
        # data_list.append((prot_seq, text_seq, prompt))
            
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
            print(prompt)
            return prot_seq, prompt, text_seq, index
        return prot_seq, text_seq, index

    def get_protein_sequence(self, index):
        return self.data_list[index][0]

    def get_text_description(self, index):
        return self.data_list[index][1]

    def get_text_id(self, text_seq):
        return self.text2id.get(text_seq, -1)