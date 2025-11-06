# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
from data_provider.stage3_dm import DeepLocBinaryDataset,AlpacaDataset,MolInstructionDataset,DeepLocMultiDataset,Deepsol,DeepsoluE,Protsolm,FLIP_GB1,FLIP_AAV
from data_provider.bindingdb import BindingDB
from data_provider.metalIonbinding import MetallonBinding
from data_provider.go import GO_BP,EC
from data_provider.production import Antibiotic_Resistance,Thermostability,Material,Clone
from data_provider.mutation import TAPE_Stability,TAPE_Fluorescence

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
        # print("=========")
        # print(prot_seqs)

        self.tokenizer.padding_side = 'right'
        prompt_tokens = self.tokenizer(prompt_seqs,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=False,
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


class Stage2DM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
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
        
          # self.train_dataset = AlpacaDataset('/nas/shared/kilab/wangyujia/pretrain_data/instruct/alpaca-gpt4-train.jsonl', prompt=self.prompt, return_prompt=True)
        # self.val_dataset = AlpacaDataset('/nas/shared/kilab/wangyujia/pretrain_data/instruct/alpaca-gpt4-valid.jsonl', prompt=self.prompt, return_prompt=True)
        # self.test_dataset = AlpacaDataset('/nas/shared/kilab/wangyujia/pretrain_data/instruct/alpaca-gpt4-test.jsonl', prompt=self.prompt, return_prompt=True)

        # self.train_dataset = MolInstructionDataset('/oss/wangyujia/pretrain-bench/mol-instruction/train.jsonl', prompt='', return_prompt=True)
        # self.val_dataset = MolInstructionDataset('/oss/wangyujia/pretrain-bench/mol-instruction/train.jsonl', prompt='', return_prompt=True)
        # self.test_dataset = MolInstructionDataset('/oss/wangyujia/pretrain-bench/mol-instruction/train.jsonl', prompt='', return_prompt=True)
     
        if self.args.dataset=='deeplocbinary':
            self.train_dataset = DeepLocBinaryDataset('/oss/wangyujia/pretrain-bench/locate/deeplocbinary/train.csv', prompt=self.prompt, return_prompt=True)
            self.val_dataset = DeepLocBinaryDataset('/oss/wangyujia/pretrain-bench/locate/deeplocbinary/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = DeepLocBinaryDataset('/oss/wangyujia/pretrain-bench/locate/deeplocbinary/test.csv', prompt=self.prompt, return_prompt=True)
        
        elif self.args.dataset=='deeplocmulti':
            self.train_dataset = DeepLocMultiDataset('/oss/wangyujia/pretrain-bench/locate/deeplocmulti/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = DeepLocMultiDataset('/oss/wangyujia/pretrain-bench/locate/deeplocmulti/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = DeepLocMultiDataset('/oss/wangyujia/pretrain-bench/locate/deeplocmulti/test.csv', prompt=self.prompt, return_prompt=True)
        
        elif self.args.dataset=='deepsol':
            self.train_dataset = Deepsol('/nas/shared/kilab/wangyujia/sft_data/deepsol/clean/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = Deepsol('/nas/shared/kilab/wangyujia/sft_data/deepsol/clean/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = Deepsol('/nas/shared/kilab/wangyujia/sft_data/deepsol/clean/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='deepsolue':
            self.train_dataset = DeepsoluE('/nas/shared/kilab/wangyujia/sft_data/deepsoluE/clean/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = DeepsoluE('/nas/shared/kilab/wangyujia/sft_data/deepsoluE/clean/test.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = DeepsoluE('/nas/shared/kilab/wangyujia/sft_data/deepsoluE/clean/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='protsolm':
            self.train_dataset = Protsolm('/oss/wangyujia/pretrain-bench/solubility/protsolm/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = Protsolm('/oss/wangyujia/pretrain-bench/solubility/protsolm/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = Protsolm('/oss/wangyujia/pretrain-bench/solubility/protsolm/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='gb1':
            self.train_dataset = FLIP_GB1('/nas/shared/kilab/wangyujia/sft_data/mutation/gb1/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = FLIP_GB1('/nas/shared/kilab/wangyujia/sft_data/mutation/gb1/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = FLIP_GB1('/nas/shared/kilab/wangyujia/sft_data/mutation/gb1/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='gb1_low':
            self.train_dataset = FLIP_GB1('/nas/shared/kilab/wangyujia/sft_data/mutation/gb1/clean/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = FLIP_GB1('/nas/shared/kilab/wangyujia/sft_data/mutation/gb1/clean/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = FLIP_GB1('/nas/shared/kilab/wangyujia/sft_data/mutation/gb1/clean/test.csv', prompt=self.prompt, return_prompt=True)
       
        elif self.args.dataset=='aav':
            self.train_dataset = FLIP_AAV('/nas/shared/kilab/wangyujia/sft_data/mutation/aav/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = FLIP_AAV('/nas/shared/kilab/wangyujia/sft_data/mutation/aav/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = FLIP_AAV('/nas/shared/kilab/wangyujia/sft_data/mutation/aav/test.csv', prompt=self.prompt, return_prompt=True)
        
        elif self.args.dataset=='bindingdb':
            self.train_dataset = BindingDB('/nas/shared/kilab/wangyujia/sft_data/bindingdb/clean/train_small.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = BindingDB('/nas/shared/kilab/wangyujia/sft_data/bindingdb/clean/valid_small.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = BindingDB('/nas/shared/kilab/wangyujia/sft_data/bindingdb/clean/test_small.csv', prompt=self.prompt, return_prompt=True)
        
        elif self.args.dataset=='metallonbinding':
            self.train_dataset = MetallonBinding('/nas/shared/kilab/wangyujia/sft_data/MetalIonBinding/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = MetallonBinding('/nas/shared/kilab/wangyujia/sft_data/MetalIonBinding/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = MetallonBinding('/nas/shared/kilab/wangyujia/sft_data/MetalIonBinding/test.csv', prompt=self.prompt, return_prompt=True)
        
        elif self.args.dataset=='bp':
            self.train_dataset = GO_BP('/nas/shared/kilab/wangyujia/sft_data/go/clean/BP_train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = GO_BP('/nas/shared/kilab/wangyujia/sft_data/go/clean/BP_valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = GO_BP('/nas/shared/kilab/wangyujia/sft_data/go/clean/BP_test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='ec':
            self.train_dataset = EC('/nas/shared/kilab/wangyujia/sft_data/EC/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = EC('/nas/shared/kilab/wangyujia/sft_data/EC/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = EC('/nas/shared/kilab/wangyujia/sft_data/EC/test.csv', prompt=self.prompt, return_prompt=True)

        elif self.args.dataset=='antibiotic':
            self.train_dataset = Antibiotic_Resistance('/nas/shared/kilab/wangyujia/sft_data/production/Antibiotic_Resistance/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = Antibiotic_Resistance('/nas/shared/kilab/wangyujia/sft_data/production/Antibiotic_Resistance/test.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = Antibiotic_Resistance('/nas/shared/kilab/wangyujia/sft_data/production/Antibiotic_Resistance/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='thermostability':
            self.train_dataset = Thermostability('/nas/shared/kilab/wangyujia/sft_data/production/Thermostability/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = Thermostability('/nas/shared/kilab/wangyujia/sft_data/production/Thermostability/valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = Thermostability('/nas/shared/kilab/wangyujia/sft_data/production/Thermostability/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='material':
            self.train_dataset = Material('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/material_production/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = Material('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/material_production/val.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = Material('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/material_production/test.csv', prompt=self.prompt, return_prompt=True)
        #6
        elif self.args.dataset=='clone':
            self.train_dataset = Clone('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/cloning_clf/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = Clone('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/cloning_clf/val.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = Clone('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/cloning_clf/test.csv', prompt=self.prompt, return_prompt=True)
        


        elif self.args.dataset=='stability':
            self.train_dataset = TAPE_Stability('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/TAPE_Stability/train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = TAPE_Stability('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/TAPE_Stability/val.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = TAPE_Stability('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/TAPE_Stability/test.csv', prompt=self.prompt, return_prompt=True)
        elif self.args.dataset=='fluorescence':
            self.train_dataset = TAPE_Fluorescence('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/TAPE_Fluorescence/fluorescence_prediction_train.csv',prompt=self.prompt, return_prompt=True)
            self.val_dataset = TAPE_Fluorescence('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/TAPE_Fluorescence/fluorescence_prediction_valid.csv', prompt=self.prompt, return_prompt=True)
            self.test_dataset = TAPE_Fluorescence('/oss/wangyujia/ProtT3/ProtT3/data/sft/dataset/TAPE_Fluorescence/test.csv', prompt=self.prompt, return_prompt=True)
        
        
        elif self.args.dataset=='empty':
            self.train_dataset = Empty( prompt=self.prompt, return_prompt=True)
            self.val_dataset = Empty( prompt=self.prompt, return_prompt=True)
            self.test_dataset = Empty(prompt=self.prompt, return_prompt=True)
        
        elif self.args.dataset=='swiss-prot':
            self.train_dataset = SwissProtDataset(root+'/SwissProtV3/train_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
            self.val_dataset = SwissProtDataset(root+'/SwissProtV3/valid_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
            self.test_dataset = SwissProtDataset(root+'/SwissProtV3/test_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)

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
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
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
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
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
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data')
        parser.add_argument('--text_max_len', type=int, default=2048)
        parser.add_argument('--q_max_len', type=int, default=29)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prompt', type=str, default='The protein has the following properties:')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser



class Stage2MixDM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        # self.prompt = args.prompt
        assert args.mix_dataset
        
        train_dataset1 = SwissProtDataset(root+'/SwissProtV3/train_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
        train_dataset2 = OntoProteinDataset(root+'/OntoProteinDatasetV2/train.txt', prompt='Gene Ontology description: ', return_prompt=True)
        self.train_dataset = ConcatDataset([train_dataset1,train_dataset2])
        self.swiss_val_dataset = SwissProtDataset(root+'/SwissProtV3/valid_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
        self.onto_val_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt', prompt='Gene Ontology description: ', return_prompt=True)
        self.swiss_test_dataset = SwissProtDataset(root+'/SwissProtV3/test_set.jsonl', prompt='Swiss-Prot description: ', return_prompt=True)
        self.onto_test_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt', prompt='Gene Ontology description: ', return_prompt=True)
        
        

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
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        return loader

    def val_dataloader(self):
        swiss_val_loader = DataLoader(
            self.swiss_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        swiss_test_loader = DataLoader(
            self.swiss_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )

        onto_val_loader = DataLoader(
            self.onto_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        onto_test_loader = DataLoader(
            self.onto_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len),
        )
        return [swiss_val_loader, swiss_test_loader, onto_val_loader, onto_test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data')
        parser.add_argument('--text_max_len', type=int, default=1024)
        parser.add_argument('--q_max_len', type=int, default=29)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        # parser.add_argument('--prompt', type=str, default='The protein has the following properties: ')
        return parent_parser


