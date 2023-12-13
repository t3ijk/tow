from src.model_byt5.tokenizer import Tokenizer_byt5
from src.model_byt5.model import Transformer_byt5
from convert import hf_model_weight_name_mapping
import torch
import json
from collections import OrderedDict
import time
from src.model_byt5.trainer import train_loop
import shutil
import os
import pandas as pd
import sys
import random
import argparse
from src.preprocess_data import preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_model", default=None, help="base-model path")
parser.add_argument("-i", "--info", default=None, help="test or train or experiment")

args = parser.parse_args()
print(args)

if args.base_model is None:
    print('error: the following arguments are required: --base_model')
    exit(0)

if args.info is None:
    print('error: the following arguments are required: --info, eg: for testing large batch size.')
    exit(0)    

base_model_path = args.base_model
model_weights_path = f"{base_model_path}/pytorch_model.bin"
model_config_path = f"{base_model_path}/config.json"

config = None
with open(model_config_path, 'r') as f:
    config = json.load(f)

model = None

def get_ddp_rank():
    return int(os.environ.get('RANK', -1))

def test_train():
    state_dict = torch.load(model_weights_path)
    state_dict_new = OrderedDict()
    for name, tensor_ in state_dict.items():
        new_name = hf_model_weight_name_mapping(name)
        if new_name:
            state_dict_new[hf_model_weight_name_mapping(name)] = tensor_
    global model        

    if model is None:
        model = Transformer_byt5(config=config)
        model.load_state_dict(state_dict_new)
    model = model.train()
    # print_model_info(model)

    preprocessed_data_path = f"./preprocessed_data_tow_byt5.jsonl"
    ddp_rank = get_ddp_rank()
    is_test = args.info == 'test' 

    tsv1 = './data/wikititles-v3.zh-en.tsv'
    tsv2 = './data/news-commentary-v14.en-zh.cleaned.tsv'
    tsv3 = './data/TED2020.en_zh.tsv'
    files = [ 
        # file, from, to
        {'path': tsv1, 'src': 'zh', 'to': 'en'},
        {'path': tsv2, 'src': 'en', 'to': 'zh'},
        {'path': tsv3, 'src': 'en', 'to': 'zh'},
        ]
    
    jsonl_positions_for_seek = preprocess_data(Tokenizer_byt5(), preprocessed_data_path, is_test, data_files=files,ddp_rank=ddp_rank)

    n_val = 30
    train_loop(model,
                preprocessed_data_path=preprocessed_data_path,
                training_data=jsonl_positions_for_seek[n_val: ],  # train data
                validation_data=jsonl_positions_for_seek[0: n_val], # validation data
                checkpoints_path='./checkpoints',
                n_epoch_=1,
                batch_size_=1,
                resume_path=None,
                device='cpu',
                steps_for_estimate_loss_=2,
                gradient_accumulation_steps_=2,
                warmup_iters_=3000)     
    
test_train()
