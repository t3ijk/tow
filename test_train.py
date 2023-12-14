from src.model_byt5.tokenizer import Tokenizer_byt5
from src.model_byt5.model import Transformer_byt5
from convert import hf_model_weight_name_mapping
import torch
import json
from collections import OrderedDict
import time
from src.model_byt5.trainer import train_loop, prepare_env
import shutil
import os
import pandas as pd
import sys
import random
from src.preprocess_data import preprocess_data

print(sys.argv)
path = sys.argv[1]
is_test = sys.argv[2]
is_test = is_test == 'test'
checkpoints_path ='./checkpoints'
base_model_path = path
model_weights_path = f"{base_model_path}/pytorch_model.bin"
model_config_path = f"{base_model_path}/config.json"

config = None
with open(model_config_path, 'r') as f:
    config = json.load(f)

model = None

def get_ddp_rank():
    return int(os.environ.get('RANK', -1))

def get_model():
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
    return model

def get_data(preprocessed_data_path):
    ddp_rank = get_ddp_rank()
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
    return jsonl_positions_for_seek

def test_train():
    model = get_model()
    prepare_env(model, checkpoints_path, config)
    preprocessed_data_path = f"./preprocessed_data_tow_byt5.jsonl"
    jsonl = get_data(preprocessed_data_path)
    n_val = 30
    train_loop(model,
                preprocessed_data_path=preprocessed_data_path,
                training_data=jsonl[n_val: ],  # train data
                validation_data=jsonl[0: n_val], # validation data
                checkpoints_path=checkpoints_path,
                n_epoch_=1,
                batch_size_=1,
                resume_path=None,
                device='cpu',
                n_iters_for_estimate_loss_=2 if is_test else 1000,
                gradient_accumulation_steps_=2,
                warmup_iters_=3000)     
    
test_train()
