from src.model_byt5.tokenizer import Tokenizer_byt5
from src.model_byt5.model import Transformer_byt5
from convert import hf_model_weight_name_mapping
import torch
import json
from collections import OrderedDict
import time
from src.model_byt5.trainer import train_loop, train_check
import shutil
import os
import pandas as pd
import sys
import random
from src.preprocess_data import preprocess_data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from math import floor
import datetime

print(sys.argv)
path = sys.argv[1]
tag = sys.argv[2]
is_test = tag == 'test'

checkpoints_path = f'./checkpoints_{tag}'

MAX_IDS_LEN = 512
preprocessed_data_path = f'./preprocessed_data_tow_byt5_{tag}.jsonl_{MAX_IDS_LEN}'

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
    if ddp_rank == 0 or ddp_rank == -1:
        files = [ 
            # file, from, to
            {'path': './data/WikiMatrix.en-es.tsv', 'src': 'en', 'to': 'es', 'c_b': 1, 'c_e': 3, 'nrows': 1000000},
            {'path': './data/WikiMatrix.en-ja.tsv', 'src': 'en', 'to': 'ja', 'c_b': 1, 'c_e': 3, 'nrows': 1000000},
            {'path': './data/WikiMatrix.en-zh.tsv', 'src': 'en', 'to': 'zh', 'c_b': 1, 'c_e': 3, 'nrows': 500000},
            {'path': './data/WikiMatrix.ja-zh.tsv', 'src': 'ja', 'to': 'zh', 'c_b': 1, 'c_e': 3, 'nrows': 1000000},
            {'path': './data/WikiMatrix.es-zh.tsv', 'src': 'es', 'to': 'zh', 'c_b': 1, 'c_e': 3, 'nrows': 1000000},
            {'path': './data/WikiMatrix.es-ja.tsv', 'src': 'es', 'to': 'ja', 'c_b': 1, 'c_e': 3, 'nrows': 1000000},
            {'path': './data/news-commentary-v14.en-zh.cleaned.tsv', 'src': 'en', 'to': 'zh', 'c_b': 0, 'c_e': 2, 'nrows': 500000},
            ]
        data = []
        for file in files:
            path = file['path']
            src = file['src']
            to = file['to']
            c_b = file['c_b']
            c_e = file['c_e']
            to = file['to']
            nrows = file['nrows']
            data_df = pd.read_csv(path, sep='\t',  on_bad_lines='skip', nrows=nrows).iloc[:, c_b:c_e]
            data_df['src'] = src
            data_df['to'] = to
            data = [*data, *data_df.values.tolist()]
            print(path, data_df["src"].value_counts()) 

        print(f'len data: {len(data):,}')

        # enhance
        all_texts = []
        all_labels = []

        for it in data:
            text0 = it[0]
            text1 = it[1]
            src = it[2]
            to = it[3]
            all_texts.append(f"{src}2{to}:{text0}")
            all_labels.append(f"{text1}")
            all_texts.append(f"{to}2{src}:{text1}")
            all_labels.append(f"{text0}")

        random.Random(999).shuffle(all_texts)
        random.Random(999).shuffle(all_labels)

        if is_test:
            n_test = 200
            all_texts = all_texts[0: n_test]
            all_labels = all_labels[0: n_test]
            for i in range(n_test):
                print(f'[{i}]{all_texts[i]}\n{all_labels[i]}')

        print(f'len all_texts: {len(all_texts):,}')
        print(f'len all_labels: {len(all_labels):,}')
        if len(all_texts) != len(all_labels):
            raise Exception('?')    

    jsonl_positions_for_seek = preprocess_data(Tokenizer_byt5(), preprocessed_data_path, all_texts, all_labels, ddp_rank=ddp_rank, max_ids_len=MAX_IDS_LEN)
    return jsonl_positions_for_seek

def get_env():
    device = 'cpu'
    if torch.cuda.device_count() > 0 :
        device = 'cuda:0'
    env_info = dict(
        ddp_rank=0,
        ddp_local_rank=0,
        ddp_world_size=1,
        device=device,
        is_master_process = True,
        is_ddp = False,
        )
    if get_ddp_rank() != -1:
        env_info['is_ddp'] = True
        init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=86400))
        env_info['ddp_rank'] = int(os.environ['RANK'])
        env_info['ddp_local_rank'] = int(os.environ['LOCAL_RANK'])
        env_info['ddp_world_size'] = int(os.environ['WORLD_SIZE'])
        env_info['device'] = f'cuda:{env_info["ddp_local_rank"]}'
        torch.cuda.set_device(env_info['device'])
        env_info['is_master_process'] = env_info['ddp_rank'] == 0
    return env_info    

def test_train():
    model = get_model()
    train_check(model, checkpoints_path, config)
    env_info = get_env()
    jsonl = get_data(preprocessed_data_path)
    n_val = 60

    jsonl_val = jsonl[0:n_val]
    jsonl_tra_all = jsonl[n_val:]

    ddp_rank = env_info['ddp_rank']
    size = floor(len(jsonl_tra_all) / env_info['ddp_world_size'])
    jsonl_tra = jsonl_tra_all[ddp_rank*size: (ddp_rank+1)*size]
    print('ddp_rank, size, len(jsonl_tra): ', ddp_rank, size, len(jsonl_tra))
    
    # test resume
    torch.manual_seed(1)
    print('env_info', env_info)
    train_loop(model,
                preprocessed_data_path=preprocessed_data_path,
                training_data=jsonl_tra,  # train data
                validation_data=jsonl_val, # validation data
                checkpoints_path=checkpoints_path,
                n_epoch_=3,
                batch_size_=5,
                resume_path=None,
                device=env_info['device'],
                n_iters_for_estimate_loss_=2 if is_test else 1500,
                gradient_accumulation_steps_=2 if is_test else 20,
                warmup_iters_=3000,
                env_info=env_info)
    
    if env_info['is_ddp']:
        destroy_process_group()    
    
test_train()
