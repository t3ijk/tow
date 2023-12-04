
from src.model_byt5.tokenizer import Tokenizer_byt5
from src.model_byt5.model import Transformer_byt5
from convert import hf_model_weight_name_mapping
import torch
import json
from collections import OrderedDict
import time
from src.utils import print_model_info, delete_files_in_directory
from src.model_byt5.train import train_loop
import shutil
import os
import pandas as pd
import sys
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_model", default=None, help="base-model path")
parser.add_argument("-d", "--datasets", default=None, help="datasets path")
parser.add_argument("-i", "--info", default=None, help="information of current experiment or experiment purpose")
parser.add_argument("--n_epoch", default=5, help="n_epoch")
parser.add_argument("--batch_size", default=5, help="batch_size")
parser.add_argument("--validation_data_size", default=10, help="validation_data_size")
parser.add_argument("--steps_for_estimate_loss", default=300, help="steps_for_estimate_loss")
parser.add_argument("--gradient_accumulation_steps", default=2, help="validation_data_size")
parser.add_argument("--warmup_iters", default=3000, help="warmup_iters")

args = parser.parse_args()
print(args)

if args.base_model is None or args.datasets is None:
    print('error: the following arguments are required: --base_model, --datasets')
    exit(0)

if args.info is None:
    print('error: the following arguments are required: --info, eg: for testing large batch size.')
    exit(0)    

base_model_path = args.base_model
model_weights_path = f"{base_model_path}/pytorch_model.bin"
model_config_path = f"{base_model_path}/config.json"
data_path = args.datasets
n_epoch = args.n_epoch
batch_size = args.batch_size
validation_data_size = args.validation_data_size 
steps_for_estimate_loss_ = args.steps_for_estimate_loss
gradient_accumulation_steps_ = args.gradient_accumulation_steps
warmup_iters_ = args.warmup_iters

config = None
with open(model_config_path, 'r') as f:
    config = json.load(f)

model = None
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

    training_data = None
    validation_data = None

    data = []
    print('data_path: ', data_path)
    if data_path.endswith('tsv'):
        data_df = pd.read_csv(data_path, sep='\t')
        data = data_df.values.tolist()
    elif data_path.endswith('csv'):
            data_df = pd.read_csv(data_path)
            data = data_df.values.tolist()
    else:
        with open('./datas/datas-v8.json', 'r') as f:
            data = json.load(f)
    

    random.Random(0).shuffle(data)
    validation_data = data[0: validation_data_size]
    training_data = data[validation_data_size:]
    checkpoints_path = './checkpoints'
    # delete_files_in_directory(checkpoints_path)

    device = 'cpu'
    if torch.cuda.device_count() > 0 :
        device = 'cuda:5'


    train_loop(model,
                training_data,
                validation_data,
                checkpoints_path,
                n_epoch,
                batch_size,
                resume_path=None,
                device=device,
                steps_for_estimate_loss_=steps_for_estimate_loss_,
                gradient_accumulation_steps_=gradient_accumulation_steps_,
                warmup_iters_=warmup_iters_)     
        

    # train_loop(model, training_data, validation_data, checkpoints_path, n_epoch, batch_size, device=device, steps_for_estimate_loss_=steps_for_estimate_loss_, gradient_accumulation_steps_=gradient_accumulation_steps_)
    # train_loop(model, training_data, validation_data, checkpoints_path, n_epoch, batch_size, resume_path='./checkpoints-resume/last_loss')

test_train()
