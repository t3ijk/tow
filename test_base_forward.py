
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
model_weights_path = "./test_models/byt5-small/pytorch_model.bin"
model_config_path = "./test_models/byt5-small/config.json"

config = None
with open(model_config_path, 'r') as f:
    config = json.load(f)

model = None

def test_model_forward(input_ids, labels, training):
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

    if training:
        model = model.train()  
    else:
        model = model.eval()
    # input_ids = torch.tensor([list(str_in.encode("utf-8"))]) + 3  # add 3 for special tokens
    # labels = torch.tensor([list(str_out.encode("utf-8"))]) + 3  # add 3 for special tokens

    t0 = time.time()
    n = 1
    for i in range(n):
        print('model.training', model.training)
        torch.manual_seed(0)
        _, loss = model(input_ids, labels=labels)
    t1 = time.time()
    print(loss, 'deltaT', (t1 - t0) / n)


def test_model_1(training):
    inputs = ["hello world"]
    outputs = ['你好世界']
    batches = [[inputs, outputs]]
    for it in batches:
        input_ids = []
        for x in it[0]:
            input_ids.append([y + 3 for y in x.encode("utf-8")])
        label_ids = []
        for x in it[1]:
            label_ids.append([y + 3 for y in x.encode("utf-8")])  
        input_ids = torch.tensor(input_ids)
        label_ids = torch.tensor(label_ids)
        test_model_forward(input_ids, label_ids, training) 

# should be: tensor(5.0514, grad_fn=<NllLossBackward0>) deltaT 0.43586087226867676
test_model_1(False)
