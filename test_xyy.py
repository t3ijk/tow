
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
model = None
def estimate_loss():
    # path = 'minimal_loss'
    path = 'last_loss'

    model_weights_path_ = f"./checkpoints/{path}/pytorch_model.bin"
    model_config_path_ = f"./checkpoints/{path}/config.json"

    state_dict = torch.load(model_weights_path_)
    config_ = None
    with open(model_config_path_, 'r') as f:
        config_ = json.load(f)


    global model    

    if model is None:
        model = Transformer_byt5(config=config_)
        model.load_state_dict(state_dict)
    model = model.eval()
    training_data = None
    validation_data = None

    with open('./datas/datas-v6.json', 'r') as f:
        data = json.load(f)
        validation_data = data[10: 20]
        training_data = data[10:]

    device = 'cpu'
    if torch.cuda.device_count() > 0 :
        device = 'cuda:5'
    model.eval()
    model.to(torch.device(device))
    tk  = Tokenizer_byt5()
    texts = []
    for index, data in enumerate(validation_data):
        input_ids = [[*[y + 3 for y in data[0].encode("utf-8")], 1, 258]]
        label_ids = [[258, *[y + 3 for y in data[1].encode("utf-8")], 1, 257]]
        input_ids = torch.tensor(input_ids).to(torch.device(device))
        label_ids = torch.tensor(label_ids).to(torch.device(device))
        output_logits, loss = model(input_ids, label_ids)
        values, indices = output_logits.topk(1)
        outputs = indices.reshape(indices.shape[0:-1]) # (batch, n, 1) -> (batch, n)

        print('==============x, y, y^: ')
        print(tk.ids2text(input_ids.tolist()[0]))
        print('-------')
        print(tk.ids2text(label_ids.tolist()[0]))
        print('-------')
        print(tk.ids2text(outputs.tolist()[0]))
   
estimate_loss()   