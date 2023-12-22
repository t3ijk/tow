
from src.model_byt5.tokenizer import Tokenizer_byt5
from src.model_byt5.model import Transformer_byt5
import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from collections import OrderedDict
from src.utils import print_model_info, delete_files_in_directory
from convert import hf_model_weight_name_mapping

model_weights_path = "./test_models/byt5-small/pytorch_model.bin"
model_config_path = "./test_models/byt5-small/config.json"

config = None
with open(model_config_path, 'r') as f:
    config = json.load(f)

state_dict = torch.load(model_weights_path)
state_dict_new = OrderedDict()
for name, tensor_ in state_dict.items():
    new_name = hf_model_weight_name_mapping(name)
    if new_name:
        state_dict_new[hf_model_weight_name_mapping(name)] = tensor_
model = None
if model is None:
    model = Transformer_byt5(config=config)
    model.load_state_dict(state_dict_new)

training = False
if training:
    model = model.train()  
else:
    model = model.eval()

tokenizer = Tokenizer_byt5()  
ids = tokenizer('aaaaaaaaaaaaaaaaaaa', max_length=1024)
len_pad = 1024 - len(ids)
if len_pad > 0:
    ids = ids + [0 for x in range(len_pad)]
print(ids)  
inputs = torch.tensor([ids]).to(torch.device('cpu'))
# outputs = model.generate(inputs, max_length=1024)
# text = tokenizer.ids2text(outputs.tolist()[0])
# print(text)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
    model.generate(inputs, max_length=1024)
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=100))
