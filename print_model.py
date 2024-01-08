import torch
import json
import sys
print(sys.argv)
path = sys.argv[1]
base_model_path = path
model_weights_path = f"{base_model_path}/pytorch_model.bin"
state_dict = torch.load(model_weights_path)
for name, tensor_ in state_dict.items():
    print(name, tensor_.shape)
