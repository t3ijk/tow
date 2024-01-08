
from tow.model_byt5.tokenizer import Tokenizer_byt5
from tow.model_byt5.model import Transformer_byt5
import torch
import json
model = None

import sys
print(sys.argv)
path = sys.argv[1]

model_weights_path_ = f"{path}/pytorch_model.bin"
model_config_path_ = f"{path}/config.json"

def estimate_loss():

    state_dict = torch.load(model_weights_path_, map_location=torch.device('cpu'))
    config_ = None
    with open(model_config_path_, 'r') as f:
        config_ = json.load(f)


    global model    

    if model is None:
        model = Transformer_byt5(config=config_)
        # print(model)
        model.load_state_dict(state_dict)
    model = model.eval()
    training_data = None
    validation_data = None

    with open('./test/val.json', 'r') as f:
        data = json.load(f)
        validation_data = data[0: 20]
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

        print(input_ids)
        gen_outputs = model.generate(input_ids, max_length=512, use_cache=True)
        print(f"{index}==============x, y, y^: ")
        print([tk.ids2text(input_ids.tolist()[0])])
        print([tk.ids2text(label_ids.tolist()[0])])
        print([tk.ids2text(gen_outputs.tolist()[0])])
   
estimate_loss()   