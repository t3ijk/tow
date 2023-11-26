
import inspect
import torch
from src.model_byt5.model import Transformer_byt5
import os
import json
from dataclasses import dataclass, asdict
import shutil
from src.utils import delete_files_in_directory
import math
import time
import datetime
from src.model_byt5.tokenizer import Tokenizer_byt5
from dataclasses import dataclass, asdict

# ref: karpathy/nanoGPT
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

def prepare_datas():
    pass

def estimate_loss(model):
    out = {}
    model.eval()
    inputs = [[y + 3 for y in 'This is training a LLM model! '.encode("utf-8")]]
    outputs = [[258, *[y + 3 for y in '这是训练LLM模型！'.encode("utf-8")], 1]]
    # batches = [[inputs, outputs]]
    input_ids = torch.tensor(inputs)
    label_ids = torch.tensor(outputs)
    _, loss = model(input_ids, label_ids)
    model.train()
    return loss


def get_batch(size, datas, sample_offset):
    inputs = []
    labels = []

    max_in_ids = 0
    max_la_ids = 0

    for data in datas[sample_offset:sample_offset+size]:
        in_ids = [*[y + 3 for y in data[0].encode("utf-8")], 1, 258]
        la_ids = [258, *[y + 3 for y in data[1].encode("utf-8")], 1]
        if len(in_ids) > max_in_ids:
            max_in_ids = len(in_ids)
        if len(la_ids) > max_la_ids:
            max_la_ids = len(la_ids)            
        inputs.append(in_ids)
        labels.append(la_ids)



    inputs_with_pads = []
    for l in inputs:
        n_pads = max_in_ids - len(l)
        l = [*l, *[0 for _ in range(n_pads)]]
        inputs_with_pads.append(l)

    labels_with_pads = []
    for l in labels:
        n_pads = max_la_ids - len(l)
        l = [*l, *[-100 for _ in range(n_pads)]]
        labels_with_pads.append(l) 


    # print(inputs_with_pads)
    # print(labels_with_pads)  
    # tk  = Tokenizer_byt5()
    # for i in range(len(labels_with_pads)):
    #     print(tk.ids2text(inputs_with_pads[i]), tk.ids2text(labels_with_pads[i]))


    return inputs_with_pads, labels_with_pads   

def save_checkpoints(index_of_epoch, steps, cur_estimate_loss, checkpoints_path, model, train_config, is_minimal_loss):
        train_info = {
            'model_args': '',
            'iter_num': f"{index_of_epoch}-{steps}",
            'best_val_loss': cur_estimate_loss.tolist(),
            'data': f"{datetime.datetime.utcnow()}"
        }
        name = 'minimal_loss' if  is_minimal_loss else 'last_loss'
        fold = f"{checkpoints_path}/{name}"
        delete_files_in_directory(fold) 
        torch.save(model.state_dict(), f"{fold}/pytorch_model.bin")
        with open(f"{fold}/config.json", "w") as f:
            json.dump(asdict(model.byt5config), f, indent=4)
        with open(f"{fold}/train_config.json", "w") as f:
            json.dump(asdict(train_config), f, indent=4)     
        with open(f"{fold}/train_info.json", "w") as f:
            json.dump(train_info, f, indent=4)   

def log_write(fd, log):
    log = log + '\n'
    os.write(fd, bytes(log, 'utf-8'))
    os.fsync(fd)


def safe_check(model, checkpoints_path, train_config):
    print(train_config)
    print(model.byt5config)
    dir = os.listdir(checkpoints_path)
    if len(dir) != 0:
        raise Exception(f"The Directory Is Not Empty. {checkpoints_path}")
    
@dataclass
class Train_config:
    # adamw optimizer
    learning_rate: float = 6e-4 # max learning rate
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    device_type: str  = 'cpu'
    # optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
    n_samples: int = 0
    batch_size: int = 0
    n_epoch: int = 0
    steps_for_estimate_loss: int = 50
    gradient_accumulation_steps: int = 2

def train_loop(model: Transformer_byt5, datas, checkpoints_path, n_epoch_, batch_size_):
    train_config = Train_config()
    train_config.n_samples = len(datas)
    train_config.batch_size = batch_size_
    train_config.n_epoch = n_epoch_

    safe_check(model, checkpoints_path, train_config)

    cur_estimate_loss = torch.tensor(-1)
    min_estimate_loss = torch.tensor(999)
    optimizer = configure_optimizers(model,
                                     train_config.weight_decay,
                                     train_config.learning_rate,
                                     (train_config.beta1, train_config.beta2),
                                     train_config.device_type)

    last_t = time.time()
    all_past_steps = 0
    fd = os.open('./out.log', os.O_RDWR)
    for index_of_epoch in range(train_config.n_epoch):
        sample_offset = 0
        steps = 0
        epoch_steps = math.floor(train_config.n_samples / train_config.batch_size)
        
        while train_config.n_samples - sample_offset > train_config.batch_size * train_config.gradient_accumulation_steps:
                
                need_estimate_loss = False
                for _ in range(train_config.gradient_accumulation_steps):
                    inputs, labels = get_batch(train_config.batch_size, datas, sample_offset)
                    input_ids = torch.tensor(inputs)
                    label_ids = torch.tensor(labels)

                    # model forward
                    _, loss = model(input_ids, label_ids)

                    # use real steps to flag need_estimate_loss
                    if steps % train_config.steps_for_estimate_loss == 0:
                        need_estimate_loss = True

                    # update steps
                    sample_offset += train_config.batch_size
                    steps += 1
                    all_past_steps += 1    

                    now = time.time()

                    delta_t = now - last_t
                    last_t = now
                    
                    all_steps = train_config.n_epoch * epoch_steps
                    remain_steps = all_steps - all_past_steps
                    progress = "{:.4f}".format(all_past_steps/all_steps) 
                    log = f"{index_of_epoch}/{train_config.n_epoch}-{steps}/{epoch_steps}-{progress}, 'loss:', {loss.tolist()}, 'ts', {now}, 'h', {delta_t * remain_steps / 3600}"
                    print(log)
                    log_write(fd, log+'\n')
                    loss = loss / train_config.gradient_accumulation_steps
                    loss.backward()

                if train_config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if need_estimate_loss:
                    need_estimate_loss = False
                    cur_estimate_loss = estimate_loss(model)
                    log = f"'cur_estimate_loss', {cur_estimate_loss.tolist()}, 'ts', {time.time()}"
                    print(log)
                    log_write(fd, log+'\n')
                    is_minimal_loss = False
                    if cur_estimate_loss < min_estimate_loss:
                        min_estimate_loss = cur_estimate_loss
                        is_minimal_loss = True
                    save_checkpoints(index_of_epoch, steps, cur_estimate_loss, checkpoints_path, model, train_config, is_minimal_loss)
       
    os.close(fd)                        