
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
# https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L263
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
    outputs = [[258, *[y + 3 for y in '这是训练LLM模型！'.encode("utf-8")]]]
    # batches = [[inputs, outputs]]
    input_ids = torch.tensor(inputs)
    label_ids = torch.tensor(outputs)
    _, loss = model(input_ids, label_ids)
    model.train()
    return loss


def get_batch(size, datas, offset):
    inputs = []
    labels = []

    max_in_ids = 0
    max_la_ids = 0

    for data in datas[offset:offset+size]:
        in_ids = [*[y + 3 for y in data[0].encode("utf-8")], 258]
        la_ids = [258, *[y + 3 for y in data[1].encode("utf-8")]]
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

    return inputs_with_pads, labels_with_pads   

# def log(index_of_epoch=0, n_epoch=1, steps=0, epoch_steps=1, remain_steps=0, all_steps=1, loss=0, ts):

        # print(f'{index_of_epoch}/{n_epoch}-{steps}/{epoch_steps}-{remain_steps}/{all_steps}', 'loss:', loss.tolist(), 'ts', delta_t, 'h', delta_t * remain_steps / 3600)



def train_loop(model: Transformer_byt5, datas, checkpoints_path):

    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    device_type  = 'cpu'
    optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)

    index_of_iter = 0
    index_of_batch = 0
    index_of_epoch = 0

    n_samples = len(datas)
    batch_size = 4
    n_epoch = 20
    steps_for_estimate_loss = 50
    last_estimate_loss = torch.tensor(-1)
    gradient_accumulation_steps = 2


    last_t = time.time()
    all_past_steps = 0
    for index_of_epoch in range(n_epoch):
        offset = 0
        steps = 0
        epoch_steps = math.floor(n_samples / batch_size)
        
        while n_samples - offset > batch_size * gradient_accumulation_steps:
                
                need_estimate_loss = False
                for _ in range(gradient_accumulation_steps):
                    inputs, labels = get_batch(batch_size, datas, offset)
                    input_ids = torch.tensor(inputs)
                    label_ids = torch.tensor(labels)

                    # print(input_ids, label_ids)
                    _, loss = model(input_ids, label_ids)
                    offset += batch_size
                    steps += 1
                    all_past_steps += 1
                    if steps % steps_for_estimate_loss == 0:
                        need_estimate_loss = True

                    now = time.time()

                    delta_t = now - last_t
                    last_t = now
                    
                    all_steps = n_epoch * epoch_steps
                    remain_steps = all_steps - all_past_steps
                    print(f'{index_of_epoch}/{n_epoch}-{steps}/{epoch_steps}-{remain_steps}/{all_steps}', 'loss:', loss.tolist(), 'ts', now, 'h', delta_t * remain_steps / 3600)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                if grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if need_estimate_loss:
                    need_estimate_loss = False
                    last_estimate_loss = estimate_loss(model)
                    print('estimate_loss: ', last_estimate_loss)

                    train_info = {
                        'model_args': '',
                        'iter_num': f"{index_of_epoch}-{steps}",
                        'best_val_loss': last_estimate_loss.tolist(),
                    }

                    delete_files_in_directory(checkpoints_path)
                    fold = f"{checkpoints_path}/{index_of_epoch}-{steps}/"
                    os.mkdir(fold) 
                    torch.save(model.state_dict(), f"{fold}/pytorch_model.bin")
                    with open(f"{fold}/config.json", "w") as f:
                        json.dump(asdict(model.byt5config), f, indent=4) 
                    with open(f"{fold}/train_info.json", "w") as f:
                        json.dump(train_info, f, indent=4)   