
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
import re
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

# ref: karpathy/nanoGPT, learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it == 0:
        it = 1

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

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
            'epoch-steps': f"{index_of_epoch}-{steps}",
            'best_val_loss': cur_estimate_loss.tolist(),
            'data': f"{datetime.datetime.utcnow().isoformat()}"
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

def log_format(train_config, index_of_epoch, steps, epoch_steps, cur_step_num, lr, all_steps, loss, now, delta_t, remain_steps):
    progress = "{:.4f}".format(cur_step_num/all_steps)
    lr_2 = "{:.5e}".format(lr)
    h = "{:.2f}".format(delta_t * remain_steps / 3600)
    return f"{index_of_epoch}/{train_config.n_epoch}-{steps}/{epoch_steps}-{progress}, 'loss:', {loss.tolist()}, 'ts', {now}, 'lr', {lr_2}, 'h', {h}"

def log_write(fd, log):
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
    # ref: karpathy/nanoGPT
    # adamw optimizer
    learning_rate: float = 6e-4 # max learning rate
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 2000 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    device_type: str  = 'cpu'
    n_sample: int = 0
    batch_size: int = 0
    n_epoch: int = 0
    steps_for_estimate_loss: int = 25
    gradient_accumulation_steps: int = 2

def train_loop(model: Transformer_byt5, datas, checkpoints_path, n_epoch_, batch_size_):
    train_config = Train_config()
    train_config.n_sample = len(datas)
    train_config.batch_size = batch_size_
    train_config.n_epoch = n_epoch_

    train_config.max_iters = math.floor((train_config.n_sample / batch_size_ /  train_config.gradient_accumulation_steps) * n_epoch_)
    train_config.warmup_iters = math.floor(train_config.max_iters / 300)
    train_config.lr_decay_iters = train_config.max_iters

    safe_check(model, checkpoints_path, train_config)

    cur_estimate_loss = torch.tensor(-1)
    min_estimate_loss = torch.tensor(999)
    optimizer = configure_optimizers(model,
                                     train_config.weight_decay,
                                     train_config.learning_rate,
                                     (train_config.beta1, train_config.beta2),
                                     train_config.device_type)

    last_t = time.time()

    now_iso = datetime.datetime.utcnow().isoformat()
    out_log_path = f'out-{re.sub(r"[^0-9]", ".", now_iso)}.log'
    # os.mkdir(out_log_path)
    fd = os.open(out_log_path, os.O_RDWR | os.O_CREAT)
 
    # all steps, model.forward()
    cur_step_num = 0
    # gradient descent steps, optimizer.step(), ~= cur_step_num / gradient_accumulation_steps
    cur_iter_num = 0
    # loop for n_epoch
    for index_of_epoch in range(train_config.n_epoch):
        sample_offset = 0
        steps = 0
        epoch_steps = math.floor(train_config.n_sample / train_config.batch_size)
        
        # loop for n_sample
        while train_config.n_sample - sample_offset > train_config.batch_size * train_config.gradient_accumulation_steps:
                # determine and set the learning rate for this iteration
                lr = get_lr(cur_iter_num,
                            train_config.warmup_iters,
                            train_config.learning_rate,
                            train_config.lr_decay_iters,
                            train_config.min_lr) if train_config.decay_lr else train_config.learning_rate
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                            
                need_estimate_loss = False

                # loop for gradient_accumulation_steps
                for _ in range(train_config.gradient_accumulation_steps):
                    inputs, labels = get_batch(train_config.batch_size, datas, sample_offset)
                    input_ids = torch.tensor(inputs)
                    label_ids = torch.tensor(labels)

                    # model forward
                    _, loss = model(input_ids, label_ids)

                    # update steps
                    sample_offset += train_config.batch_size
                    steps += 1
                    cur_step_num += 1    

                    now = time.time()
                    delta_t = now - last_t
                    last_t = now
                    
                    all_steps = train_config.n_epoch * epoch_steps
                    remain_steps = all_steps - cur_step_num
                    log = log_format(train_config, index_of_epoch, steps, epoch_steps, cur_step_num, lr, all_steps, loss, now, delta_t, remain_steps)
                    print(log)
                    log_write(fd, log+'\n')
                    loss = loss / train_config.gradient_accumulation_steps
                    loss.backward()

                if train_config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                
                # use cur_iter_num to flag need_estimate_loss
                if cur_iter_num % train_config.steps_for_estimate_loss == 0:
                    need_estimate_loss = True
                cur_iter_num += 1
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