
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
import random
import gc

# ref: karpathy/nanoGPT
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):

    # freeze_encoder(model)

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

def safe_encode_utf8(txt):
    return f"{txt}".encode("utf-8")

@torch.no_grad()
def validate_loss(jsonl_f, model, validation_data, device):
    print('validate_loss...')
    out = {}
    model.eval()

    tk  = Tokenizer_byt5()

    n = len(validation_data)
    loss_n = torch.zeros([n]).to(torch.device(device))

    for index, pos in enumerate(validation_data):
        jsonl_f.seek(pos)
        line = jsonl_f.readline()
        lo = json.loads(line)
        in_ids = lo['input']
        la_ids = lo['label'] 
        input_ids = torch.tensor([in_ids]).to(torch.device(device))
        label_ids = torch.tensor([la_ids]).to(torch.device(device))
        output_logits, loss = model(input_ids, label_ids)
        values, indices = output_logits.topk(1)

        # teacher forcing outputs
        outputs = indices.reshape(indices.shape[0:-1]) # (batch, n, 1) -> (batch, n)

        # free generated outputs
        gen_outputs = model.generate(input_ids, max_length=1024, use_cache=True)
        print(index, '-----------------')
        print([tk.text_clean_special_tokens(tk.ids2text(input_ids.tolist()[0]))])
        print([tk.text_clean_special_tokens(tk.ids2text(label_ids.tolist()[0]))])
        print(["{:.2}".format(loss.tolist()), tk.ids2text(outputs.tolist()[0])])
        print([tk.ids2text(gen_outputs.tolist()[0])])
        loss_n[index] = loss
    model.train()
    return torch.mean(loss_n)

def get_batch(jsonl_f, size, training_data, it_cur_sample_offset, data_indexes_shuffled):
    inputs = []
    labels = []

    in_ids_len = 0
    la_ids_len = 0

    for index in data_indexes_shuffled[it_cur_sample_offset:it_cur_sample_offset+size]:
        # get data from index
        jsonl_f.seek(training_data[index])
        line = jsonl_f.readline()
        lo = json.loads(line)
        in_ids = lo['input']
        la_ids = lo['label']    
        in_ids_len = len(in_ids)
        la_ids_len = len(in_ids)        
        inputs.append(in_ids)
        labels.append(la_ids)
    n_token = (in_ids_len + la_ids_len) * size
    return inputs, labels, n_token   

def save_checkpoints(it_info,
                    checkpoints_path,
                    model,
                    train_config,
                    is_minimal_loss,
                    optimizer):
    
        name = 'minimal_loss' if  is_minimal_loss else 'last_loss'
        fold = f"{checkpoints_path}/{name}"
        delete_files_in_directory(fold) 
        torch.save(model.state_dict(), f"{fold}/pytorch_model.bin")
        torch.save(optimizer.state_dict(), f"{fold}/optimizer.bin")
        with open(f"{fold}/config.json", "w") as f:
            json.dump(asdict(model.byt5config), f, indent=4)
        with open(f"{fold}/train_config.json", "w") as f:
            json.dump(asdict(train_config), f, indent=4)     
        with open(f"{fold}/it_info.json", "w") as f:
            json.dump(it_info, f, indent=4)  

def log_format(train_config,
            it_index_of_epoch,
            it_micro_step_index_cur_epoch,
            it_micro_step_num_per_epoch,
            it_cur_micro_step_index,
            it_cur_iter_index,
            lr,
            all_micro_step_num,
            it_cur_sample_offset,
            all_sample_num,
            loss,
            now,
            delta_t,
            remain_steps,
            it_tokens_consumed,
            it_cur_estimate_loss):
    
    progress = "{:.4f}".format(it_cur_micro_step_index/all_micro_step_num)
    lr_2 = "{:.5e}".format(lr)
    h = "{:.2f}".format(delta_t * remain_steps / 3600)

    loss = "{:.4}".format(loss)
    v_loss = "{:.4}".format(it_cur_estimate_loss)

    info = {
        'ep': f'{it_index_of_epoch}/{train_config.n_epoch}',
        'st': f'{it_micro_step_index_cur_epoch}/{it_micro_step_num_per_epoch}',
        'sa': f'{it_cur_sample_offset}/{all_sample_num}',
        'iter': it_cur_iter_index,
        'pro': progress,
        'ls': loss,
        'vls': v_loss,
        'ts': now,
        'lr': lr_2,
        'tks': it_tokens_consumed,
        'h': h
    }
    return json.dumps(info)

def safe_check(model, checkpoints_path, train_config):
    print(train_config)
    print(model.byt5config)
    parameters_count = sum(p.numel() for p in model.parameters())
    print(f'parameters count: {parameters_count:,}')
    isExist = os.path.exists(checkpoints_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(checkpoints_path)
        print("The new directory is created!", checkpoints_path)

    dir = os.listdir(checkpoints_path)
    if len(dir) != 0:
        raise Exception(f"The directory is not empty. You may need to back up checkpoints and then clear the directory. {checkpoints_path}")
    
    
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
    n_iters_for_estimate_loss: int = 25
    gradient_accumulation_steps: int = 2


# def freeze_encoder(model):
#     print('freeze_encoder')
#     for n, p in model.named_parameters():
#         if n.startswith("encoder") or n.startswith("shared"):
#             p.requires_grad = False

def train_loop(model_,
               preprocessed_data_path,
               training_data,
               validation_data,
               checkpoints_path,
               n_epoch_,
               batch_size_,
               resume_path=None,
               device='cpu',
               n_iters_for_estimate_loss_=None,
               gradient_accumulation_steps_=None,
               warmup_iters_=None):
    
    jsonl_f = open(preprocessed_data_path, "r")
    model: Transformer_byt5

    is_resume_training = resume_path is not None

    if is_resume_training:
        print('is_resume_training', is_resume_training)
        with open(f'{resume_path}/config.json') as f:
            config = json.load(f)
        model = Transformer_byt5(config=config)
        model.load_state_dict(torch.load(f'{resume_path}/pytorch_model.bin'))
        model = model.train()
        with open(f'{resume_path}/train_config.json') as f:
            cf = json.load(f)
            train_config = Train_config(**cf)
        
        safe_check(model, checkpoints_path, train_config)
        optimizer = configure_optimizers(model,
                                        train_config.weight_decay,
                                        train_config.learning_rate,
                                        (train_config.beta1, train_config.beta2),
                                        train_config.device_type)
        optimizer.load_state_dict(torch.load(f'{resume_path}/optimizer.bin'))

        last_t = time.time()
        with open(f'{resume_path}/it_info.json') as f:
            it_info = json.load(f)
            print('Resumed it_info:', it_info)
            it_cur_estimate_loss = torch.tensor(it_info['it_cur_estimate_loss']).to(torch.device(device))
            it_min_estimate_loss = torch.tensor(it_info['it_min_estimate_loss']).to(torch.device(device))
            it_cur_micro_step_index = it_info['it_cur_micro_step_index']
            it_cur_iter_index = it_info['it_cur_iter_index']
            it_tokens_consumed = it_info['it_tokens_consumed'] 
                      
            it_index_of_epoch_resume = it_info['it_index_of_epoch']
            it_cur_sample_offset_resume = it_info['it_cur_sample_offset']
            it_micro_step_index_cur_epoch_resume = it_info['it_micro_step_index_cur_epoch']
    
    else:
        print('is_resume_training', is_resume_training)
        model = model_
        train_config = Train_config()
        train_config.n_sample = len(training_data)
        train_config.batch_size = batch_size_
        train_config.n_epoch = n_epoch_
        train_config.max_iters = math.floor((train_config.n_sample / batch_size_ /  train_config.gradient_accumulation_steps) * n_epoch_)
        train_config.lr_decay_iters = train_config.max_iters
        if n_iters_for_estimate_loss_ is not None:
            train_config.n_iters_for_estimate_loss = n_iters_for_estimate_loss_
        if gradient_accumulation_steps_ is not None:
            train_config.gradient_accumulation_steps = gradient_accumulation_steps_
        if warmup_iters_ is not None:
            train_config.warmup_iters = warmup_iters_    

        safe_check(model, checkpoints_path, train_config)
        optimizer = configure_optimizers(model,
                                        train_config.weight_decay,
                                        train_config.learning_rate,
                                        (train_config.beta1, train_config.beta2),
                                        train_config.device_type)

        last_t = time.time()
        now_iso = datetime.datetime.utcnow().isoformat()
        it_cur_estimate_loss = torch.tensor(-1.0).to(torch.device(device))
        it_min_estimate_loss = torch.tensor(999.0).to(torch.device(device))
        # all steps, count model.forward()
        it_cur_micro_step_index = 0
        # gradient descent steps, count optimizer.step(), ~= it_cur_micro_step_index / gradient_accumulation_steps
        it_cur_iter_index = 0
        it_tokens_consumed = 0

        it_index_of_epoch_resume = 0

    # set device
    model.to(torch.device(device))

    print('train loop will start with train_config: ', train_config)
    for it_index_of_epoch in range(it_index_of_epoch_resume, train_config.n_epoch):

        # only shuffle per epoch
        data_indexes_shuffled = [*range(train_config.n_sample)]
        random.Random(it_index_of_epoch).shuffle(data_indexes_shuffled)
        print('epoch shuffle with seed: ', it_index_of_epoch)
    
        # only the resumed epoch is special
        is_resumed_epoch = is_resume_training and it_index_of_epoch == it_index_of_epoch_resume
        it_cur_sample_offset = it_cur_sample_offset_resume if is_resumed_epoch else 0
        it_micro_step_index_cur_epoch = it_micro_step_index_cur_epoch_resume if is_resumed_epoch else 0
        it_micro_step_num_per_epoch = math.floor(train_config.n_sample / train_config.batch_size)
        
        # loop for n_sample
        while train_config.n_sample - it_cur_sample_offset >= train_config.batch_size * train_config.gradient_accumulation_steps:
                # determine and set the learning rate for this iteration
                lr = get_lr(it_cur_iter_index,
                            train_config.warmup_iters,
                            train_config.learning_rate,
                            train_config.lr_decay_iters,
                            train_config.min_lr) if train_config.decay_lr else train_config.learning_rate
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                            
                need_estimate_loss = False

                # loop for gradient_accumulation_steps
                for _ in range(train_config.gradient_accumulation_steps):
                    inputs, labels, n_token = get_batch(jsonl_f, train_config.batch_size, training_data, it_cur_sample_offset, data_indexes_shuffled)
                    input_ids = torch.tensor(inputs).to(torch.device(device))
                    label_ids = torch.tensor(labels).to(torch.device(device))
                    it_tokens_consumed += n_token

                    # model forward
                    _, loss = model(input_ids, label_ids)

                    now = time.time()
                    delta_t = now - last_t
                    last_t = now
                    
                    all_micro_step_num = train_config.n_epoch * it_micro_step_num_per_epoch
                    remain_steps = all_micro_step_num - it_cur_micro_step_index
                    all_sample_num = train_config.n_sample
                    log = log_format(train_config,
                                     it_index_of_epoch,
                                     it_micro_step_index_cur_epoch,
                                     it_micro_step_num_per_epoch,
                                     it_cur_micro_step_index,
                                     it_cur_iter_index,
                                     lr,
                                     all_micro_step_num,
                                     it_cur_sample_offset,
                                     all_sample_num,
                                     loss,
                                     now,
                                     delta_t,
                                     remain_steps,
                                     it_tokens_consumed,
                                     it_cur_estimate_loss)
                    print(log)
                    loss = loss / train_config.gradient_accumulation_steps
                    loss.backward()
                    # update steps counter
                    it_cur_sample_offset += train_config.batch_size
                    it_micro_step_index_cur_epoch += 1
                    it_cur_micro_step_index += 1 

                if train_config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                it_cur_iter_index += 1

                # Samples consumed in one iter equals: gradient_accumulation_steps * batch_size.
                
                # use it_cur_iter_index to flag need_estimate_loss
                if it_cur_iter_index % train_config.n_iters_for_estimate_loss == 0:
                    need_estimate_loss = True
                
                if need_estimate_loss:
                    print(it_cur_iter_index, train_config.n_iters_for_estimate_loss)
                    need_estimate_loss = False
                    it_cur_estimate_loss = validate_loss(jsonl_f, model, validation_data, device)
                    is_minimal_loss = False
                    if it_cur_estimate_loss < it_min_estimate_loss:
                        it_min_estimate_loss = it_cur_estimate_loss
                        is_minimal_loss = True

                    it_info = {
                        'it_cur_estimate_loss': it_cur_estimate_loss.tolist(),
                        'it_min_estimate_loss': it_min_estimate_loss.tolist(),
                        'it_cur_micro_step_index': it_cur_micro_step_index,
                        'it_cur_iter_index': it_cur_iter_index,
                        'it_cur_sample_offset': it_cur_sample_offset,
                        'it_micro_step_index_cur_epoch': it_micro_step_index_cur_epoch,
                        'it_micro_step_num_per_epoch': it_micro_step_num_per_epoch,
                        'it_index_of_epoch': it_index_of_epoch,
                        'it_date': f"{datetime.datetime.utcnow().isoformat()}",
                        'it_tokens_consumed': it_tokens_consumed,
                        'is_resume_training': is_resume_training,
                    }    
                    save_checkpoints(it_info,
                                     checkpoints_path,
                                     model,
                                     train_config,
                                     is_minimal_loss,
                                     optimizer)
                # ?
                # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/19
                # gc.collect()
                # torch.cuda.empty_cache()     
    os.close(fd)

    print('all_tokens_consumed: ', it_tokens_consumed)
    print('Training is completed')                    
