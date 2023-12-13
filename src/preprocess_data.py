from transformers import MT5ForConditionalGeneration, T5ForConditionalGeneration, MT5Tokenizer, ByT5Tokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback, TrainerState, TrainerControl
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import os
import json
import sys
import torch.distributed as dist

LOOP_ALL_COUNT = 0
LOOP_CUR_COUNT = 0
LOOP_TOKEN_COUNT = 0

def tk_loop(tokenizer, texts, pad_id=0, is_test=False):
    global LOOP_ALL_COUNT
    global LOOP_CUR_COUNT
    global LOOP_TOKEN_COUNT
    texts_tkd = []
    len_max = 0

    if is_test:
        len_max = 1024 # for test oom

    for text in texts:
        LOOP_CUR_COUNT += 1
        progress = LOOP_CUR_COUNT / LOOP_ALL_COUNT
        print(f'{progress:.6f}', end='\r', flush=True)
        ids = tokenizer(text, max_length=1024)
        texts_tkd.append(ids)
        cur_len = len(ids)
        LOOP_TOKEN_COUNT += cur_len
        if cur_len > len_max:
            len_max = cur_len

    texts_tkd_padded = []
    for ids in texts_tkd:
        LOOP_CUR_COUNT += 1
        progress = LOOP_CUR_COUNT / LOOP_ALL_COUNT
        print(f'{progress:.6f}', end='\r', flush=True)
        len_ids = len(ids)
        len_pad = len_max - len_ids
        if len_pad > 0:
            ids = ids + [pad_id for x in range(len_pad)]
        texts_tkd_padded.append(ids)
    
    print('\nmax ids len: ', len_max)    
    return texts_tkd_padded   

def preprocess_data(tokenizer, preprocessed_data_path, is_test, data_files, ddp_rank=-1):
    if ddp_rank != 0 and ddp_rank != -1:
        dist.barrier()

    data = []
    for file in data_files:
        path = file['path']
        src = file['src']
        to = file['to']

        data_df = pd.read_csv(path, sep='\t',  on_bad_lines='skip', nrows=1000000)
        data_df['src'] = src
        data_df['to'] = to
        data = [*data, *data_df.values.tolist()]
        print(path, data_df["src"].value_counts()) 
  
    random.Random(0).shuffle(data)
    print('len(data)', len(data))

    if is_test:
        data = data[0: 200]

    all_texts = []
    all_labels = []

    for it in data:
        text0 = it[0]
        text1 = it[1]
        src = it[2]
        to = it[3]
        all_texts.append(f"{src}2{to}:{text0}")
        all_labels.append(f"{text1}")
        all_texts.append(f"{to}2{src}:{text1}")
        all_labels.append(f"{text0}")

    global LOOP_ALL_COUNT
    global LOOP_CUR_COUNT
    LOOP_ALL_COUNT = len(all_texts) * 4 # tk_loop twice, labels and texts
    LOOP_CUR_COUNT = 0

    print('len(all_texts)', len(all_texts))
    print('len(all_labels)', len(all_labels))

    isExist = os.path.exists(preprocessed_data_path)
    if not isExist:
        print('tokenizing...', preprocessed_data_path)
        all_texts_ids = tk_loop(tokenizer, all_texts, pad_id=0, is_test=is_test)
        all_labels_ids = tk_loop(tokenizer, all_labels, pad_id=-100, is_test=is_test)
        fd = os.open(preprocessed_data_path, os.O_RDWR | os.O_CREAT)
        line_index = 0
        pos_offset = 0
        jsonl_positions_for_seek = []
        n = len(all_texts_ids)
        for index in range(n):
            line_index += 1
            print(f'{line_index}/{n}', end="\r")
            jsonl_positions_for_seek.append(pos_offset)
            line = json.dumps({'input': all_texts_ids[index], 'label': all_labels_ids[index]}) + '\n'
            # print('len: ', len(line), len(bytes(line, 'utf-8')))

            linesep_offset = 1 if os.linesep == '\r\n' else 0
            pos_offset += len(line) + linesep_offset
            os.write(fd, bytes(line, 'utf-8'))
        os.fsync(fd)
        os.close(fd)
        print('TOKEN_COUNT: ', LOOP_TOKEN_COUNT)
        if ddp_rank == 0:
            dist.barrier()
    else:
        print('will load from cache...', preprocessed_data_path)
        fh = open(preprocessed_data_path, 'r')
        line = True
        line_index = 0
        pos_offset = 0
        jsonl_positions_for_seek = []
        all_texts_ids = []
        all_labels_ids = []
        while line:
            line = fh.readline()
            if line:
                jsonl_positions_for_seek.append(pos_offset)
                linesep_offset = 1 if os.linesep == '\r\n' else 0
                pos_offset += len(line) + linesep_offset
                line_index += 1
            else:
                print('\nlast line?', line)    

        print('all lines', line_index) 
        fh.close()

        if ddp_rank == 0:
            dist.barrier()

    return jsonl_positions_for_seek
