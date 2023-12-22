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

def preprocess_data(tokenizer, preprocessed_data_path, is_test, data, ddp_rank=-1):
    if ddp_rank != 0 and ddp_rank != -1:
        dist.barrier()

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

    random.Random(0).shuffle(all_texts)
    random.Random(0).shuffle(all_labels)

    global LOOP_ALL_COUNT
    global LOOP_CUR_COUNT
    LOOP_ALL_COUNT = len(all_texts) * 4 # tk_loop twice, labels and texts
    LOOP_CUR_COUNT = 0

    print(f'len all_texts: {len(all_texts):,}')
    print(f'len all_labels: {len(all_labels):,}')
    if len(all_texts) != len(all_labels):
        raise Exception('?')

    isExist = os.path.exists(preprocessed_data_path)
    max_ids_len = 1024
    if not isExist:
        print('tokenizing...', preprocessed_data_path)
        # all_texts_ids = tk_loop(tokenizer, all_texts, pad_id=0, is_test=is_test)
        # all_labels_ids = tk_loop(tokenizer, all_labels, pad_id=-100, is_test=is_test)
        fd = os.open(preprocessed_data_path, os.O_RDWR | os.O_CREAT)
        line_index = 0
        pos_offset = 0
        jsonl_positions_for_seek = []
        n = len(all_texts)
        for index in range(n):
            text = all_texts[index]
            label = all_labels[index]
            if line_index % 1000 == 0:
                progress = line_index / n
                print(f'{progress:.3f}', end="\r")

            ids_1 = tokenizer(text, max_length=1024)
            len_pad = max_ids_len - len(ids_1)
            if len_pad > 0:
                ids_1 = ids_1 + [0 for x in range(len_pad)]

            ids_2 = tokenizer(label, max_length=1024)
            len_pad = max_ids_len - len(ids_2)
            if len_pad > 0:
                ids_2 = ids_2 + [-100 for x in range(len_pad)]    

            line_index += 1
            jsonl_positions_for_seek.append(pos_offset)
            line = json.dumps({'input': ids_1, 'label': ids_2}) + '\n'
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
                if line_index % 1000 == 0 and ddp_rank == 0:
                    print(f'{line_index}', end="\r")
                line_index += 1
            else:
                print('\nlast line?', line)    

        print('all lines', line_index) 
        fh.close()

        if ddp_rank == 0:
            dist.barrier()

    return jsonl_positions_for_seek
