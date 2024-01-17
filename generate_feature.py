###
# File: /generate_feature.py
# Created Date: Wednesday January 17th 2024
# Author: Zihan
# -----
# Last Modified: Wednesday, 17th January 2024 2:45:43 am
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import json
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 初始化分布式环境


def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


# 安装所需库（如果尚未安装）
# !pip install transformers nltk



def preprocess_text(text):
    # 基本文本清洗和去除停止词
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])

class ReviewDataset(Dataset):
    def __init__(self, reviews, tokenizer, max_len=512):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }


if __name__ == "__main__":
    # load rank from env
    import os
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"rank: {rank}, world_size: {world_size}")
