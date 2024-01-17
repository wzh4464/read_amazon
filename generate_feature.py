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




# 初始化 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 将模型设置为评估模式
model.eval()

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


def main(rank, world_size):

    # 加载停止词
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


        # 加载数据
    with open('/workspace/combined_reviews.json', 'r', encoding='utf-8') as file:
        reviews = [json.loads(line)['reviewText'] for line in file]

    # 预处理文本
    preprocessed_reviews = [preprocess_text(review) for review in reviews]

    setup(rank, world_size)

    # 初始化 BERT 模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 将模型设置为评估模式并转移到对应的 GPU
    model.eval()
    model.to(rank)

    # 包装模型以进行分布式训练
    model = DDP(model, device_ids=[rank])

    # 创建数据集和 DataLoader
    dataset = ReviewDataset(preprocessed_reviews, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    all_feature_vectors = []

    # 向量化评论
    with torch.no_grad():
        for data in dataloader:
            ids = data['ids'].to(rank, dtype=torch.long)
            mask = data['mask'].to(rank, dtype=torch.long)

            outputs = model(ids, attention_mask=mask)
            features = outputs[0][:, 0, :].cpu()
            all_feature_vectors.append(features)

    all_feature = torch.cat(all_feature_vectors, dim=0)
    torch.save(all_feature, '/workspace/all_feature.pt')

    cleanup()


if __name__ == "main":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
