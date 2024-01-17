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
import numpy as np

# 初始化分布式环境


def download_nltk_resources(resource_name, download_dir):
    import os
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)
    nltk.download(resource_name, download_dir=download_dir)


def setup(rank, world_size):
    
    print(f"rank: {rank}, world_size: {world_size}")
    dist.init_process_group(
        backend='nccl', rank=rank, world_size=world_size)
    
    nltk_data_path = "/workspace/nltk_data"
    if rank == 0:
        # 主进程下载资源 if not exists
        if not os.path.isdir(nltk_data_path):
            os.makedirs(nltk_data_path)
            download_nltk_resources('punkt', nltk_data_path)
            download_nltk_resources('stopwords', nltk_data_path)
        else:
            print("nltk_data_path already exists")

    # 同步所有进程，确保主进程下载完成
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
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


def main(rank, world_size):
    setup(rank, world_size)

    import os
    if rank == 0:
        # 主进程：读取数据
        print("主进程：读取数据")
        with open('/workspace/combined_reviews_with_category.json', 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
            reviews = [item['reviewText'] for item in data]
            categories = [item['category'] for item in data]

        # 创建类别到数字的映射
        unique_categories = list(set(categories))
        category_to_id = {category: idx for idx, category in enumerate(unique_categories)}

        # 转换类别为数字标签
        numeric_labels = [category_to_id[category] for category in categories]

        # 预处理文本
        preprocessed_reviews = [preprocess_text(review) for review in reviews]
        # Save preprocessed reviews and numeric labels
        with open('/workspace/preprocessed_reviews.json', 'w', encoding='utf-8') as file:
            for review in preprocessed_reviews:
                file.write(review + '\n')

        with open('/workspace/doc_labels.txt', 'w', encoding='utf-8') as file:
            for label in numeric_labels:
                file.write(str(label) + '\n')

    dist.barrier()

    if rank != 0:
        print(f"从主进程读取数据 on rank {rank}")
        with open('/workspace/preprocessed_reviews.json', 'r', encoding='utf-8') as file:
            preprocessed_reviews = [line.strip()
                                    for line in file.readlines()]

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

    print(f"开始生成特征向量 on rank {rank}")

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
    torch.save(all_feature, f"/workspace/all_feature_{rank}.pt")

    dist.barrier()

    if rank == 0:
        # 主进程：读取所有特征向量文件并合并
        all_features = []
        for r in range(world_size):
            feature_part = torch.load(f"/workspace/all_feature_{r}.pt")
            all_features.append(feature_part)

        # 合并所有特征向量
        combined_features = torch.cat(all_features, dim=0)
        combined_features_np = combined_features.numpy()

        # 保存 NumPy 矩阵
        np.save('/workspace/feature_matrix.npy', combined_features_np)

    cleanup()

if __name__ == "__main__":
    import os
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # print(f"rank: {rank}, world_size: {world_size}")
    main(rank, world_size)
