###
 # File: /read_amazon.py
 # Created Date: Wednesday January 17th 2024
 # Author: Zihan
 # -----
 # Last Modified: Wednesday, 17th January 2024 3:44:14 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import json
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# 定义处理单个文件的函数
def process_file(file_name):
    reviews_data = []
    category = file_name.split('_')[1]
    
    with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
        for line in file:
            review = json.loads(line)
            reviews_data.append({'reviewText': review['reviewText'], 'category': category})

    df = pd.DataFrame(reviews_data)
    df['length'] = df['reviewText'].apply(len)
    return df.nlargest(1000, 'length')[['reviewText', 'category']]

# 初始化一个空的 DataFrame 用于存储所有评论及其类别
all_reviews = pd.DataFrame(columns=['reviewText', 'category'])

# 指定文件夹路径
folder_path = 'Amazon'

# 列出所有符合条件的文件
file_list = [f for f in os.listdir(folder_path) if f.startswith('reviews') and f.endswith('_5.json')]

# 使用 ThreadPoolExecutor 并行处理文件
with ThreadPoolExecutor() as executor:
    results = executor.map(process_file, file_list)

# 合并结果
for result in results:
    all_reviews = pd.concat([all_reviews, result])

# 保存为 JSON 文件
output_file = 'combined_reviews_with_category.json'
try:
    all_reviews.to_json(output_file, orient='records', lines=True)
    print(f"评论已成功保存到文件 '{output_file}'。")
except Exception as e:
    print(f"保存文件时发生错误: {e}")

# 打印总选择数目
print("总选择评论数目:", len(all_reviews))
