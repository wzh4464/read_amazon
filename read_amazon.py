###
 # File: /read_amazon.py
 # Created Date: Wednesday January 17th 2024
 # Author: Zihan
 # -----
 # Last Modified: Wednesday, 17th January 2024 9:52:00 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import json
import pandas as pd
import os

# 初始化一个空的 DataFrame 用于存储所有评论
all_reviews = pd.DataFrame(columns=['reviewText'])

# 指定文件夹路径
folder_path = 'Amazon'

# 列出所有符合条件的文件
file_list = [f for f in os.listdir(folder_path) if f.startswith('reviews') and f.endswith('_5.json')]

# 逐个处理每个文件
for file_name in file_list:
    try:
        # 用于存储当前文件的 reviewText
        review_texts = []

        # 打开并读取文件
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            for line in file:
                # 解析 JSON 数据
                review = json.loads(line)
                # 将 reviewText 添加到列表
                review_texts.append(review['reviewText'])

        # 创建 DataFrame 并选取最长的 1000 条评论
        df = pd.DataFrame(review_texts, columns=['reviewText'])
        df['length'] = df['reviewText'].apply(len)
        df = df.nlargest(1000, 'length')

        # 将选取的评论添加到总 DataFrame
        all_reviews = pd.concat([all_reviews, df])

        # 打印处理状态
        print(f"文件 '{file_name}' 处理成功，选取了 {len(df)} 条评论。")
    except Exception as e:
        print(f"处理文件 '{file_name}' 时发生错误: {e}")

# 删除不再需要的长度列
all_reviews.drop(columns='length', inplace=True)

# 保存为 JSON 文件
output_file = 'combined_reviews.json'
try:
    all_reviews.to_json(output_file, orient='records', lines=True)
    print(f"评论已成功保存到文件 '{output_file}'。")
except Exception as e:
    print(f"保存文件时发生错误: {e}")

# 打印总选择数目
print("总选择评论数目:", len(all_reviews))
