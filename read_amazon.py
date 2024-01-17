###
 # File: /read_amazon.py
 # Created Date: Wednesday January 17th 2024
 # Author: Zihan
 # -----
 # Last Modified: Wednesday, 17th January 2024 9:46:37 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import json
import pandas as pd

# 用于存储 reviewText 的列表
review_texts = []

# 打开 JSON 文件并逐行读取
with open('Amazon/reviews_Musical_Instruments_5.json', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析 JSON 数据
        review = json.loads(line)
        # 添加 reviewText 到列表
        review_texts.append(review['reviewText'])

# 创建 DataFrame
df = pd.DataFrame(review_texts, columns=['reviewText'])

# 输出 DataFrame 的前几行进行检查
print(df.head())

# 计算条目数量
num_entries = len(df)
print("总条目数量:", num_entries)
