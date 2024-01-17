###
 # File: /load_feature.py
 # Created Date: Wednesday January 17th 2024
 # Author: Zihan
 # -----
 # Last Modified: Wednesday, 17th January 2024 3:22:15 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import torch
import os
import numpy as np

# 文件夹路径
folder_path = '/home/zihan/dataset/features'

# 获取所有 .pt 文件的路径
file_paths = [os.path.join(folder_path, f)
              for f in os.listdir(folder_path) if f.endswith('.pt')]

# 打印文件
print(file_paths)

# 加载并拼接 tensors
all_tensors = []
for file_path in file_paths:
    tensor = torch.load(file_path, map_location=torch.device('cpu'))
    all_tensors.append(tensor)

# 拼接成一个大的 tensor
big_tensor = torch.cat(all_tensors, dim=0)

# save the big tensor
numpy_array = big_tensor.numpy()

output_file = '/home/zihan/dataset/features/feature.npy'

np.save(output_file, numpy_array)

print(f'Saved to {output_file}')
