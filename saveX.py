###
 # File: /load_feature.py
 # Created Date: Wednesday January 17th 2024
 # Author: Zihan
 # -----
 # Last Modified: Wednesday, 17th January 2024 5:52:13 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

import numpy as np
import multiprocessing
import os

# 要处理的文件路径
file_path = '~/amazon_data/feature_matrix.npy'

# 加载 NumPy 矩阵
matrix = np.load(os.path.expanduser(file_path))

# 定义每个进程的工作函数
def process_chunk(start_row, end_row, process_id):
    chunk_data = []
    for row_idx in range(start_row, end_row):
        chunk_data.extend(
            f"{row_idx},{col_idx},{value}"
            for col_idx, value in enumerate(matrix[row_idx])
            if value != 0
        )
    # 将每个进程的结果写入临时文件
    with open(f'/tmp/chunk_{process_id}.csv', 'w') as f:
        for line in chunk_data:
            f.write(line + '\n')

def save_submatrix_to_csv(matrix, subrows, subcols, file_path="subX.csv"):
    """
    保存指定子矩阵到 CSV 文件。

    :param matrix: 完整的 NumPy 矩阵。
    :param subrows: 子矩阵的行索引列表。
    :param subcols: 子矩阵的列索引列表。
    :param file_path: 保存 CSV 的文件路径，默认为 'subX.csv'。
    """
    # 提取子矩阵
    submatrix = matrix[np.ix_(subrows, subcols)]

    # 生成要写入的数据
    rows, cols = submatrix.shape
    data_to_write = [
        f"{row},{col},{submatrix[row, col]}"
        for row in range(rows)
        for col in range(cols)
        if submatrix[row, col] != 0
    ]

    # 写入 CSV 文件
    with open(file_path, 'w') as f:
        f.write("row_idx,col_idx,data\n")
        f.write("\n".join(data_to_write))

    print(f"数据已保存到 {file_path}")

def parrelel_main():
    # 分块大小
    chunk_size = len(matrix) // 48  # 假设有 48 个 CPU

    # 创建进程
    processes = []
    for i in range(48):
        start_row = i * chunk_size
        end_row = start_row + chunk_size if i != 47 else len(matrix)
        process = multiprocessing.Process(target=process_chunk, args=(start_row, end_row, i))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    # 合并临时文件到最终文件
    with open('matrix_data.csv', 'w') as output_file:
        output_file.write("row_idx,col_idx,data\n")  # 写入头部
        for i in range(48):
            with open(f'/tmp/chunk_{i}.csv', 'r') as input_file:
                output_file.writelines(input_file.readlines())
            os.remove(f'/tmp/chunk_{i}.csv')  # 删除临时文件

    print("数据已保存到 matrix_data.csv")

# 示例使用
if __name__ == "__main__":
    # 加载矩阵
    matrix = np.load(os.path.expanduser('~/amazon_data/feature_matrix.npy'))

    # 定义子矩阵的行和列索引
    subrows = list(range(1000))
    # subcols : all columns
    subcols = list(range(matrix.shape[1]))

    # 保存子矩阵到 CSV
    save_submatrix_to_csv(matrix, subrows, subcols, "submatrix.csv")
