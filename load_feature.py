###
# File: /load_feature.py
# Created Date: Wednesday January 17th 2024
# Author: Zihan
# -----
# Last Modified: Wednesday, 17th January 2024 4:56:03 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import numpy as np

# show the row and column of the matrix


def show_shape(matrix):
    print("The shape of the matrix is: ", matrix.shape)

# load the feature matrix


def load_feature(path):
    feature_matrix = np.load(path)
    show_shape(feature_matrix)
    return feature_matrix


if __name__ == "__main__":
    path = "/home/zihan/amazon_data/feature_matrix.npy"
    feature_matrix = load_feature(path)
    print(feature_matrix)
