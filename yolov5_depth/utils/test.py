# -*- python coding: utf-8 -*-
# @Time: 4月 08, 2022
# ---

import numpy as np






'''
import glob
from tqdm import tqdm
# scan the txt is empty and the value is ok?
# path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/yolo_data/labels/train/*.txt"
# txt_list = glob.glob(path)
# a = []
# for txt in tqdm(txt_list):
#     with open(txt, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             if not line:
#                 print('error!')
#             line = line.strip().split()
#             a.append(line[-1])
#             if float(line[-1]) > 100:
#                 print(line[-1])
# print(len(a))


# scan the num of layers and parameters, need run the yolo.py acquire the model (n s m l x)
# import torch
# sum_ = 0
# n = 0
# mm = torch.load("../mms.pt")
# for name, param in mm.named_parameters():
#     mul = 1
#     for size_ in param.shape:
#         mul *= size_							# 统计每层参数个数
#     sum_ += mul									# 累加每层参数个数
#     print('%14s : %s' % (name, param.shape))  	# 打印参数名和参数数量
#     # print('%s' % param)						# 这样可以打印出参数，由于过多，我就不打印了
#     print(sum_)
#     n += 1
# print(n)
'''
