# -*- coding: utf-8 -*-
# @Time: 3月 28, 2022
# ---
import os

import matplotlib.pyplot
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from PIL import Image

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# x = np.arange(-np.pi, np.pi, 0.001)
# y = np.sin(x)
# plt.xlabel("X")
# plt.ylabel("Y")
# # plt.ylim(-1000, 1000)
# plt.title("y = tan(x)")
# plt.plot(x, y)
# plt.show()




