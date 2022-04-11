# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 15:16
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : toWangBo.py
# @Software: PyCharm


import cv2
import numpy as np
np.set_printoptions(suppress=True)

image_path = ['C:/Users/zhangdan/Desktop/m2python/000000.png']
calib_path = ['C:/Users/zhangdan/Desktop/m2python/000000.txt']
bin_path = ['C:/Users/zhangdan/Desktop/m2python/000000.bin']

FileNumbers = 1

for i in range(FileNumbers):
    print('正在处理%d/%d' % (i + 1, FileNumbers))

    file_name = image_path[i]
    I = cv2.imread(file_name)

    # 读txt
    with open(calib_path[i], 'r') as f:
        raw_data = f.readlines()

    datas = []
    names = []
    for j in range(len(raw_data) - 1):  # .txt最后有一个空格
        sub_raw_data = raw_data[j].strip().split(' ')
        names.append(sub_raw_data[0])
        datas.append(list(map(float, sub_raw_data[1:])))
    P0 = np.array(datas[0]).reshape(3, 4)
    P1 = np.array(datas[1]).reshape(3, 4)
    P2 = np.array(datas[2]).reshape(3, 4)
    P3 = np.array(datas[3]).reshape(3, 4)
    R0_rect = np.array(datas[4]).reshape(3, 3)
    Tr_velo_to_cam = np.array(datas[5]).reshape(3, 4)
    Tr_imu_to_velo = np.array(datas[6]).reshape(3, 4)

    # 读bin
    velo = np.fromfile(bin_path[i], dtype=np.float32, count=-1).reshape(-1, 4)

    idx = np.where(velo[:, 0] >= 5)[0]  # 满足<5条件的位置
    velo = velo[idx]
    velo[:, 3] = 1

    R0_rect_ = np.zeros((4, 4))
    R0_rect_[:R0_rect.shape[0], :R0_rect.shape[1]] = R0_rect
    R0_rect_[3, 3] = 1
    Tr_velo_to_cam_ = np.zeros((4, 4))
    Tr_velo_to_cam_[:Tr_velo_to_cam.shape[0], :Tr_velo_to_cam.shape[1]] = Tr_velo_to_cam
    Tr_velo_to_cam_[3, 3] = 1
    P = np.matmul(np.matmul(P2, R0_rect_), Tr_velo_to_cam_)

    px = np.matmul(P, velo.T).T

    px[:, 0] /= px[:, 2]
    px[:, 1] /= px[:, 2]

    n, m, k = I.shape
    r = 1 - (np.array(px[:, 0] < 1) + \
        np.array(px[:, 0] >= m) + \
        np.array(px[:, 1] >= n))
    remain_ids = np.where(r)[0]
    px = px[remain_ids]

    def dense_depth_map(pts, n, m, grid=4):
        mD = np.zeros((n, m))
        mD[np.round(pts[:, 1]).astype('int')-1, np.round(pts[:, 0]).astype('int')-1] = pts[:, 2]
        return mD

    depth = dense_depth_map(px, n, m, 4)
    depth /= 73
    cv2.imshow('depth', depth)
    cv2.waitKey(0)
