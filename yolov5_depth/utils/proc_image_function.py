# -*- coding: utf-8 -*-
# @Time: 3月 29, 2022
# ---
# calculate the loss or the difference between two images with MSE, SIMSE, OrthoLoss
import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def depth_read(file_name):
    # loads depth map D from png file from KITTI
    # and returns it as a numpy array,
    # for details see readme.txt

    # Image(W*H) ->numpy.array(H*W) Shape transpose
    depth_png = np.array(Image.open(file_name), dtype=int)
    # make sure we have a proper 16bit depth map here... not 8bit!
    assert (np.max(depth_png) > 255)
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class OrthoLoss(nn.Module):

    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2 = input1
        input2_l2 = input2

        ortho_loss = 0
        dim = input1.shape[1]
        for i in range(input1.shape[0]):
            ortho_loss += torch.mean(((input1_l2[i:i + 1, :].mm(input2_l2[i:i + 1, :].t())).pow(2)) / dim)
        ortho_loss = ortho_loss / input1.shape[0]

        return ortho_loss


# 直方图  请直接使用opencv代码
def calc_gray_hist(img):
    rows, cols = img.shape
    gray_hist = np.zeros([256])
    for r in range(rows):
        for c in range(cols):
            gray_hist[int(img[r, c])] += 1

    return gray_hist


def gray_stretch(img, thrd_min=0, thrd_max=255):
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = cv2.copyTo(img, None)
    min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(gray_img)
    output = np.uint8(255. / (max_val - min_val) * (gray_img - max_val) + 0.5)

    return output


def gamma_trans(img, gamma=1):
    if gamma == 1:
        mean_val = np.mean(img)
        if mean_val > 128:
            gamma = 2
        else:
            gamma = np.log10(0.5) / np.log10(mean_val / 255.)
        print(gamma)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


# 图像分割方法
def max_entropy(img, g_p=3, filter_size=3):
    thresh = 0
    height, width = img.shape
    total_pixel = height * width
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    sum_hist = np.cumsum(hist)
    max_en = 0.0

    for i in range(256):
        h_front = 0.0
        h_back = 0.0
        front_num = 0
        for j in range(i):
            front_num = sum_hist[i]
            if hist[j] != 0:
                probability = float(hist[j]) / front_num
                h_front += probability * np.log2(1. / probability)
        for k in range(i, 256):
            if hist[k] != 0:
                probability = float(hist[k]) / (total_pixel - front_num)
                h_back += probability * np.log2(1. / probability)
        entropy_ = h_front + h_back
        if max_en < entropy_:
            max_en = entropy_
            thresh = i + g_p
    # _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.copyTo(img, None)
    thresh_img[thresh_img < thresh] = 0
    thresh_img[thresh_img >= thresh] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filter_size, filter_size))
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    # cv2.putText(thresh_img, str(thresh), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    # cv2.imshow("1", thresh_img)
    return thresh_img, thresh


# if __name__ == "__main__":
#     path = r"D:/ResourceLib_Datasets/dataset/TNO_Image_Fusion_Dataset/FEL_images/Duine_sequence/thermal/"
#     filenames = os.listdir(path)
#     for filename in filenames:
#         filepath = os.path.join(path, filename)
#         image = cv2.imread(filepath)
#         image = gray_stretch(image)
#         image = cv2.resize(image, (640, 512))
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = cv2.copyTo(image, None)
#         g_img = gamma_trans(gray)
#         g_img = cv2.ximgproc.guidedFilter(g_img, g_img, 3, 500)
#         thre_img, thre = max_entropy(g_img)
#         cv2.imshow("0", thre_img)
#         cv2.waitKey()
