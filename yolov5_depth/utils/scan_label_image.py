# -*- python coding: utf-8 -*-
# @Time: 4月 07, 2022
# ---
import cv2
import os
import math
from tqdm import tqdm

def draw_box_in_single_image(image_path, txt_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 读取txt文件信息
    def read_list(txt_path):
        pos = []
        with open(txt_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                    pass
                # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                p_tmp = [float(i) for i in lines.strip().split(' ')]
                pos.append(p_tmp)  # 添加新读取的数据
                # Efield.append(E_tmp)
                pass
        return pos

    # txt转换为box
    def convert(size, box):
        xmin = (box[1] - (box[3] / 2.)) * size[1]
        xmax = (box[1] + (box[3] / 2.)) * size[1]
        ymin = (box[2] - (box[4] / 2.)) * size[0]
        ymax = (box[2] + (box[4] / 2.)) * size[0]
        # box1 = (int(xmin), int(ymin), int(xmax), int(ymax))
        box1 = (math.floor(xmin), math.floor(ymin), math.ceil(xmax), math.ceil(ymax))
        return box1

    pos = read_list(txt_path)
    tl = int((image.shape[0] + image.shape[1]) / 2) + 1
    lf = max(tl - 1, 1)
    for i in range(len(pos)):
        label = str(int(pos[i][0]))
        box = convert(image.shape, pos[i])
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.putText(image, label, (box[0], box[1] - 2), 0, 0.75, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
        pass

    cv2.imshow("images", image)


img_folder = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/image_2"
label_folder = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/yolo_label_2"
kitti_label_folder = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/label_2"
img_list = os.listdir(img_folder)

for img in tqdm(img_list):
    filename = os.path.basename(img)
    label = filename[:-4] + '.txt'
    label_path = os.path.join(label_folder, label)
    image_path = os.path.join(img_folder, img)
    kitti_label_path = os.path.join(kitti_label_folder, label)
    draw_box_in_single_image(image_path, label_path)

    # image_ = cv2.imread(image_path)
    # pos1 = []
    # with open(kitti_label_path, 'r') as fread:
    #     while True:
    #         lines1 = fread.readline()  # 整行读取数据
    #         if not lines1:
    #             break
    #             pass
    #         # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
    #         p_tmp1 = [ii for ii in lines1.strip().split(' ')]
    #         pos1.append(p_tmp1)  # 添加新读取的数据
    #         # Efield.append(E_tmp)
    # for i in range(len(pos1)):
    #     label_ = pos1[i][0]
    #     xmin1 = float(pos1[i][4])
    #     ymin1 = float(pos1[i][5])
    #     xmax1 = float(pos1[i][6])
    #     ymax1 = float(pos1[i][7])
    #     rect = (math.floor(xmin1), math.floor(ymin1), math.ceil(xmax1), math.ceil(ymax1))
    #     image = cv2.rectangle(image_, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
    #     cv2.putText(image_, label_, (rect[0], rect[1] - 2), 0, 0.75, [0, 0, 255], thickness=1, lineType=cv2.LINE_AA)
    # cv2.imshow("source", image_)

    cv2.waitKey()
