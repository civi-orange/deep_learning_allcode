# -*- python coding: utf-8 -*-
# @Time: 3月 31, 2022
# ---
import glob
import os
import cv2
import torch
import mayavi.mlab
import numpy as np
from tqdm import tqdm
from PIL import Image

train_image_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/image_2"  # train image
train_label_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/label_2"  # KITTI labels
train_velo_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/velodyne"  # lidar data
test_image_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/testing/image_2"  # test image
test_velo_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/testing/velodyne"  # lidar data
calib_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/calib"
yolo_label_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/yolo_label_2"  # yolo_label

depth_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/depth"

if not os.path.exists(yolo_label_path):
    os.mkdir(yolo_label_path)


txt_list = glob.glob(r'E:/KITTI/ObjectDetection/detect_depth_estimate/training/label_2/*.txt')
velo_list = glob.glob(r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/velodyne/*.bin")

'''# 注释表示已转换
# 将分开的项，重新合并
def merge(line):
    each_line = ''
    for i in range(len(line)):
        if i != (len(line) - 1):
            each_line = each_line + line[i] + ' '
        else:
            each_line = each_line + line[i]  # 最后一条字段后面不加空格
    each_line = each_line + '\n'
    return (each_line)


# 重新定义分类（合并车，合并人）
for item in txt_list:
    new_txt = []
    try:
        with open(item, 'r') as r_tdf:
            for each_line in r_tdf:
                labeldata = each_line.strip().split(' ')
                if labeldata[0] in ['Car', 'Truck', 'Van', 'Tram']:  # 合并车类
                    labeldata[0] = labeldata[0].replace(labeldata[0], 'truck')
                if labeldata[0] in ['Pedestrian', 'Person_sitting', 'Cyclist']:  # 合并行人类
                    labeldata[0] = labeldata[0].replace(labeldata[0], 'person')
                if labeldata[0] in ['DontCare', 'Misc']:  # 忽略Dontcare类、Misc类
                    continue
                new_txt.append(merge(labeldata))  # 重新写入新的txt文件
        with open(item, 'w+') as w_tdf:  # w+是打开原文件将内容删除，另写新内容进去
            for temp in new_txt:
                w_tdf.write(temp)
    except IOError as ioerr:
        print('File error:' + str(ioerr))
'''


def label_kitti2yolo():  # add the depth at the end
    for txt in tqdm(txt_list):
        file_name = os.path.basename(txt)
        yolo_txt = os.path.join(yolo_label_path, file_name)
        depth_name = file_name[:-4] + '.png'
        depth_file = os.path.join(depth_path, depth_name)
        depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        image_path = os.path.join(train_image_path, depth_name)
        img = cv2.imread(image_path)
        height, width, channel = img.shape

        with open(txt, 'r') as kitti_f:
            lines = kitti_f.readlines()
            for line in lines:
                # cls, cut, shelter, alpha, box(x1,y1,x2,y2), (h,w,len), (x,y,z)in camera, rotation_y, conf
                line = line.strip().split()
                # cls, box(x1,y1,x2,y2)
                if line[0] == 'person':
                    cls = str(0)
                elif line[0] == 'truck':
                    cls = str(1)
                box = [float(line[4]), float(line[5]), float(line[6]), float(line[7])]
                rect = np.array(box).astype(int)
                depth_rect = depth_map[rect[1]: rect[3], rect[0]:rect[2]]
                depth_rect_ = depth_rect[depth_rect != 0]
                mid = np.median(depth_rect_)
                # mean = np.mean(depth_rect_)
                if np.isnan(mid):
                    mid = 100*1000  # 没有深度数据内容
                x = (float(line[6]) + float(line[4])) / 2. / width
                y = (float(line[7]) + float(line[5])) / 2. / height
                w = (float(line[6]) - float(line[4])) / width
                h = (float(line[7]) - float(line[5])) / height
                d = mid / 1000.0
                target = [cls, str(x), str(y), str(w), str(h), str(d)]

                with open(yolo_txt, 'a') as yolo_f:
                    for tar in target:
                        yolo_f.write(tar + ' ')
                    yolo_f.write('\n')


def viz_mayavi(points, vals="distance"):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = points[:, 3]
    d = torch.sqrt(x ** 2 + y ** 2)

    if vals == "height":
        col = z
    else:
        col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    mayavi.mlab.points3d(x, y, z,
                         col,
                         mode="point",
                         colormap='spectral',
                         figure=fig,
                         )

    mayavi.mlab.show()


def get_target_depth(depth_map, box):
    rect = np.array(box).astype(int)
    depth_rect = depth_map[rect[1]: rect[3], rect[0]:rect[2]]
    depth_rect_ = depth_rect[depth_rect != 0]
    t_depth = np.median(depth_rect_)
    # t_depth = np.mean(depth_rect_)
    return t_depth


if __name__ == '__main__':
    # point_cloud = np.fromfile(r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/velodyne/000010.bin", dtype=np.float32, count=-1).reshape([-1, 4])
    # point_cloud = torch.from_numpy(point_cloud)
    # print(point_cloud.size())
    # print(point_cloud.type())
    # viz_mayavi(point_cloud, vals='height')

    label_kitti2yolo()
