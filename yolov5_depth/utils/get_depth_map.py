# -*- python coding: utf-8 -*-
# @Time: 4月 07, 2022
# ---
import os
import numpy as np
import glob
import cv2
from tqdm import tqdm

velo_list = glob.glob(r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/velodyne/*.bin")
calib_path = r"E:/KITTI/ObjectDetection/detect_depth_estimate/training/calib"
image_path = r'E:/KITTI/ObjectDetection/detect_depth_estimate/training/image_2'
depth_save_path_train = r'E:/KITTI/ObjectDetection/detect_depth_estimate/training/depth'

velo_list_test = glob.glob(r"E:/KITTI/ObjectDetection/detect_depth_estimate/testing/velodyne/*.bin")
image_path_test = r'E:/KITTI/ObjectDetection/detect_depth_estimate/testing/image_2'
calib_path_test = r"E:/KITTI/ObjectDetection/detect_depth_estimate/testing/calib"
depth_save_path_test = r'E:/KITTI/ObjectDetection/detect_depth_estimate/testing/depth'

calib_P2 = np.zeros((3, 4))
calib_R0 = np.eye(4)
calib_Tr = np.eye(4)
calib_labels = ['P0:', 'P1:', 'P2:', 'P3:', 'R0_rect:', 'Tr_velo_to_cam:', 'Tr_imu_to_velo:']

for velo in tqdm(velo_list_test):
    file = os.path.basename(velo)
    file = file[:-4] + '.txt'
    calib_txt = os.path.join(calib_path_test, file)
    file_im = file[:-4] + '.png'
    img_file = os.path.join(image_path_test, file_im)
    img = cv2.imread(img_file)
    height, width, channel = img.shape
    with open(calib_txt, "r") as calib_f:
        calib_line = calib_f.readline()
        while calib_line:
            calib_line = calib_f.readline()
            calib_ = calib_line.strip().split(' ')
            if calib_[0] == calib_labels[2]:
                calib_P2[:3] = np.array(calib_[1:]).reshape(3, 4)
            elif calib_[0] == calib_labels[4]:
                calib_R0[:3, :3] = np.array(calib_[1:]).reshape(3, 3)
            elif calib_[0] == calib_labels[5]:
                calib_Tr[:3] = np.array(calib_[1:]).reshape(3, 4)
    velo_2_image_1 = np.dot(calib_P2, calib_R0)
    velo_2_image = np.dot(velo_2_image_1, calib_Tr)
    velo_file = file[:-4] + ".bin"
    velo_data = np.fromfile(velo, dtype=np.float32, count=-1).reshape([-1, 4])
    idx = np.where(velo_data[:, 0] < 5)
    velo_data[:, 3] = 1
    velo_data[idx, :] = np.nan
    result = np.transpose(np.dot(velo_2_image, np.transpose(velo_data)))
    result[:, 0] = result[:, 0] / result[:, 2]
    result[:, 1] = result[:, 1] / result[:, 2]

    idx_w = np.where(result[:, 0] >= width)
    idx_h = np.where(result[:, 1] >= height)
    idx_r = np.where(result[:, 0] < 1)
    result[idx_r, :] = np.nan
    result[idx_w, :] = np.nan
    result[idx_h, :] = np.nan
    out = result[~np.isnan(result).any(axis=1)]
    out = np.c_[np.floor(out[:, 0:2]), out[:, 2]]
    depth_map = np.zeros((height, width))
    # 未完成，待编辑
    depth_map[np.int32(out[:, 1]), np.int32(out[:, 0])] = out[:, 2]
    depth_map = 1000 * depth_map
    img_16 = depth_map.astype(np.uint16)
    depth_file = os.path.join(depth_save_path_test, file_im)
    cv2.imwrite(depth_file, img_16)
    # cv2.imshow("1", img_16)
    # cv2.waitKey()
