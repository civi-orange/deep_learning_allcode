# -*- python coding: utf-8 -*-
# @Time: 5月 23, 2022
# ---
import os
import cv2
import glob

# pathname = r"../calib_img/rgb/*.bmp"
# x = glob.glob(pathname)
# for i in x:
#     img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#     file = os.path.basename(i)
#     file2 = file.replace("rgb", "gray")
#     print(file2)
#     img = cv2.resize(img, (1280, 1024))
#     cv2.imwrite("../calib_img/gray/" + file2, img)


frame_path = r"D:\ShowMeTheCode\pyProject\SPAAM_AND_IMG2IMG\calib_img\0606\gray\1\f_000008.bmp"
while True:
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=200, param2=50, minRadius=5, maxRadius=80)
    import numpy as np
    try:
        circles = np.uint16(np.around(circles))
    except:
        print("未找到圆!")
    cimg = frame.copy()
    center = []
    radius = []
    if len(circles) != 0:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (255, 255, 255), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (255, 255, 255), 3)
            center.append(i[0])
            center.append(i[1])
            radius.append(i[2])
        center = np.array(center).reshape(-1, 2)
        print("图中圆的个数为：", len(circles))
        cv2.imshow("win_name", cimg)
        cv2.waitKey(1)


