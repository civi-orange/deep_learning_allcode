# -*- python coding: utf-8 -*-
# @Time: 5月 30, 2022
# ---

import cv2, glob, os
from Grab_deal import *

# cap = cv2.VideoCapture(0)
# while True:
#     _, frame = cap.read()
#     cv2.imshow("win", frame)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     p, r = gen_circle_center(gray, "233")
#     frame_n = cv2.resize(gray, (1280, 1024))
#     cv2.imshow("win1", frame_n)
#     try:
#         p, r = gen_circle_center(frame_n)
#         print(p)
#         print(r)
#     except:
#         print("There is no circle in image!")
#
#     cv2.waitKey(1)


# list = glob.glob(r"D:\ShowMeTheCode\pyProject\SPAAM_AND_IMG2IMG\calib_img\c0530\frgb\*.bmp")
# from tqdm import tqdm
# for i in tqdm(range(len(list))):
#     img_path = list[i]
#     image = cv2.imread(img_path)
#     kx = 1280 / 640.0
#     ky = 1024 / 480.0
#     print(kx, "--", ky)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray_big = cv2.resize(gray, None, fx=kx, fy=ky)
#     file_name = img_path[-14:]
#     file = os.path.join(r"D:\ShowMeTheCode\pyProject\SPAAM_AND_IMG2IMG\calib_img\c0530\frgbbig", file_name)
#
#     cv2.imwrite(file, gray_big)
#     cv2.waitKey(1)
# cv2.destroyAllWindows()


import numpy as np
import cv2


def main():
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()
        _, frame1 = cap1.read()
        cv2.imshow('Video', frame)
        cv2.imshow('Video1', frame1)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
