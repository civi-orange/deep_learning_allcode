# -*- python coding: utf-8 -*-
# @Time: 4月 08, 2022
# ---

import cv2

img = cv2.imread("D:/ShowMeTheCode/pyProject/yolov5/yolov5-for-depth/data/images/bus.jpg")

print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img.shape)
