# -*- python coding: utf-8 -*-
# @Time: 5月 31, 2022
# ---
# -*- coding=GBK -*-
import cv2
import cv2 as cv


# 高斯金字塔
def pyramid_image(image):
    level = 3  # 金字塔的层数
    temp = image.copy()  # 拷贝图像
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("高斯金字塔" + str(i), dst)
        temp = dst.copy()
    return pyramid_images


# 拉普拉斯金字塔
def laplian_image(image):
    pyramid_images = pyramid_image(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("拉普拉斯" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow("拉普拉斯" + str(i), lpls)


def gaussian(ori_image, down_times=5):
    # 1：添加第一个图像为原始图像
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        # 2：连续存储5次下采样，这样高斯金字塔就有6层
        temp_gau = cv2.pyrDown(temp_gau)
        cv.imshow("g" + str(i), temp_gau)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid


def laplacian(gaussian_pyramid, up_times=5):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        # i的取值为5,4,3,2,1,0也就是拉普拉斯金字塔有6层
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
        cv.imshow("l" + str(i), temp_lap)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid


src = cv.imread(r"C:\Users\37236\Pictures\Camera Roll\3.jpg", cv2.IMREAD_GRAYSCALE)
print(src.shape)
cv.imshow("0", src)
g_src = gaussian(src, 4)
laplacian(g_src, 4)
cv.waitKey(0)
cv.destroyAllWindows()
