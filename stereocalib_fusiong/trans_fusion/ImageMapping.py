# -*- python coding: utf-8 -*-
# @Time: 6月 06, 2022
# ---
import cv2
import numpy as np
from utils.little_function import *


def get_region_depth(frame_src, frame_dst, map_object, flag=0):
    """
    :param frame_src:
    :param frame_dst:
    :param map_object:
    :param flag: 0:frame_src为左图像；1：frame_dst为左图像
    :return:
    """
    depth = [5000]  # default value
    if flag == 0:
        region = np.zeros_like(frame_src)
    else:
        region = np.zeros_like(frame_dst)
    size = 80
    try:
        #  The current code does not guarantee that the two sets of points correspond to each other,
        #  So we must ensure that there is only one circle in the image
        p_src, _ = gen_circle_center(frame_src, "c_src")
        p_dst, _ = gen_circle_center(frame_dst, "c_dst")
    except:
        # The reason for the inaccuracy of finding the circle is the change of light and shadow
        print("There is no circle in image!")
        p_src = p_dst = None

    # 根据点对，计算出深度：即目标离相机的距离
    try:
        if len(p_src) == len(p_dst) and len(p_src) != 0 and len(p_dst) != 0:
            if flag == 0:
                distance = map_object.get_depth(p_src, p_dst)
            else:
                distance = map_object.get_depth(p_dst, p_src)
            print(distance)
            # print("distance=", distance)
            region[p_src[:, 1][0] - size:p_src[:, 1][0] + size, p_src[:, 0][0] - size:p_src[:, 0][0] + size] = 255
            depth = distance
            x, y = map_object.compute_error(p_src, p_dst, depth)
            print(x, y)
            # 深度误差与融合误差分析
            for depth_err in range(-1000, 3000, 100):
                depth_wrong = depth + depth_err
                x_wrong, y_wrong = map_object.compute_error(p_src, p_dst, depth_wrong)
        else:
            print("没找到目标！")
            depth = [5400]
    except:
        print("未知错误！")

    return region, depth


class ImageMapping:
    def __init__(self, Kl=None, Kr=None, R=None, T=None, size_l=(), size_r=(), flag=0):
        """
        :param Kl: 左相机内参
        :param Kr: 右相机内参
        :param R: 左相机->右相机旋转
        :param T: 左相机->右相机平移
        :param size_l: 左相机图像尺寸
        :param size_r: 右相机图像尺寸
        :param flag: 0:left->right  1:right->left
        """
        self.Kl = Kl
        self.Kr = Kr
        self.R = R
        self.T = T
        self.size_l = size_l
        self.size_r = size_r
        self.flag = flag
        self.RT = np.concatenate((self.R, self.T), axis=1)

    def image_align(self, image_src, image_dst, region=None):
        """
        Image fidelity expansion based on transform size
        :param image_src:
        :param image_dst:
        :param region:
        :return:
        """
        h_d, w_d = image_dst.shape[:2]
        h_s, w_s = image_src.shape[:2]

        wh_rate = w_d / h_d  # 保真映射长宽比
        new_h = h_s
        new_w = round(h_s * wh_rate)
        if new_w - w_s > 0:
            new_src = np.concatenate((np.zeros((new_h, new_w - w_s)), image_src), axis=1)
            new_region = np.concatenate((np.zeros((new_h, new_w - w_s)), region), axis=1)
        else:
            new_src = image_src[:, w_s - new_w:]
            new_region = region[:, w_s - new_w:]

        rate = [new_h / h_d, new_w / w_d]  # (h_rate, w_rate)
        trans = [new_h - h_s, new_w - w_s]  # 偏移量

        # new Kl
        new_Kl = self.Kl.copy()
        new_Kl[0][2] += trans[1]
        new_Kl[0][1] += trans[0]

        # new Kr
        new_Kr = self.Kr.copy()
        rate_arr = np.expand_dims(np.append(rate, 1), axis=1)
        rate_arr = np.concatenate((rate_arr, rate_arr, rate_arr), axis=1)
        new_Kr = np.multiply(new_Kr, rate_arr)

        return new_src, new_region, new_Kr, new_Kl

    @classmethod
    def xy_trans(cls, point, _Kl, Kr, RT, depth, size=(1280, 1024)):
        """
        left -> right
        :param point: 图像坐标,[[x,y],[],[]]
        :param _Kl: 左相机内参矩阵的逆矩阵
        :param Kr: 右相机内参矩阵的逆矩阵
        :param RT: 右相机相对左相机的RT矩阵即 Pl = R Pr + T
        :param depth: point的世界坐标深度
        :param size: 图像的尺寸（w,h）
        :return: mask:转换后图像坐标是否还在图像范围内，result图像坐标[[x,y,1],[]]
        """
        xy1 = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)  # [[x,y,1],[]]
        xy1 = np.transpose(xy1, (1, 0))  # [[x...],[y...],[1...]]
        xyd = _Kl.dot(xy1) * depth  # [[x...],[y...],[d...]]
        xyd1 = np.concatenate((xyd, np.ones((1, xyd.shape[1]))), axis=0)  # [[x...],[y...],[d...],[1...]]
        xyz = Kr.dot(RT.dot(xyd1))
        xyz = np.transpose(xyz, (1, 0))
        result_xyz = xyz / xyz[:, 2:]
        out = np.round(result_xyz).astype(np.int32)
        mask = np.array(out[:, 0] > 0) * np.array(out[:, 0] < size[0]) * \
               np.array(out[:, 1] > 0) * np.array(out[:, 1] < size[1])
        result = out[mask]
        return mask, result

    @classmethod
    def xy_trans_rl(cls, point, Kl, _Kr, R, T, depth, size=(1280, 1024)):
        """
        right -> left
        :param point:图像坐标,[[x,y],[],[]]
        :param Kl: 左相机内参矩阵的逆矩阵
        :param _Kr:
        :param R:
        :param T: 右相机相对左相机的RT矩阵即 Pr = R^-1(Pl - T)
        :param depth: point的世界坐标深度
        :param size: 目标图像的尺寸（w,h）
        :return: mask:转换后图像坐标是否还在图像范围内，result图像坐标[[x,y,1],[]]
        """
        xy1 = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)  # [[x,y,1],[]]
        xy1 = np.transpose(xy1, (1, 0))  # [[x...],[y...],[1...]]
        # xyz = Kl.dot(np.linalg.inv(R).dot(_Kr.dot(xy1) * depth - T))
        xyd = _Kr.dot(xy1) * depth  # [[x...],[y...],[d...]]
        xyz = Kl.dot(np.linalg.inv(R).dot(xyd - T))  # [[x...],[y...],[z...]]
        xyz = np.transpose(xyz, (1, 0))
        result_xyz = xyz / xyz[:, 2:]
        out = np.round(result_xyz).astype(np.int32)
        mask = np.array(out[:, 0] > 0) * np.array(out[:, 0] < size[0]) * \
               np.array(out[:, 1] > 0) * np.array(out[:, 1] < size[1])
        result = out[mask]
        return mask, result

    def compute_error(self, point_src, point_dst, depth, size=(1280, 1024)):
        assert len(point_src) == len(point_dst), "the points num is unequal!"
        src_xy = np.array(point_src).reshape(-1, 2)
        if self.flag == 0:
            _, dst_xyz = self.xy_trans(src_xy, np.linalg.inv(self.Kl), self.Kr, self.RT, depth, size)
        else:
            _, dst_xyz = self.xy_trans_rl(src_xy, self.Kl, np.linalg.inv(self.Kr), self.R, self.T, depth, size)

        if dst_xyz.size == 0:
            return [], []
        else:
            out = point_dst - dst_xyz[:, 0:2]
            x_error, y_error = out[0]
            return x_error, y_error

    def get_depth(self, point_left, point_right):
        """
        :param point_left: 左相机图像坐标[[x,y],[]]
        :param point_right: 右相机图像坐标
        :return: 估计深度
        """
        _R = np.linalg.inv(self.R)
        _k1 = np.linalg.inv(self.Kl)
        _k2 = np.linalg.inv(self.Kr)
        xy = np.array(point_left).reshape(-1, 2)
        xy1 = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=1).T
        uv = np.array(point_right).reshape(-1, 2)
        uv1 = np.concatenate((uv, np.ones((uv.shape[0], 1))), axis=1).T
        xyz = _k1.dot(xy1)
        uvw = _k2.dot(uv1)
        result_ = self.R.dot(xyz) - uvw
        out = np.divide(self.T, result_)
        # the value of axis=x is most correct
        depth = np.abs(np.round(out[0, :]))
        return depth

    def fusion(self, image_src, image_dst, region=None, depth=0):

        new_src, new_region, new_Kr, new_Kl = self.image_align(image_src, image_dst, region)

        _new_Kl = np.linalg.inv(new_Kl)
        h, w = new_src.shape[:2]

        yx_input = np.argwhere(new_region > 0)
        if yx_input.size == 0:
            new_region = np.full_like(new_src, 255)
            yx_input = np.argwhere(new_region > 0)
        xy_input = yx_input[:, [1, 0]]
        if self.flag == 0:
            mask, result_ = self.xy_trans(xy_input, _new_Kl, new_Kr, self.RT, depth, (w, h))
        else:
            mask, result_ = self.xy_trans_rl(xy_input, self.Kl, np.linalg.inv(self.Kr), self.R, self.T, depth, (w, h))

        src_ = xy_input[mask]
        img = np.zeros_like(new_src)
        img[(result_[:, 1], result_[:, 0])] = new_src[(src_[:, 1], src_[:, 0])]
        cv2.imshow("tttt", img)  # 马赛克需要插值
        img = cv2.resize(img, (image_dst.shape[1], image_dst.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("trans", img)
        out = image_dst.copy()
        out = cv2.addWeighted(out, 0.5, img, 0.5, 0)
        return out
