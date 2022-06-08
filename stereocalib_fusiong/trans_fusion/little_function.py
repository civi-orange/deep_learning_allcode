# -*- python coding: utf-8 -*-
# @Time: 5月 18, 2022
# ---
import cv2
from tqdm import tqdm
import os
import numpy as np


# picture to video class
class Picture2Video:
    def __init__(self, root_path='.', realtime=False, video_name=None, fps=24, size=None):
        if realtime:
            assert size is not None, "size is None"
            h, w = size
        else:
            self.paths = self.__load_picture(root_path)
            frame = cv2.imread(self.paths[0], cv2.IMREAD_GRAYSCALE)
            h, w = frame.shape  # 需要转为视频的图片的尺寸

        self.save_path = root_path + "/VideoTest.mp4" if video_name is None else os.path.join(root_path, video_name)
        if not self.save_path.endswith('mp4'):
            self.save_path += '.mp4'
        self.video = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'AVC1'), fps, (w, h), True)

    def __load_picture(self, root_path):
        paths = [d for d in os.listdir(root_path) if d.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'))]
        paths.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
        paths = [os.path.join(root_path, p) for p in paths]
        return paths

    def toVideo(self, start_frame=0, end_frame=-1):
        end_frame = len(self.paths) if end_frame == -1 else min(end_frame, len(self.paths))
        assert end_frame >= start_frame, "保存帧数不对"
        for i in tqdm(range(start_frame, end_frame)):
            img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
            self.video.write(img)

        print('保存完毕!')
        print('保存地址为：', self.save_path)

    def pic2video(self, img):
        self.video.write(img)

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()


# other function
def findcircle(frame):
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=200, param2=50, minRadius=5, maxRadius=80)
    try:
        circles = np.uint16(np.around(circles))
    except:
        print("未找到圆!")
        return []
    return circles


def gen_circle_center(frame, win_name="circle"):
    circles = findcircle(frame)
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
        # cv2.imshow(win_name, cimg)
        # cv2.waitKey(1)
        return center, radius


def save(image, num, file_path=None):
    if file_path is None:
        os.mkdir("./images")
        file_path = os.path.join("./images", r"{:>6}.bmp".format(num))
    else:
        file_path = os.path.join(file_path, r"{:>6}.bmp".format(num))
    cv2.imwrite(file_path, image)
