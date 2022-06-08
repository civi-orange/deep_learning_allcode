# -*- python coding: utf-8 -*-
# @Time: 5月 27, 2022
# ---
import threading
import numpy as np
import cv2
import sys
sys.path.append("./MvImport")
from MvImport.MvCameraControl_class import *

from Grab_deal import *


# get the image buffer data from sdk
def readimage(cam=0):
    stOutFrame = MV_FRAME_OUT()
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    cam.MV_CC_GetIntValue("PayloadSize", stParam)
    datasize = stParam.nCurValue
    pdata = (c_ubyte * datasize)
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if None != stOutFrame.pBufAddr and 0 == ret:
        # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
        #     stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
        cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                           stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
        data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                             dtype=np.uint8)
        image = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
    return image


# 线程获取结果
class MyThread(threading.Thread):
    def __init__(self, number):
        threading.Thread.__init__(self)
        self.number = number

    def run(self):
        self.result = readimage(self.number)

    def get_result(self):
        return self.result


if __name__ == "__main__":
    enumcccc = EnumMVCamera()
    cam = enumcccc.init_camera(camera_index=0)

    while True:
        t = MyThread(cam)
        t.start()
        t.join()
        frame = t.get_result()
        cv2.imshow("1", frame)
        cv2.waitKey(1)
cv2.destroyAllWindows()
