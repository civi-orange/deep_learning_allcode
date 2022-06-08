# -*- python coding: utf-8 -*-
# @Time: 6月 06, 2022
# ---
import time
from ImageMapping import *
from SDKCameraRead import *
# camera image
from utils.little_function import *
import imageio

# 获取相机的内外参信息
from cameracalib.calib_info import left_Kl
from cameracalib.calib_info import left_Kr
from cameracalib.calib_info import left_R
from cameracalib.calib_info import left_T
from cameracalib.calib_info import right_Kl
from cameracalib.calib_info import right_Kr
from cameracalib.calib_info import right_R
from cameracalib.calib_info import right_T
from cameracalib.calib_info import gray_Kl
from cameracalib.calib_info import gray_Kr
from cameracalib.calib_info import gray_R
from cameracalib.calib_info import gray_T

depth = [5700]
running = False

if __name__ == "__main__":
    save_text = save_video = False
    save_calib_image = False
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    camera_class = EnumMVCamera()
    cam1 = camera_class.init_camera(camera_index=0)
    cam2 = camera_class.init_camera(camera_index=1)

    num = 0
    now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    if save_text:
        file_error_depend_on_depth = "../error_txt/" + now + r"_depth.txt"
        f_err_depth = open(file_error_depend_on_depth, "w")
    if save_video:
        io_writer = imageio.get_writer('../video/{}.mp4'.format(str(now)), fps=24)
    size_src = (640, 480)
    size_dst = (1280, 1024)
    test_map = ImageMapping(left_Kl, left_Kr, left_R, left_T, size_src, size_dst)
    test_map_right = ImageMapping(gray_Kl, gray_Kr, gray_R, gray_T, size_dst, size_dst, 1)
    while not running:
        t1 = MyThread(cam2)
        t1.start()
        t1.join()
        t2 = MyThread(cam1)
        t2.start()
        t2.join()
        _, frame_l = capture.read()
        _, frame_r = capture1.read()
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        frame_1 = t1.get_result()
        frame_2 = t2.get_result()

        num += 1
        print("The frame number is:", num)

        if not save_calib_image:
            cv2.imshow("frame_l", frame_l)
            cv2.imshow("frame_r", frame_r)
            cv2.imshow("frame_1", frame_1)
            cv2.imshow("frame_2", frame_2)
            cv2.waitKey(1)

            region = np.zeros_like(frame_l)
            size = 80
            try:
                #  The current code does not guarantee that the two sets of points correspond to each other,
                #  So we must ensure that there is only one circle in the image
                pl, _ = gen_circle_center(frame_l, "c_left")
                pr, _ = gen_circle_center(frame_1, "c_right")
            except:
                # The reason for the inaccuracy of finding the circle is the change of light and shadow
                print("There is no circle in image!")
                pl = pr = None

            # 根据点对，计算出深度：即目标离相机的距离
            try:
                if len(pl) == len(pr) and len(pl) != 0 and len(pr) != 0:
                    distance = test_map.get_depth(pl, pr)
                    print(distance)
                    # print("distance=", distance)
                    region[pl[:, 1][0] - size:pl[:, 1][0] + size, pl[:, 0][0] - size:pl[:, 0][0] + size] = 255
                    depth = distance
                    x, y = test_map.compute_error(pl, pr, depth)
                    print(x, y)
                    if save_text:
                        # 深度误差与融合误差分析
                        for depth_err in range(-1000, 3000, 100):
                            depth_wrong = depth + depth_err
                            # 计算误差
                            x_wrong, y_wrong = test_map.compute_error(pl, pr, depth_wrong)
                            txt = f_err_depth.write(
                                str(x_wrong) + "\t" + str(y_wrong) + "\t" + str(depth) + "\t" + str(
                                    [depth_err]) + "\t" + str(
                                    depth_wrong) + "\n")
                        txt = f_err_depth.write("\n\n")
                else:
                    print("没找到目标！")
                    depth = [5400]
            except:
                print("未知错误！")

            # 图像融合
            # out = test_map.fusion(frame_l, frame_1, region, depth[0])
            # cv2.imshow("out", out)
            # cv2.waitKey(1)

            region_rrr, depth_rrr = get_region_depth(frame_2, frame_1, test_map_right, 1)
            out_right = test_map_right.fusion(frame_2, frame_1, region_rrr, depth_rrr[0])
            cv2.imshow("out_right", out_right)
            cv2.waitKey(1)

            # if save_video:
            #     io_writer.append_data(out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):  # press 'q' to exit
                running = True
        else:
            img_path_rgb = "../calib_img/0606/rgb/"
            img_path_gray = "../calib_img/0606/gray/"
            img_path_left = "../calib_img/0606/left/"
            img_path_right = "../calib_img/0606/right/"
            cv2.imshow("f1", frame_2)
            cv2.imshow("f2", frame_r)
            # cv2.imwrite(img_path_right + "1/f_{:0>6}.bmp".format(num), frame_2)
            # cv2.imwrite(img_path_right + "2/rgb_{:0>6}.bmp".format(num), frame_r)
            # cv2.waitKey(2000)
            cv2.waitKey(1)

    if save_text:
        f_err_depth.close()
    if save_video:
        io_writer.close()
cv2.destroyAllWindows()
