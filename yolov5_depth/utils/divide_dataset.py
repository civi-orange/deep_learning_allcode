import os
from shutil import copy, rmtree
import random
from tqdm import tqdm

data_root = r'E:/KITTI/ObjectDetection/detect_depth_estimate'


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.15

    # 源目录
    origin_image_path = os.path.join(data_root, "training", "image_2")
    assert os.path.exists(origin_image_path), "path '{}' does not exist.".format(origin_image_path)
    origin_label_path = os.path.join(data_root, "training", "yolo_label_2")
    assert os.path.exists(origin_label_path), "path '{}' does not exist.".format(origin_label_path)

    # 建立保存训练集的文件夹
    train_img = os.path.join(data_root, "yolo_data", "images", "train")
    mk_file(train_img)
    train_label = os.path.join(data_root, "yolo_data", "labels", "train")
    mk_file(train_label)

    # 建立保存验证集的文件夹
    val_img = os.path.join(data_root, "yolo_data", "images", "val")
    mk_file(val_img)
    val_label = os.path.join(data_root, "yolo_data", "labels", "val")
    mk_file(val_label)

    images = os.listdir(origin_image_path)
    img_num = len(images)
    eval_image = random.sample(images, k=int(img_num * split_rate))
    for index, image in enumerate(tqdm(images)):
        image_path = os.path.join(origin_image_path, image)
        label = image[:-4] + ".txt"
        label_path = os.path.join(origin_label_path, label)
        if image in eval_image:
            copy(image_path, val_img)
            copy(label_path, val_label)
        else:
            copy(image_path, train_img)
            copy(label_path, train_label)


if __name__ == '__main__':
    # 数据集制作，按比例拆分训练集集、验证集
    main()
    print("processing done!")
