import os
import json
from shutil import copy


def rename_all_file(file_path: str, pre_: str):  # 批量给文件增加前缀
    file_names = os.listdir(file_path)  # 获取文件夹内所有文件的名字
    for name in file_names:  # 如果某个文件名在file_names内
        old_name = file_path + '/' + name  # 获取旧文件的名字，注意名字要带路径名
        new_name = file_path + '/' + pre_ + "_" + name  # 定义新文件的名字，这里给每个文件名前加了前缀 a_
        os.rename(old_name, new_name)  # 用rename()函数重命名
        # print(new_name)  # 打印新的文件名字


'''
本文件是用来将场景数据集places365_standard分成城市urban和郊外outdoor两种分类的脚本
running this file,
the datasets of places365_standard will be 
divided into two categories: outdoor and urban
'''
image_class_names = []
train_dir_pre = "D:/WorkData/dataset/SceneRecognition/places365_standard/train"
val_dir_pre = "D:/WorkData/dataset/SceneRecognition/places365_standard/val"
dst_train_file_path = os.path.join(train_dir_pre, "..", "new_train")
dst_val_file_path = os.path.join(train_dir_pre, "..", "new_val")
# x = os.listdir(dst_train_file_path)
# y = os.listdir(dst_val_file_path)


# for file in os.listdir(train_dir_pre):
#     image_class_names.append(file)
# image_class_names.sort()  # 只有字符的情况下与Windows自然排序一致
# print(image_class_names[145])
# class_list = [
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 0
#     1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
#     1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
#     1, 1, 1, 1, 1, 1, 1, 0, 0, 0,  # 50
#     1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
#     1, 1, 1, 0, 1, 1, 0, 1, 0, 1,
#     1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
#     1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
#     1, 1, 1, 1, 0, 0, 1, 1, 1, 1,  # 100
#     0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
#     0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
#     0, 0, 0, 0, 1, 1, 1, 1, 1, 1,  #150
#     1, 1, 1, 0, 0, 0, 1, 0, 1, 1,
#     1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
#     0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
#     0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
#     1, 1, 1, 1, 0, 0, 1, 1, 1, 0,  # 200
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
#     1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
#     1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
#     1, 1, 1, 1, 1, 1, 1, 1, 0, 1,  # 250
#     1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
#     1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
#     1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
#     1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1, 0, 0, 1, 1, 0,  # 300
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
#     1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
#     0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
#     0, 1, 1, 1, 1, 0, 0, 0, 1, 0,  # 350
#     0, 1, 1, 1, 1,
# ]  # 1: urban, 0: wild or outdoor
# class_temp = dict()
# # 满足assert条件时，程序才继续执行
# assert len(image_class_names) == len(class_list), "The list lengths are not equal"
# # 获得分类字典
# for i in range(len(image_class_names)):
#     class_temp[image_class_names[i]] = class_list[i]
# json_str = json.dumps(class_temp, indent=4)
# with open("classify_file.json", "w") as json_file:
#     json_file.write(json_str)

dst_train_file_path_urban = os.path.join(dst_train_file_path, "urban")
dst_train_file_path_outdoor = os.path.join(dst_train_file_path, "outdoor")
dst_val_file_path_urban = os.path.join(dst_val_file_path, "urban")
dst_val_file_path_outdoor = os.path.join(dst_val_file_path, "outdoor")

with open('./classify_file.json', 'r') as jf:
    dict_class = json.load(jf)
    for k, v in dict_class.items():
        src_file_path_t = os.path.join(train_dir_pre, k)
        images_t = os.listdir(src_file_path_t)

        src_file_path_v = os.path.join(val_dir_pre, k)
        images_v = os.listdir(src_file_path_v)

        if v == 0:  # outdoor
            for image in images_t:
                image_path = os.path.join(src_file_path_t, image)
                new_image = k + "_" + image
                copy(image_path, os.path.join(dst_train_file_path_outdoor, new_image))
            for image in images_v:
                image_path = os.path.join(src_file_path_v, image)
                new_image = k + "_" + image
                copy(image_path, os.path.join(dst_val_file_path_outdoor, new_image))
        if v == 1:  # urban
            for image in images_t:
                image_path = os.path.join(src_file_path_t, image)
                new_image = k + "_" + image
                copy(image_path, os.path.join(dst_train_file_path_urban, new_image))
            for image in images_v:
                image_path = os.path.join(src_file_path_v, image)
                new_image = k + "_" + image
                copy(image_path, os.path.join(dst_val_file_path_urban, new_image))

print("Finish!")
# if os.path.splitext(file)[1] == ".jpg":  # 0:filename, 1:filename extension
