# -*- python coding: utf-8 -*-
# @Time: 4月 02, 2022
# ---
import glob
import string
import os
from xml.dom.minidom import Document
import cv2

txt_list = glob.glob(r'E:/KITTI/ObjectDetection/detect_depth_estimate/training/label_2/*.txt')


def show_category(txt_list):
    category_list = []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ')  # 去掉前后多余的字符并把其分开
                    category_list.append(labeldata[0])  # 只要第一个字段，即类别
        except IOError as ioerr:
            print('File error:' + str(ioerr))
    print(set(category_list))  # 输出集合


show_category(txt_list)


def merge(line):
    each_line = ''
    for i in range(len(line)):
        if i != (len(line) - 1):
            each_line = each_line + line[i] + ' '
        else:
            each_line = each_line + line[i]  # 最后一条字段后面不加空格
    each_line = each_line + '\n'
    return (each_line)


print('before modify categories are:\n')
show_category(txt_list)

for item in txt_list:
    new_txt = []
    try:
        with open(item, 'r') as r_tdf:
            for each_line in r_tdf:
                labeldata = each_line.strip().split(' ')
                if labeldata[0] in ['Car', 'Truck', 'Van', 'Tram']:  # 合并卡车类
                    labeldata[0] = labeldata[0].replace(labeldata[0], 'truck')
                if labeldata[0] in ['Pedestrian', 'Person_sitting', 'Cyclist']:  # 合并行人类
                    labeldata[0] = labeldata[0].replace(labeldata[0], 'person')
                if labeldata[0] in ['DontCare', 'Misc']:  # 忽略Dontcare类、Misc类
                    continue
                new_txt.append(merge(labeldata))  # 重新写入新的txt文件
        with open(item, 'w+') as w_tdf:  # w+是打开原文件将内容删除，另写新内容进去
            for temp in new_txt:
                w_tdf.write(temp)
    except IOError as ioerr:
        print('File error:' + str(ioerr))

print('\nafter modify categories are:\n')
show_category(txt_list)


def generate_xml(name, split_lines, img_size, class_ind):
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    title = doc.createElement('folder')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    annotation.appendChild(title)
    img_name = name + '.png'
    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)
    source = doc.createElement('source')
    annotation.appendChild(source)
    title = doc.createElement('database')
    title_text = doc.createTextNode('The KITTI Database')
    title.appendChild(title_text)
    source.appendChild(title)
    title = doc.createElement('annotation')
    title_text = doc.createTextNode('KITTI')
    title.appendChild(title_text)
    source.appendChild(title)
    size = doc.createElement('size')
    annotation.appendChild(size)
    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for split_line in split_lines:
        line = split_line.strip().split()
        if line[0] in class_ind:
            object = doc.createElement('object')
            annotation.appendChild(object)

            title = doc.createElement('name')
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[5]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[6]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[7]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open('/home/huichang/wj/yolo_learn/kitti_VOC/Annotations/' + name + '.xml', 'w')  # create a new xml file
    f.write(doc.toprettyxml(indent=''))
    f.close()


# #source code
if __name__ == '__main__':
    class_ind = ('person', 'car', 'truck')
    # cur_dir=os.getcwd()  # current path
    # labels_dir=os.path.join(cur_dir,'labels') # get the current path and build a new path.and the result is'../yolo_learn/labels'
    labels_dir = '/home/huichang/wj/yolo_learn/kitti/training (copy)/label_2'
    for parent, dirnames, filenames in os.walk(labels_dir):  # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)  # 获取文件全路径
            f = open(full_path)
            split_lines = f.readlines()
            name = file_name[:-4]  # 后四位是扩展名.txt，只取前面的文件名
            img_name = name + '.png'
            img_path = os.path.join('/home/huichang/wj/yolo_learn/kitti/data_object_image_2/training/image_2',
                                    img_name)  # 路径需要自行修改
            print(img_path)
            img_size = cv2.imread(img_path).shape
            generate_xml(name, split_lines, img_size, class_ind)
print('all txts has converted into xmls')
