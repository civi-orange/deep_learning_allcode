# -*- coding: utf-8 -*-
# @File: predict.py
# @Time: 2021/12/26
# --- ---
import json
import os
import torch
from torchvision import transforms
import model
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_path = "./test_image/tulip.jpg"
    assert os.path.exists(image_path),"file:{} is not existing.".format(image_path)
    #pil
    img = Image.open(image_path)
    plt.imshow(img)
    #cv2  接着尝试自己写cv2读取图像预测，测试需要哪些图像预处理
    image = cv2.imread(image_path)
    cv2.imshow(image)

    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # expand batch dimension

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path,"r")
    class_indict = json.load(json_file)

    model_net = model.resnet34(num_classes=5).to(device)
    weights_path = "./result/resnet34.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model_net.load_state_dict(torch.load(weights_path, map_location=device)) #map_location=True自动转换设备？

    model_net.eval()
    with torch.no_grad():
        output = model_net(img.to(device))
        output = torch.squeeze(output).cpu()
        predict = torch.softmax(output, dim=0)#预测结果
        predict_cla = torch.argmax(predict).numpy()#预测结果筛选返回索引

        # class_dict[str(predict_cla)] 值的输出与class_indices.json文件有关
        print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())

        # 弄清楚变量的含义 此时未搞清楚
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        plt.show()


if __name__ == "__main__":
    main()