# -*- coding: utf-8 -*-
# @File: train.py
# @Time: 2021/12/26
# --- ---
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import model
import sys
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device))

    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data", "flower_data")
    assert os.path.exists(image_path), '{} path is not exist.'.format(image_path)

    # 数据初始化方法
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])  # 数据初始化
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
    ])

    # 载入数据 初始化处理
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=train_transform)
    train_num = len(train_dataset)
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=val_transform)
    val_num = len(val_dataset)
    print("train images num:{}, val images num:{}".format(train_num, val_num))

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 16
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('cpu works num:{}'.format(num_workers))

    # 准备数据 为训练做初始化
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    net = model.resnet34()
    model_weight_path = "./weights/resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file{} is not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载预训练权值

    channel_in = net.fc.in_features  # 获取网络输出参数
    net.fc = nn.Linear(channel_in, 5)  # 加上一个全连接层，以完成此次分类
    net.to(device)
    lr = 1e-4
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr)

    epochs = 3
    best_acc = 0.0
    save_path = "./result"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path_file = os.path.join(save_path, "resnet34.pth")
    train_steps = len(train_loader)

    for epoch in tqdm(range(epochs)):  # 进度条1
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_steps, file=sys.stdout)  # 进度条2
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images)
                loss_val = loss_function(outputs)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.sum().item())

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

            val_accurate = acc / val_num
            print("[epoch %d], train_loss: %.3f, val_accuracy:%.3f" %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path_file)
    print("train is finished")


if __name__ == '__main__':
    main()
