# -*- coding: utf-8 -*-
# @File: model.py
# @Time: 2021/12/26
# --- ---
import torch
import torch.nn as nn


class BasicBlock(nn.Module):  # 18 34
    expansion = 1

    def __init__(self, channel_in, channel_out, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 第一层
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 第一个卷积
        self.bn1 = nn.BatchNorm2d(channel_out)
        # 激活函数
        self.relu = nn.ReLU()
        # 第二层
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 中间层的卷积
        self.bn2 = nn.BatchNorm2d(channel_out)
        # 下采样
        self.downsample = downsample  # 是否有下采样

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 50 101 152
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 channel_in,
                 channel_out,
                 stride=1,
                 downsample=None,
                 groups=1,
                 width_per_group=64
                 ):
        super(BottleNeck, self).__init__()

        # if the args:groups and width_per_group is default value it is resnet, other is resneXt
        width = int(channel_out * (width_per_group / 64.0)) * groups  #

        # 降维
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 特征提取
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        # 升维拼接
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=channel_out * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(channel_out * self.expansion)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)  # inplace=true 直接覆盖修改值 节省内存使用

        # 下采样
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

    '''
    关于计算卷积的公式：  out = (in - filter + 2*padding) / stride + 1
    filter：滤波核大小；padding：扩充边界；stride：卷积步长
    感受也计算公式：上一层 = 当前层 +（滤波核filter-1）* 计算层到当前层的累积步长stride乘积
    '''


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 block_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.channel_in = 64  # 某一组残差结构第一层输入

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.channel_in, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel_in)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer1 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer1 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer1 = self._make_layer(block, 512, block_num[3], stride=2)

        if self.include_top:  # 是否使用输出，False即本网络为骨干网络， True还是最终的分类网络
            self.argpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.channel_in != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.channel_in, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )  # 通过调整stride 和 channel*expansion调整shortcut的深度和特征map大小
        layers = []
        layers.append(block(self.channel_in,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        # channel_in
        self.channel_in = channel * block.expansion  # 此步骤将得到残差结构每层输出
        for _ in range(1, block_num):
            layers.append(block(self.channel_in,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.argpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True): #include_top 是否使用全连接层输出
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(BasicBlock, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(BottleNeck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 4
    return ResNet(BottleNeck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
