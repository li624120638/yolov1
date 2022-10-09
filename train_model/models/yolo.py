# author: lgx
# date: 2022/9/24 11:06
# description: Implementation for YOLO network

from math import ceil
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class YOLOv1Net(nn.Module):
    def __init__(self, **kwargs):
        super(YOLOv1Net, self).__init__()
        self.S = kwargs['S']
        self.B = kwargs['B']
        self.C = kwargs['C']

        in_channels = 3
        in_channels, self.conv1 = self.make_conv_layer(in_channels, kernel_sizes=[7], filters=[64], conv_pool=True, pool=True)
        in_channels, self.conv2 = self.make_conv_layer(in_channels, [3], [192], False, True)
        in_channels, self.conv3 = self.make_conv_layer(in_channels, [1, 3, 1, 3], [128, 256, 256, 512], False, True)
        in_channels, self.conv4 = self.make_conv_layer(in_channels, [1, 3]*4+[1, 3], [256, 512]*4+[512, 1024], False, True)
        in_channels, self.conv5 = self.make_conv_layer(in_channels, [1, 3]*2+[3, 3], [512, 1024]*2+[1024, 1024], True, False)
        in_channels, self.conv6 = self.make_conv_layer(in_channels, [3, 3], [1024, 1024], False, False)
        self.flatten = nn.Flatten()
        in_channels, self.fc1 = self.make_fc_layer(in_channels=in_channels*self.S*self.S, out_channels=4096, droupt=True)
        in_channels, self.fc2 = self.make_fc_layer(in_channels, self.S*self.S*(self.B*5+self.C), False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.conv5(res)
        res = self.conv6(res)
        res = self.flatten(res)
        res = self.fc1(res)
        res = self.fc2(res)
        # res = self.sigmoid(res)
        return res

    @staticmethod
    def make_conv_layer(in_channels, kernel_sizes: [int], filters: [int], conv_pool=False, pool=False):
        assert len(kernel_sizes) == len(filters)
        convs = []
        n = len(kernel_sizes)
        for i, (kernel_size, filter) in enumerate(zip(kernel_sizes, filters)):
            if i == n-1 and conv_pool:
                convs.append(nn.Conv2d(in_channels=in_channels, out_channels=filter,
                                       kernel_size=kernel_size, stride=2, padding=(ceil(kernel_size/2) - 1)))
            else:
                convs.append(nn.Conv2d(in_channels=in_channels, out_channels=filter,
                                       kernel_size=kernel_size, stride=1, padding='same'))
            convs.append(nn.LeakyReLU(0.1))
            convs.append(nn.BatchNorm2d(filter))
            in_channels = filter
        if pool:
            convs.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return in_channels, nn.Sequential(*convs)

    @staticmethod
    def make_fc_layer(in_channels, out_channels, droupt=False):
        fcs = []
        fcs.append(nn.Linear(in_channels, out_channels))
        if droupt:
            fcs.append(nn.Dropout(0.5))
        return out_channels, nn.Sequential(*fcs)


class YOLOv1ResNet(nn.Module):
    def __init__(self, **kwargs):
        super(YOLOv1ResNet, self).__init__()
        self.S = kwargs['S']
        self.B = kwargs['B']
        self.C = kwargs['C']

        self.backbone = nn.Sequential(*list(resnet50().children())[:-1])
        self.fc = nn.Linear(2048, self.S*self.S*(self.B*5+self.C))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.backbone(x)
        res = self.fc(torch.flatten(res, start_dim=1))
        # res = self.sigmoid(res)
        return res

