#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x

class Model(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10'):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class BarlowTwins(torch.nn.Module):
    def __init__(self, feature_dim = 128):
        super().__init__()
        
        self.l1_conv = torch.nn.Conv2d(3, 64, 3)
        self.l1_relu = torch.nn.ReLU()

        self.l2_conv = torch.nn.Conv2d(64, 128, 3)
        self.l2_relu = torch.nn.ReLU()
        self.l2_pool = torch.nn.MaxPool2d(2, 2)

        self.l3_conv = torch.nn.Conv2d(128, 128, 3)
        self.l3_relu = torch.nn.ReLU()
        
        self.l4_conv = torch.nn.Conv2d(128, 128, 3)
        self.l4_relu = torch.nn.ReLU()
        
        self.l5_conv = torch.nn.Conv2d(128, 256, 3)
        self.l5_relu = torch.nn.ReLU()
        self.l5_pool = torch.nn.MaxPool2d(2, 2)

        self.l6_conv = torch.nn.Conv2d(256, 512, 3)
        self.l6_relu = torch.nn.ReLU()
        self.l6_pool = torch.nn.MaxPool2d(2, 2)
        
        self.l7_conv = torch.nn.Conv2d(512, 512, 3)
        self.l7_relu = torch.nn.ReLU()
        
        self.l8_conv = torch.nn.Conv2d(512, 512, 3)
        self.l8_relu = torch.nn.ReLU()
        self.l8_pool = torch.nn.MaxPool2d(4,4)        
        # self.l8_flatten = torch.nn.Flatten()

        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
        
        self.p1_linear = nn.Linear(512, 512, bias=True)
        self.p1_batchnorm = nn.BatchNorm1d(512)
        self.p1_relu = nn.ReLU(inplace=True)
        self.p2_linear = nn.Linear(512, feature_dim, bias=True)

    def encode(self, x):
        k =3
        p = (k-1)/2
        x = F.pad(x, (1, 1, 1, 1))
        x = self.l1_conv(x)
        x = self.l1_relu(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l2_conv(x)
        x = self.l2_relu(x)
        x = self.l2_pool(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l3_conv(x)
        x = self.l3_relu(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l4_conv(x)
        x = self.l4_relu(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l5_conv(x)
        x = self.l5_relu(x)
        x = self.l5_pool(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l6_conv(x)
        x = self.l6_relu(x)
        x = self.l6_pool(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l7_conv(x)
        x = self.l7_relu(x)

        x = F.pad(x, (1, 1, 1, 1))
        x = self.l8_conv(x)
        x = self.l8_relu(x)

        x = self.l8_pool(x)
        # x = self.l8_flatten(x)
        return x

    def projection(self, x):
        x = self.p1_linear(x)
        x = self.p1_batchnorm(x)
        x = self.p1_relu(x)
        x = self.p2_linear(x)
        return x
        # self.p1_linear = nn.Linear(512, 512, bias=False)
        # self.p1_batchnorm = nn.BatchNorm1d(512)
        # self.p1_relu = nn.ReLU(inplace=True)
        # self.p2_linear = nn.Linear(512, feature_dim, bias=True)

    def forward(self, x):
        x = self.encode(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        out = self.projection(feature)
        return F.normalize(out, dim=-1), F.normalize(out, dim=-1)
