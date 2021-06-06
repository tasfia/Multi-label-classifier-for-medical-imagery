# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:59:46 2021

@author: tshermin
"""
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

resnet_dict = {"resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}


class BaseFeatExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatExtractor, self).__init__()
    def output_num(self):
        pass
class ResNetFeat(BaseFeatExtractor):
    def __init__(self, model_name='resnet101',model_path=None, normalize=True):
        super(ResNetFeat, self).__init__()
        self.model_resnet = resnet_dict[model_name](pretrained=False)
        if not os.path.exists(model_path):
            model_path = None
            print('invalid model path!')
        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def getMean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def getStd(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.getMean()) / self.getStd()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.size())
        x = self.avgpool(x)
        #print(x.size())        
        x = x.view(x.size(0), -1)
        #print(x.size())
        return x

    def output_num(self):
        return self.__in_features
    
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, HiddenLayerDim=256):
        super(Classifier, self).__init__()
        self.bottleneck = nn.Linear(input_dim, HiddenLayerDim)
        self.fc = nn.Linear(HiddenLayerDim, output_dim)
        self.main = nn.Sequential(
        self.bottleneck,
        nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        self.fc
        ))

    def forward(self, x, batchNorm = False):
        out = [x]
        if batchNorm == True:
            for module in self.main.children():
                x = module(x)
                out.append(x)
        else:
            for module in self.main.children():
                x = module(x)
                out.append(x)
        return out
    