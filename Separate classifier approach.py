# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:49:33 2021

@author: tshermin
"""

from __future__ import print_function
from easydict import EasyDict
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from scipy.io import loadmat
import torchvision.transforms as transforms
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import time
import copy
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
from glob import iglob
import os, fnmatch
import csv
import pandas as pd

###################################################################################################
#custom dataset
class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.all_image_names = self.csv[:]['ID']
        self.labels_shape = np.array(self.csv.drop(['ID','orientation'], axis=1))
        self.labels_orientation = np.array(self.csv.drop(['ID','shape'], axis=1))
        self.train_ratio = int(0.70 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio
        
        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels_shape = list(self.labels_shape[:self.train_ratio])
            self.labels_orientation = list(self.labels_orientation[:self.train_ratio])
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        # set the test data images and labels
        elif self.test == True and self.train == False:
            print(f"Number of testing images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:])
            self.labels_shape = list(self.labels_shape[-self.valid_ratio:])
            self.labels_orientation = list(self.labels_orientation[-self.valid_ratio:])
             # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = cv2.imread(f"tech_task_data/{self.image_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        targets_shape = self.labels_shape[index]
        targets_orientation = self.labels_orientation[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label_s': targets_shape,
            'label_o': targets_orientation
        }
####################################################################################################
# read the training csv file

train_csv = pd.read_csv('newdata3.csv')

batch_size = 8
# train dataset
train_data = ImageDataset(
    train_csv, train=True, test=False
)
# test dataset
test_data = ImageDataset(
    train_csv, train=False, test=True
)
# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)
# test data loader
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)
###########################################################################################################
#Network
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
        self.model_resnet = resnet_dict[model_name](pretrained=True)
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
        #nn.BatchNorm1d(HiddenLayerDim),
        nn.LeakyReLU(0.2, inplace=True),
        self.fc
        ),
        nn.Softmax(dim=-1)
        )

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
    
########################################################################################################################
        #test method
def test():
    Net.eval()
    C_shape.eval()
    C_orientation.eval()
    running_corrects1 = 0
    running_corrects2 = 0

    for counter, data in enumerate(test_loader):
        image, target_shape, target_orientation = data['image'].to(device), data['label_s'].to(device), data['label_o'].to(device)
        features = Net.forward(image)
        _,_,_, predProb_s = C_shape.forward(features)
        _,_,_, predProb_o = C_orientation.forward(features)
        predo = predProb_o.data.max(1)[1]
        preds = predProb_s.data.max(1)[1]
        running_corrects1 += torch.sum(preds == target_shape.data)
        running_corrects2 += torch.sum(predo == target_orientation.data)
  
  
    ACC_shape = float(running_corrects1.cpu().detach().numpy()/71.0)
    ACC_orientation = float(running_corrects2.cpu().detach().numpy()/71.0)
    return ACC_shape, ACC_orientation 
        
########################################################################################################################
#training the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Net = ResNetFeat(model_name='resnet50',model_path='resnet50-19c8e357.pth').to(device)
C_shape = Classifier(Net.output_num(), 3, HiddenLayerDim=256).to(device)
C_orientation = Classifier(Net.output_num(), 2, HiddenLayerDim=256).to(device)

optimizer_Net = optim.SGD(Net.parameters(), lr=.001, weight_decay=.0005, momentum=0.9, nesterov=True)
optimizer_C_shape = optim.SGD(C_shape.parameters(), lr=.001, weight_decay=.0005, momentum=0.9, nesterov=True)
optimizer_C_orientation = optim.SGD(C_orientation.parameters(), lr=.001, weight_decay=.0005, momentum=0.9, nesterov=True)


criterion = nn.CrossEntropyLoss()

num_epochs = 150


def train_model():
    since = time.time()

    best_model_wts_Net = copy.deepcopy(Net.state_dict())
    best_model_wts_C_shape = copy.deepcopy(C_shape.state_dict())
    best_model_wts_C_orientation = copy.deepcopy(C_orientation.state_dict())
    best_acc1 = 0.0
    best_acc2 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        Net.train()
        C_shape.train()
        C_orientation.train()
    
        running_loss1 = 0.0        
        running_loss2 = 0.0

        counter = 0

        for i, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
            counter += 1
            inputs, labels, labelo = data['image'].to(device), data['label_s'].to(device), data['label_o'].to(device)
        
            optimizer_C_shape.zero_grad()
            optimizer_Net.zero_grad()

                
            fs1 = Net.forward(inputs)
            _,_,s_logit, predProb_s = C_shape.forward(fs1)
                    
            loss1 = criterion(predProb_s, labels.squeeze(1))
            loss1.backward()
            
            optimizer_C_shape.step()
            optimizer_Net.step()

               
            optimizer_C_orientation.zero_grad()
            optimizer_Net.zero_grad()

               
            ft1 = Net.forward(inputs)
            _,_,o_logit, predProb_o = C_orientation.forward(ft1)

            predo = predProb_o.data.max(1)[1]
            loss2 = criterion(predProb_o, labelo.squeeze(1))
                    
            loss2.backward()
            optimizer_C_orientation.step()
            optimizer_Net.step()

            running_loss1 += loss1.item()
                
            running_loss2 += loss2.item() 
                

        epoch_loss = running_loss1 / counter

            
        epoch_loss2 = running_loss2 / counter

        acc_s, acc_o = test()
        
        print('\nLoss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, acc_s))
        print(' Loss2: {:.4f} Acc2: {:.4f}'.format(
                 epoch_loss2, acc_o))
            
        
        if acc_s > best_acc1:
            best_acc1 = acc_s
            best_model_wts = copy.deepcopy(Net.state_dict())
                
        if acc_o > best_acc2:
            best_acc2 = acc_o
            best_model_wts = copy.deepcopy(Net.state_dict())

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Test Acc shape: {:4f}'.format(best_acc1))
    print('Best Test Acc orientation: {:4f}'.format(best_acc2))

    # load best model weights
    Net.load_state_dict(best_model_wts_Net)
    C_shape.load_state_dict(best_model_wts_C_shape)
    C_orientation.load_state_dict(best_model_wts_C_orientation)
    
    return Net, C_shape, C_orientation
        

Base, C_shape, C_orientation = train_model()
                
       