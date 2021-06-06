# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:56:03 2021

@author: tshermin
"""

from __future__ import print_function
from easydict import EasyDict
import argparse
import torch
#from utils.utils import *
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from scipy.io import loadmat
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
import copy
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
from matplotlib.pyplot import imshow
from glob import iglob
import os, fnmatch
import csv
import random
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

###############################################################################
import ImageDataset
import BaseNet
###############################################################################
#random seed initialization for better reproducibility
random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(10)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 200
batch_size = 4
###############################################################################

# read the training csv file
train_csv = pd.read_csv('newdata2.csv')
# train dataset
train_data = ImageDataset.ImageDataset(
    train_csv, train=True, test=False
)
# validation dataset
valid_data = ImageDataset.ImageDataset(
    train_csv, train=False, test=False
)
# train data loader
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=batch_size,
    shuffle=False
)

labels = train_csv.columns.values[1:]
# prepare the test dataset and dataloader
test_data = ImageDataset.ImageDataset(
    train_csv, train=False, test=True
)
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
)
##############################################################################

Net = BaseNet.ResNetFeat(model_name='resnet50',model_path='resnet50-19c8e357.pth').cuda()
Classifier_final = BaseNet.Classifier(Net.output_num(), 5, HiddenLayerDim=256).cuda()
            
optimizer_Net = optim.SGD(Net.parameters(), lr=.001, weight_decay=.0005, momentum=0.9, nesterov=True)
optimizer_Classifier_final = optim.SGD(Classifier_final.parameters(), lr=.001, weight_decay=.0005, momentum=0.9, nesterov=True)

#computing positive weights for binary cross-entropy loss to handle class imbalance in the training dataset
def calculate_pos_weights(class_counts):
    pos_weights = np.ones_like(class_counts)
    neg_counts = [41-pos_count for pos_count in class_counts] #41=length of train set/ batch size
    cdx = 0
    for i in range(5):
        pos_weights[i] = neg_counts[i] / (class_counts[i] + 1e-5)
        cdx += 1

    return torch.as_tensor(pos_weights, dtype=torch.float)

# training function
def train(dataloader, train_data, device):
    print('Training')
    Net.train()
    Classifier_final.train()
    counter = 0
    train_running_loss = 0.0
    Round = 0
    oval = 0
    irregular = 0
    parallel = 0
    not_parallel = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
      
        b = dataloader.batch_size
        for j in range(b):
            if target[j,0]== 1:
                Round += 1
            if target[j,1]== 1:
                oval += 1
            if target[j,2]== 1:
                irregular += 1
            if target[j,3]== 1:
                parallel +=1
            if target[j,4]== 1:
                not_parallel += 1
                
        class_counts = [Round, oval, irregular, parallel, not_parallel] #column-wise positive class counts 
 
        
        pos_weight=calculate_pos_weights(class_counts) #positive weights for the batch
        #print(pos_weight)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device) #handles class imbalance using the positive weights
         
        optimizer_Classifier_final.zero_grad()
        optimizer_Net.zero_grad()
        
        features = Net.forward(data)
        _,_, predProb = Classifier_final.forward(features)
 
        loss = criterion(predProb, target)
        train_running_loss += loss.item()

        loss.backward()

        optimizer_Classifier_final.step()
        optimizer_Net.step()
        
    train_loss = train_running_loss / counter
    return train_loss

def validate(dataloader, criterion, val_data, device):
    print('Validating')
    Net.eval()
    Classifier_final.eval()
    counter = 0
    val_running_loss = 0.0
    
    with torch.no_grad():
        
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            
            fs1 = Net.forward(data)
            _,s_logit, predProb_s = Classifier_final.forward(fs1)
            
            loss = criterion(predProb_s, target)
            val_running_loss += loss.item()
        
        val_loss = val_running_loss / counter
        return val_loss
    
def test():
    Net.eval()
    Classifier_final.eval()
    
    hacc = 0 #hacc denotes hamming score based accuracy 
    precisi = 0
    recall = 0
    F1 = 0
    
    y_true = np.empty((len(test_loader), 5))
    y_pred = np.empty((len(test_loader), 5))
    
    i = 0
    for counter, data in enumerate(test_loader):
        temp2 = np.zeros(5)
        image, target = data['image'].to(device), data['label']
        y_true[i] = target.numpy()

        features = Net.forward(image)
        _,_, predProb = Classifier_final.forward(features)
        outputs = predProb.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
    
        best = sorted_indices[-2:]
   
        temp2[best[0]] = 1
        temp2[best[1]] = 1
        y_pred[i] = temp2
        
        hacc += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        
        if sum(y_pred[i]) != 0:
            precisi += sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
            
        if sum(y_true[i]) != 0:
            recall += sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
            
        if (sum(y_true[i]) != 0) and (sum(y_pred[i]) != 0):
            F1 += (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
        
        i += 1

    
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred) #confusion matrix for multi-label classifier
    
    Hamming_score_acc = float(hacc / len(test_loader))*100
    
    precision = float(precisi / len(test_loader))
    recall1 = float(recall / len(test_loader))
    F11 = float(F1 / len(test_loader))
    
    return Hamming_score_acc, precision, recall1, F11
    
#test()
###############################################################################

# start the training and validation
train_loss = []
valid_loss = []
best_acc = 0.0 
best_f1 = 0.0
best_pre = 0.0
best_re = 0.0

criterion = nn.BCEWithLogitsLoss().to(device)
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    
    train_epoch_loss = train(
         train_loader, train_data, device
    )
    valid_epoch_loss = validate(
       valid_loader, criterion, valid_data, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print('Train Loss: {:.4f}'.format(train_epoch_loss))
    #print('Val Loss: {:.4f}'.format(valid_epoch_loss))
    
    Hamming_score_acc, precision, recall, F1 = test()
    
    print('\nAverage Hamming score accuracy: {:.4f}'.format(Hamming_score_acc))
    print('\nAverage precision: {:.4f}'.format(precision))
    print('\nAverage recall: {:.4f}'.format(recall))
    print('\nAverage F1 score: {:.4f}'.format(F1))
    
    if Hamming_score_acc > best_acc and F1 > best_f1 and precision > best_pre and recall > best_re:
        torch.save(Classifier_final.state_dict(),"Classifier_final.pt")
        torch.save(Net.state_dict(),"Net.pt")
        best_acc = Hamming_score_acc
        best_f1 = F1
        best_pre = precision
        best_re = recall
        
print('Best Hamming score: {:.4f}'.format(
        best_acc))
print('Best Average precision: {:.4f}'.format(
        best_pre))
print('Best Average recall: {:.4f}'.format(
        best_re))
print('Best Average F1 score: {:.4f}'.format(
        best_f1))
###############################################################################
# test dataset and dataloader

labels = train_csv.columns.values[1:]

#Classifier_final.load_state_dict(torch.load('Classifier_final.pt'))
#Net.load_state_dict(torch.load('Net.pt'))

def test_label():
    Net.eval()
    Classifier_final.eval()
    
    for counter, data in enumerate(test_loader):
        image, target = data['image'].to(device), data['label']
        features = Net.forward(image)
        _,_, predProb = Classifier_final.forward(features)
        outputs = predProb.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
        best = sorted_indices[-2:]
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
          
        string_predicted = ''
        string_actual = ''
        print('predicted')
        for i in range(len(best)):
            string_predicted += f"{labels[best[i]]}    "
            
        print(string_predicted)
        print('actual')
        for i in range(len(target_indices)):
            string_actual += f"{labels[target_indices[i]]}    "
            
        print(string_actual)
        print("------------------------------------------------------")
        
    
test_label()
