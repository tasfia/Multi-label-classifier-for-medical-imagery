# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:58:07 2021

@author: tshermin
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.all_image_names = self.csv[:]['ID']
        self.all_labels = np.array(self.csv.drop(['ID'], axis=1))
        self.train_ratio = int(0.70 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio
        
        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:-40])
            print(len(self.image_names))
            self.labels = list(self.all_labels[-self.valid_ratio:-40])
            print(len(self.labels))
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # set the test data images and labels
        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[-40:])
            self.labels = list(self.all_labels[-40:])
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
        #data directory
        image = cv2.imread(f"tech_task_data/{self.image_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        targets = self.labels[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }