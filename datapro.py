#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:47:42 2019

@author: luodi
"""
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os
from PIL import Image
import csv
from csv import *

train = os.listdir('traindata/')
train.remove('.DS_Store')
train.sort()

train_image = []
for i in range(len(train)):
    img = Image.open(('traindata/%s'%train[i]))
    Img = img.convert('L')

    threshold = 200
     
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    photo = Img.point(table, '1')
    image = (np.asarray(photo)+0).reshape(1,10000)
    train_image.append(image)

train_img = np.zeros((25,10000))
for i in range(25):
    train_img[i,:] = np.array(train_image[i])
    
label = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
label = label.reshape((25,1))

train_label=np.ones((25,5))*(-1)
for i in range(len(label)):
    n=label[i]
    train_label[i,n]=1
   
tu=train_img[18]
tu=tu.reshape((100,100))
plt.imshow(tu)

with open('train_img.csv','w',newline='') as f:
    f_csv = csv.writer(f)
    for i in range(25):
        f_csv.writerow(train_img[i,:])
        
with open('train_label.csv','w',newline='') as f:
    f_csv = csv.writer(f)
    for i in range(25):
        f_csv.writerow(label[i,:])
        
with open('label_onehot.csv','w',newline='') as f:
    f_csv = csv.writer(f)
    for i in range(25):
        f_csv.writerow(train_label[i,:])