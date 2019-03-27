#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:04:40 2019

@author: luodi
"""
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def nmp2img(data):
    '''
    transform the data to image
    '''
    im = data.reshape(28,28)
    img = Image.fromarray(im)
    return img

pig = np.load('/Users/luodi/Desktop/271B/PROJECT/google_image/diamond.npy')

for i in range(101):
    data = pig[i+131286,:]
    img = nmp2img(data)
    plt.imshow(img)
    plt.savefig('/Users/luodi/Desktop/271B/PROJECT/google_image/test_pic/diamond/%d.png'%(i))
 