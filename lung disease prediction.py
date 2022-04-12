# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:48:05 2021

@author: Hp
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('lung disease prediction model - vgg16.h5')

dir = input('Please enter the specific directory with filename: ')
#img = image.load_img("C:/Users/Hp/Data Science/phenomonia prediction/chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg",target_size=(224,224))
img = image.load_img(dir,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
image_data = preprocess_input(x)
classes = model.predict(image_data)  


