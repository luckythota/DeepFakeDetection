import dlib
import cv2
import os
import re
import json
from pylab import *
from PIL import Image, ImageChops, ImageEnhance

a= r'F:\deepfakeds\realds'
b= r'F:\deepfakeds\fakeds'

train_frame_folder =os.listdir(a) 

train_frame_folder=os.listdir(b)

#preprocessing
list_of_train_data = [f for f in train_frame_folder if f.endswith('.mp4')]

detector = dlib.get_frontal_face_detector()

for vid in list_of_train_data:
    count = 0
    cap = cv2.VideoCapture(os.path.join(a, vid))#a
    #print(vid)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        #print(frameId)
        ret, frame = cap.read()
        #print(ret)
        
        if ret != True:
            break
        
        if frameId % ((int(frameRate)+1)*1) == 0:
            #print('746767')
            face_rects, scores, idx = detector.run(frame, 0)
            #print(face_rects)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                
               # print(x1,y2)
                crop_img = frame[y1:y2, x1:x2]
                if vid in os.listdir(b):
                    #print('hi')
                    cv2.imwrite(r"F:\deepfakeds\FAKEDS_/"+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                elif vid in os.listdir(a):
                    #print('hello')
                    cv2.imwrite(r"F:\deepfakeds\REALDS_/"+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                count+=1



import os
import cv2
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


input_shape = (128, 128, 3)
data_dir = r'F:\deepfakeds'


real_data = [f for f in os.listdir(data_dir+'/REALDS_') if f.endswith('.png')]
fake_data = [f for f in os.listdir(data_dir+'/FAKEDS_') if f.endswith('.png')]

X = []
Y = []

for img in real_data:
    X.append(img_to_array(load_img(data_dir+'/REALDS_/'+img)).flatten() / 255.0)
    Y.append(1)
for img in fake_data:
    X.append(img_to_array(load_img(data_dir+'/FAKEDS_/'+img)).flatten() / 255.0)
    Y.append(0)

Y_val_org = Y

#Normalization
X = np.array(X)
Y = to_categorical(Y, 2)

#Reshape
X = X.reshape(-1,128, 128, 3)

#Train-Test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
googleNet_model.trainable = True
model = Sequential()
model.add(googleNet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False),
              metrics=['accuracy'])
model.summary()



EPOCHS = 10
BATCH_SIZE = 100

tf.config.run_functions_eagerly(True)
history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_val, Y_val), verbose = 1)



model.save(r'F:\ml\deepfake_detection_model.h5')

           








