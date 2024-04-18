import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


'''
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale_factor': self.scale_factor})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(scale_factor=config.get('scale_factor', 1.0), **config)


tf.keras.utils.get_custom_objects()['CustomScaleLayer'] = CustomScaleLayer
import tensorflow as tf
import tensorflow_hub as hub

# Define the custom_objects dictionary
custom_objects = {'KerasLayer': hub.KerasLayer}
'''

# Load the model without the custom_objects argument
model = tf.keras.models.load_model(r"F:\ml\deepfake_detection_modelupdated.h5")

'''
# Rebuild the model with the custom_objects argument
model = tf.keras.models.load_model(r"F:\ml\deepfake_detection_model.h5", custom_objects=custom_objects)

import tensorflow_hub as hub
model = tf.keras.models.load_model(r"F:\ml\deepfake_detection_model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
'''


pr_data = []
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(r"F:\deepfakeds\realds\1.mp4")
frameRate = cap.get(5)
dic={1:0,0:0}
while cap.isOpened():
    frameId = cap.get(1)
    ret, frame = cap.read()
    if ret != True:
        break
    if frameId % ((int(frameRate)+1)*1) == 0:
        face_rects, scores, idx = detector.run(frame, 0)
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            crop_img = frame[y1:y2, x1:x2]
            data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
            data = data.reshape(-1, 128, 128, 3)
            #print(model.predict_classes(data))
            predict_x=model.predict(data) 
            classes_x=np.argmax(predict_x,axis=1)
            if classes_x[0] in dic.keys():
                dic[classes_x[0]]+=1
            
            print(classes_x[0],predict_x)

if dic[1]>dic[0]:
    print('REAL')
else:
    print("Fake")
    
    
    
    
    
    

