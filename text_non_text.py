# %%
# from keras.l typing_extensions import Concatenate
from os import name
from keras import layers
from keras.applications import mobilenet
from numpy.lib.npyio import load
import tensorflow as tf
from keras import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.layers import Flatten, Dropout,Dense
from keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.utils import plot_model
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv
%matplotlib inline
# %%
class MSPNet(object):
    def __init__(self):
        self.image_size = (224,224, 3)
        self.lr = 0.01
        self.saved_model_name = 'model.h5'
        self.model = self.build_model()

    def build_model(self):
        mobilenet_model = MobileNet(weights='imagenet', input_shape=self.image_size, include_top=False, 
                                alpha=0.75)

        x = Flatten(name='flatten')(mobilenet_model.output)
        x = Dropout(rate=0.5)(x)
        x = Dense(1024, name='fc1')(x)
        output = Dense(2, activation='softmax', name='output')(x)

        model = Model(inputs=mobilenet_model.input, outputs=output)

        for layer in mobilenet_model.layers:
            layer.trainable = False

        model.summary()
        plot_model(model, to_file='network.png')
        
        opt = RMSprop(learning_rate=self.lr)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                      optimizer=opt, metrics=['accuracy'])

        return model

    def load_image(self):
        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=self.image_size[:-1])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)
        return x

    # def train(self, path_of_dataset_to_train=''):
    #     # https://github.com/fchollet/deep-learning-models/blob/master/mobilenet.py#L90
    #     img = self.load_image()

    #     # self.model.fit(img, ...)
    
    # def predict(self, img):
    #     img = self.load_image(img)

    #     return self.model.fit(img)

    def save_model(self):
        self.model.save(self.saved_model_name)
    
    def load_model(self):
        self.model = load_model(self.saved_model_name)

    



test = MSPNet()
# %%
x = np.arange(5).reshape(1, 5)

print(x)
# [[ 0  1  2  3  4]
#   [ 5  6  7  8  9]
#   [10 11 12 13 14]
#   [15 16 17 18 19]]
y = np.arange(5, 15).reshape(1, 10)
y

# z = np.arange(20, 30).reshape(3, 2, 5)
# w = np.arange(20, 30).reshape(3, 2, 5)
# print(y)
# # [[20 21 22 23 24]
# #  [25 26 27 28 29]]
x = tf.keras.layers.Concatenate()([x, y])
x
# tf.keras.layers.Flatten()(x)
