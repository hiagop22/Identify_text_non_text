# %%
import os
import glob
import random
import shutil
import numpy as np
import gdown
import tensorflow as tf
from keras import Model
from matplotlib.image import imread
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Flatten, Dropout,Dense
from keras.applications.mobilenet import MobileNet as MobNet
from keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.utils import plot_model -> Necessary only if you desire to see the structure of NN
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from shutil import copyfile
# %%
class PrepareEnvironment(object):
    def __init__(self):
        self.population_dataset = 'data/'
        self.sampled_dataset = 'dataset/'
        self.subdirs = ['training_set/', 'validation_set/', 'testing_set/']
        self.amount_of_sample = {self.subdirs[0]: 1000, 
                               self.subdirs[1]: 500,
                               self.subdirs[2]: 10}
        self.batch_size = 64
        self.labeldirs = ['nontext', 'text']

    def create_diretories(self):
        shutil.rmtree(self.sampled_dataset)

        for subdir in self.subdirs:
            if subdir != self.subdirs[2]:
                for labeldir in self.labeldirs:
                    newdir = self.sampled_dataset + subdir + labeldir
                    os.makedirs(newdir, exist_ok=True)
            else:
                newdir = self.sampled_dataset + subdir + 'test'
                os.makedirs(newdir, exist_ok=True)

    def sample_data_to_diretories(self):

        for subdir in self.subdirs:
            for label in self.labeldirs:
                if subdir != self.subdirs[2]:
                    destionation_dir = self.sampled_dataset + subdir + label
                else:
                    destionation_dir = self.sampled_dataset + subdir + 'test'

                for file in random.sample(glob.glob(self.population_dataset + label + '/*.jpg'), 
                                        self.amount_of_sample[subdir]):
                    shutil.copy(file, destionation_dir)

    def initial_setup(self):
        self.create_diretories()
        self.sample_data_to_diretories()

class MobileNet(PrepareEnvironment):
    def __init__(self):
        super().__init__()
        self.image_size = (224,224, 3)
        self.lr = 0.01
        self.saved_model_name = 'model.h5'
        self.modelCheckPoint_name = 'best_model.h5'
        self.model = self.build_model()

        self.url_of_model_stored_in_drive = 'https://drive.google.com/uc?id=1UPMwzSu1C-bugrZP1BcvJQxtS-MKToOa'

    def build_model(self):
        mobilenet_model = MobNet(weights='imagenet', input_shape=self.image_size, include_top=False, 
                                alpha=0.75)

        x = Flatten(name='flatten')(mobilenet_model.output)
        x = Dropout(rate=0.2)(x)
        x = Dense(1024, name='fc1')(x)
        x = Dropout(rate=0.2)(x)
        output = Dense(2, activation='softmax', name='output')(x)

        model = Model(inputs=mobilenet_model.input, outputs=output)

        for layer in mobilenet_model.layers:
            layer.trainable = False

        # model.summary()
        # plot_model(model, to_file='network.png')
        
        opt = RMSprop(learning_rate=self.lr)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy, 
                      optimizer=opt, metrics=['accuracy'])

        return model

    def download_model_from_drive(self):
        
        gdown.download(self.url_of_model_stored_in_drive, self.saved_model_name, quiet=False)

    def load_just_one_image(self, img_path):
        try:
            img = image.load_img(img_path, target_size=self.image_size[:-1])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)
            
            return x
        except Exception as e:
            print(f'It was not possible to load \'{img}\'. The error \'{e}\'was raised.')
    
    def prepare_data_and_perform_data_augmentation(self):
        self.train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.5, rotation_range=90, 
                                                     zoom_range=0.2, horizontal_flip=True, height_shift_range=0.5,
                                                     width_shift_range=0.5)
        self.validation_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_set = self.train_datagen.flow_from_directory(self.sampled_dataset+self.subdirs[0], classes=self.labeldirs, 
                                                    batch_size=self.batch_size, target_size=self.image_size[:-1], class_mode='categorical')
        self.validation_set = self.validation_datagen.flow_from_directory(self.sampled_dataset+self.subdirs[1], classes=self.labeldirs, 
                                                    batch_size=self.batch_size, target_size=self.image_size[:-1], class_mode='categorical')
        self.test_set = self.test_datagen.flow_from_directory(self.sampled_dataset+self.subdirs[2], batch_size=1,
                                                    target_size=self.image_size[:-1], class_mode=None, shuffle=False)

    def train(self):
        checkpoint = ModelCheckpoint(filepath=self.modelCheckPoint_name, monitor='val_loss', save_best_only=True, mode='min', verbose=0)

        self.model.fit(self.train_set, steps_per_epoch=(self.amount_of_sample[self.subdirs[0]])//self.batch_size, epochs=20, 
                                validation_data=self.validation_set, validation_steps=(self.amount_of_sample[self.subdirs[1]])//self.batch_size, 
                                callbacks=[checkpoint])
        
        self.evaluate_model()

    def evaluate_model(self):
        _, score = self.model.evaluate(self.validation_set, steps=(self.amount_of_sample[self.subdirs[1]])//self.batch_size,
                                                    verbose=0)
        print('Accuracy: %.3f' % np.round(score*100,1))        
    
    def test_model(self):
        nontext_index = 0
        onehot_outputs = self.model.predict(self.test_set)

        for index, onehot_output in enumerate(onehot_outputs):            
            img_path = self.sampled_dataset + self.subdirs[2] + self.test_set.filenames[index]
            
            img = imread(img_path)
            plt.imshow(img)
            if np.argmax(onehot_output) == nontext_index:
                    plt.title('Nontext')
            else:
                plt.title('Text')
            plt.show()

    def predict(self, img):
        img = self.load_image(img)

        return self.model.fit(img)

    # def save_model(self):
    #     self.model.save(self.saved_model_name)
    def load_checkpoint_model(self):
        try:
            self.model = load_model(self.modelCheckPoint_name)
        except Exception as e:
            print(f'The error \'{e}\' was raised when tried to load {self.modelCheckPoint_name}.')
    
    def load_model(self, model_name=''):
        if model_name == '':
            model_name = self.saved_model_name
        try:
            if not os.path.exists(model_name):
                print('Downloading model from Google Drive.')
                self.download_model_from_drive()

            self.model = load_model(model_name)
            print(f'The \'{model_name}\' was loaded.')
        except Exception as e:
            print(f'Is was not possible to load \'{model_name}\'. \nThe exception\'{e}\' was raised.')
    

test = MobileNet()
test.initial_setup()
test.prepare_data_and_perform_data_augmentation()

# %%
test.load_model()
# %%
# test.train()

# %%
test.test_model()

# %%
test.evaluate_model()