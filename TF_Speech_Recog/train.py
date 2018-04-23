import IPython.display as ipd
import librosa
import librosa.display
from scipy.io import wavfile
import soundfile as sf
import os
from os.path import basename
import cv2
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import IPython.display as ipd
import gc

import pickle
import tempfile
import time
import math
import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.utils.data_utils import Sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from utils import *

seed = 0
np.random.seed(seed)
import random
random.seed(seed)

paths = pickle.load(open('./Data/train_val_paths.pickle', 'rb'))

bg_files = ['pink_noise.wav', 'dude_miaowing.wav', 'exercise_bike.wav',
                    'doing_the_dishes.wav', 'white_noise.wav', 'running_tap.wav']
bg_wavs = {file_ : sf.read('./Data/train/audio/_background_noise_/'+file_)[0] for file_ in bg_files}


class train_generator(Sequence):

    def __init__(self):
        self.ids_train_split = paths[0]
        self.batch_size = batch_size
        self.audio_length = 45
        self.n_mels = 128
        self.label_list = 'yes no up down left right on off stop go'.split()+['unknown', 'silence']

    def __len__(self):
        return len(self.ids_train_split) // self.batch_size

    def __getitem__(self, idx):
        ids_train_batch = self.ids_train_split[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x_batch = []
        y_batch = []
        for file_ in ids_train_batch:
            label = file_[file_[:file_.rfind('/')].rfind('/')+1:file_.rfind('/')]
            if label not in self.label_list[:-2]:
                unknown_label = label
                label = 'unknown'
            if label=='_background_noise_':
                label = 'silence'
            speaker = file_[file_.rfind('/')+1:file_.find('_')]
            
            if label == 'silence':
                start_ = np.random.choice((np.arange(16000*60-self.audio_length*370)))
                X, sr = sf.read(file_, start=start_, stop=start_+self.audio_length*370)
            else:
                X, sr = sf.read(file_)
                X, _ = librosa.effects.trim(X)
            if label == 'unknown':
                X = randomMix(X, sr, speaker, unknown_label, u=0.3)
            if label != 'silence':
                X = randomNoise(X, sr, bg_wavs, u=0.7)
                X = randomSpeed(X, sr, u=0.7)                
            
            file_feature = librosa.feature.melspectrogram(X, sr, n_mels=self.n_mels, hop_length=370)
#             file_feature2 = librosa.feature.mfcc(X, sr, n_mfcc=self.n_mels, hop_length=370)
            file_feature = cv2.resize(librosa.power_to_db(file_feature, ref=np.max), (self.audio_length, self.n_mels))
#             file_feature2 = cv2.resize(file_feature2, (self.audio_length, self.n_mels))
#             file_feature = np.stack((file_feature, file_feature2), axis=2)
            
            label_np = np.zeros((12))
            label_np[self.label_list.index(label)] = 1.0
            x_batch.append(file_feature.reshape((self.audio_length, self.n_mels, 1)))
            y_batch.append(label_np)
            
        return np.array(x_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        pass


class valid_generator(Sequence):

    def __init__(self):
        self.ids_train_split = paths[1]
        self.batch_size = batch_size
        self.audio_length = 45
        self.n_mels = 128
        self.label_list = 'yes no up down left right on off stop go'.split()+['unknown', 'silence']

    def __len__(self):
        return len(self.ids_train_split) // self.batch_size

    def __getitem__(self, idx):
        ids_train_batch = self.ids_train_split[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x_batch = []
        y_batch = []
        for file_ in ids_train_batch:
            label = file_[file_[:file_.rfind('/')].rfind('/')+1:file_.rfind('/')]
            if label not in self.label_list[:-2]:
                unknown_label = label
                label = 'unknown'
            if label=='_background_noise_':
                label = 'silence'
            speaker = file_[file_.rfind('/')+1:file_.find('_')]
            
            if label == 'silence':
                start_ = np.random.choice((np.arange(16000*60-self.audio_length*370)))
                X, sr = sf.read(file_, start=start_, stop=start_+self.audio_length*370)
            else:
                X, sr = sf.read(file_)
                X, _ = librosa.effects.trim(X)
            
            file_feature = librosa.feature.melspectrogram(X, sr, n_mels=self.n_mels, hop_length=370)
#             file_feature2 = librosa.feature.mfcc(X, sr, n_mfcc=self.n_mels, hop_length=370)
            file_feature = cv2.resize(librosa.power_to_db(file_feature, ref=np.max), (self.audio_length, self.n_mels))
#             file_feature2 = cv2.resize(file_feature2, (self.audio_length, self.n_mels))
#             file_feature = np.stack((file_feature, file_feature2), axis=2)
            
            label_np = np.zeros((12))
            label_np[self.label_list.index(label)] = 1.0
            x_batch.append(file_feature.reshape((self.audio_length, self.n_mels, 1)))
            y_batch.append(label_np)
            
        return np.array(x_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        pass

#Create model - (Convolution layers --> Max Pooling layer) x3 --> Fully connected layer --> Dropout layer --> Readout layer
model=Sequential()

model.add(Conv2D(15, (3, 3),input_shape=(45, 128, 1), padding="same", kernel_initializer='glorot_uniform'))
model.add(Conv2D(15, (3, 3),padding="same", kernel_initializer='glorot_uniform'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(0.0))

model.add(Conv2D(45, (2, 2), padding="same", kernel_initializer='glorot_uniform'))
model.add(Conv2D(45, (2, 2), padding="same", kernel_initializer='glorot_uniform'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(0.0))

model.add(Conv2D(75, (2, 2), padding="same", kernel_initializer='glorot_uniform'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(0.0))

model.add(Flatten())
model.add(Dense(256,kernel_initializer='TruncatedNormal',activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(12,kernel_initializer='TruncatedNormal',activation='softmax'))


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-6),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-6),
             ModelCheckpoint(monitor='val_loss',
                             filepath='saved_model/keras/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='TensorBoard/keras')]

#Compile model
adam=Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

# Training parameters
epochs = 100
batch_size = 24

# Train the model
# model.load_weights('./CASIA_weights/best_weights.hdf5')
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(len(paths[0]) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks, max_queue_size=56000,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(paths[1])) / float(batch_size)),
                    workers=8, use_multiprocessing=True)

