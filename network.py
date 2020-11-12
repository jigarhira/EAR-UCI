"""Neural Network structure

Neural Network for classifying audio samples
Current structure type: CNN

Author: Ian Flores, Jigar Hira
"""

import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image

from dataset import EARDataset


# directories
LOG_DIR = './logs/fit/'

# path to data samples
training_data_path = '/home/hiraj/projects/ear-uci-dataset/spectrograms/train'
validation_data_path = '/home/hiraj/projects/ear-uci-dataset/spectrograms/validation'

# load dataset
dataset = EARDataset()
dataset.load(training_data_path, validation_data_path)
train_x, train_y, test_x, test_y = dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y

# normalize each spectrogram's values individually from 0.0 to 1.0
train_x, test_x = map(lambda x: (x[:, :] - x[:, :].min()) / (x[:, :].max() - x[:, :].min()), [train_x, test_x])

# print data shape
print('Training data shape : ', train_x.shape, train_y.shape)
print('Testing data shape : ', test_x.shape, test_y.shape)

# print number of classifications
print('Total number of outputs : ', len(dataset.SAMPLE_CATEGORIES))
print('Output classes : ', dataset.SAMPLE_CATEGORIES)

# test that data was properly loaded by displaying a couple samples
plt.figure(figsize=[8, 4])

# first spectrogram in training set
plt.subplot(121)
plt.imshow(train_x[0, 0, :, :], cmap='gray')
plt.title("Training Set Ground Truth : {}".format(train_y[0, 0]))
# first spectrogram in testing set
plt.subplot(122)
plt.imshow(test_x[0, 0, :, :], cmap='gray')
plt.title("Test Set Ground Truth : {}".format(test_y[0, 0]))

plt.show()

# reshape data to flatten all demensions before the spectrograms
train_x = train_x.reshape(-1, dataset.SAMPLE_SHAPE[0], dataset.SAMPLE_SHAPE[1], 1)
test_x = test_x.reshape(-1, dataset.SAMPLE_SHAPE[0], dataset.SAMPLE_SHAPE[1], 1)
train_y = train_y.reshape(-1)
test_y = test_y.reshape(-1)
# print reshaped
print('Training Set Reshape : {} {}'.format(train_x.shape, train_y.shape))
print('Test Set Reshape : {} {}'.format(test_x.shape, test_y.shape))

# format data type
train_x, test_x = map(lambda x: x.astype('float32'), [train_x, test_x])

# convert labels to one-hot encoding
train_y_one_hot = utils.to_categorical(train_y)
test_y_one_hot = utils.to_categorical(test_y)

# show the converted labels vs original
print('Original label: ', train_y[0])
print('Converted label: ', train_y_one_hot[0])

# Neural Network Structure
BATCH_SIZE = dataset.SAMPLES_PER_FOLD
EPOCHS = 20
NUM_CLASSES = len(dataset.SAMPLE_CATEGORIES)
KERNEL_SIZE = 5

# build the sequential network
##NEED TO LOOK INTO DROPOUT (HELPS PREVENT OVERFITTING MODEL)##
model = keras.Sequential()
model.add(layers.Conv2D(24, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same', input_shape=(dataset.SAMPLE_SHAPE[0], dataset.SAMPLE_SHAPE[1], 1)))
model.add(layers.MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding ='same'))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(48, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding ='same'))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(48, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# display model summary
model.summary()

# optimize model using cross entropy and the 'Adam" optimizer
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# setup tensorboard
log_dir = LOG_DIR + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train model
train = model.fit(train_x, 
                  train_y_one_hot, 
                  batch_size=BATCH_SIZE, 
                  epochs=EPOCHS,
                  verbose=1,
                  validation_data=(test_x, test_y_one_hot),
                  callbacks=[tensorboard_callback])

