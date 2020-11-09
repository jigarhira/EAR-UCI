"""Neural Network structure

Neural Network for classifying audio samples
Current structure type: CNN

Author: Ian Flores
"""

import tensorflow as tf
import matplotlib as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image

from dataset import EARDataset

#Load the data
WIDTH = 128
HEIGHT = 130

# path to data samples
training_data_path = '/home/hiraj/projects/ear-uci-dataset/spectrograms/train'
validation_data_path = '/home/hiraj/projects/ear-uci-dataset/spectrograms/validation'

# load dataset
dataset = EARDataset()
dataset.load(training_data_path, validation_data_path)
train_x, train_y, test_x, test_y = dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y

# print data shape
print('Training data shape : ', train_x.shape, train_y.shape)
print('Testing data shape : ', test_x.shape, test_y.shape)

# normalize thei pixel values of spectrograms
train_data, test_data = train_data / 255.0, test_images / 255.0

#Find number of classifications and display
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

#Test that data was properly loaded
plt.figure(figsize=[5,5])

# Display first image in training set
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : ()".format(train_Y[0]))

# Display first image in testing set
plt.subplot(121)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : ()".format(test_Y[0]))

#Reshape data
train_X = train_X.reshape(-1, WIDTH, HEIGHT, 1)
test_X = test_X.reshape(-1, WIDTH, HEIGHT, 1)
train_X.shape, test_X.shape

#Format data type
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

#Convert labels to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

#Show the converted labels vs original
print('Original label: ', train_Y[0])
print('Converted label: ', train_Y_one_hot[0])

#Neural Network Structure
BATCH_SIZE = 16
EPOSCHS = 20
NUM_CLASSES = nClasses
KERNEL_SIZE = 5

#Build the sequential network
##NEED TO LOOK INTO DROPOUT (HELPS PREVENT OVERFITTING MODEL)##
model = keras.Sequential()
model.add(layers.Conv2D(24, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same', input_shape=(WIDTH,HEIGHT,1)))
model.add(layers.MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding ='same'))
#model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(48, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding ='same'))
#model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(48, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
#model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

#Display model summary
model.summary()

#Optimize model using cross entropy and the 'Adam" optimizer
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#Train model
train = model.fit(train_X, 
                  train_label, 
                  batch_size=BATCH_SIZE, 
                  epochs=EPOCHS,
                  verbose=1,
                  validation_data=(valid_X, valid_label),
                  callbacks=[tensorboard_callback])

#Model Evaluation
test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss : ', test_eval[0])
print('Test accuracy : ', test_eval[1])

#Make plots
acc = train.history['acc']
val_acc = train.history['val_acc']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()
