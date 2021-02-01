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

class Network:
    # default directories
    LOG_DIR = './LOGS/FIT/'
    
    def __init__(self) -> None:        
        # training dataset
        self.dataset = EARDataset()
        
        # Dataset variables
        self.num_classes = len(self.dataset.SAMPLE_CATEGORIES)
        
        #training data
        self.train_x = None
        self.train_y = None
        
        #validation data
        self.test_x = None
        self.test_y = None
        
        #one_hot labels
        self.train_y_one_hot = None
        self.test_y_one_hot = None
        
        #network model
        self.model = None
    
    def load_dataset(self, dataset_file_path:str) -> None:
        """Loads the EAR dataset, normalizes, reshapes, and converts labels to one-hot encoding

        """
        # load the dataset binares
        self.train_x, self.train_y, self.test_x, self.test_y = self.dataset.load(dataset_file_path)

        # normalize each spectrogram's values individually from 0.0 to 1.0
        print('Normalizing dataset samples')
        for i, sample in enumerate(self.train_x):
            self.train_x[i] = (sample - sample.min()) / (sample.max() - sample.min())
        for i, sample in enumerate(self.test_x):
            self.test_x[i] = (sample - sample.min()) / (sample.max() - sample.min())
        print('Dataset normalization complete')

        # print data shape
        print('Training data shape : ', self.train_x.shape, self.train_y.shape)
        print('Testing data shape : ', self.test_x.shape, self.test_y.shape)

        # print number of classifications
        print('Total number of outputs : ', len(self.dataset.SAMPLE_CATEGORIES))
        print('Output classes : ', self.dataset.SAMPLE_CATEGORIES)

        # test that data was properly loaded by displaying a couple samples
        plt.figure(figsize=[8, 4])

        # first spectrogram in training set
        plt.subplot(121)
        plt.imshow(self.train_x[0, :, :], cmap='plasma')
        plt.title("Training Set Ground Truth : {}".format(self.train_y[0]))
        # first spectrogram in testing set
        plt.subplot(122)
        plt.imshow(self.test_x[0, :, :], cmap='plasma')
        plt.title("Test Set Ground Truth : {}".format(self.test_y[0]))

        plt.show()

        # reshape data to flatten all demensions before the spectrograms
        self.train_x = self.train_x.reshape(-1, self.dataset.SAMPLE_SHAPE[0], self.dataset.SAMPLE_SHAPE[1], 1)
        self.test_x = self.test_x.reshape(-1, self.dataset.SAMPLE_SHAPE[0], self.dataset.SAMPLE_SHAPE[1], 1)
        self.train_y = self.train_y.reshape(-1)
        self.test_y = self.test_y.reshape(-1)
        # print reshaped
        print('Training Set Reshape : {} {}'.format(self.train_x.shape, self.train_y.shape))
        print('Test Set Reshape : {} {}'.format(self.test_x.shape, self.test_y.shape))

        # format data type
        self.train_x, self.test_x = map(lambda x: x.astype('float32'), [self.train_x, self.test_x])

        #convert labels to one-hot encoding
        self.train_y_one_hot = utils.to_categorical(self.train_y)
        self.test_y_one_hot = utils.to_categorical(self.test_y)

        # show the converted labels vs original
        print('Original label: ', self.train_y[0])
        print('Converted label: ', self.train_y_one_hot[0])

    def create_network(self, kernel_size=5, conv_layers=2, conv_size=[24, 48], pool_size=(4, 2), strides=(4, 2), dense_size=64) -> None:
        """Create the sequential network model using Keras API
        
        """
        # build the sequential network
        self.model = keras.Sequential()

        # add convolutional layers
        self.model.add(layers.Conv2D(conv_size[0], kernel_size=(kernel_size, kernel_size), activation='relu', padding='same', input_shape=(self.dataset.SAMPLE_SHAPE[0], self.dataset.SAMPLE_SHAPE[1], 1)))
        
        for i in range(conv_layers - 1):
            self.model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding ='same'))
            self.model.add(layers.Conv2D(conv_size[i+1], (kernel_size, kernel_size), activation='relu', padding='same'))
        
        self.model.add(layers.Flatten())

        # add dense layers
        self.model.add(layers.Dense(dense_size, activation='relu'))
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))

        # display model summary
        self.model.summary()

    def train_model(self, batch_size=160, epochs=20) -> None:
        """Optimizes the model, sets up TensorBoard output, and trains the model

        """
        # optimize model using cross entropy and the 'Adam" optimizer
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        # setup tensorboard
        log_dir = self.LOG_DIR + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # train model
        self.model.fit( self.train_x, 
                        self.train_y_one_hot, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=1,
                        validation_data=(self.test_x, self.test_y_one_hot),
                        callbacks=[tensorboard_callback])

    def save_model(self, save_path='./saved_models'):
        """Save the model.

        Args:
            save_path (str): saved model filepath. Defaults to './saved_models'.
        """
        print('Saving model')
        models.save_model(self.model, save_path)
        print('Model saved')


if __name__ == "__main__":
    #training_data_path = 'C:/Users/Ian/EAR-UCI-Dataset/Spectrograms/train'
    #validation_data_path = 'C:/Users/Ian/EAR-UCI-Dataset/Spectrograms/validation'
    
    dataset_file_path = './dataset/'

    network = Network()
    network.load_dataset(dataset_file_path)
    network.create_network()
    network.train_model()
    network.save_model()