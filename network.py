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
    #default network parameters
    EPOCHS = 20
    KERNEL_SIZE = 5

    #default directories
    LOG_DIR = './LOGS/FIT/'
    
    def __init__(self) -> None:        
        #training dataset
        self.dataset = EARDataset()
        
        #Dataset variables
        self.BATCH_SIZE = None
        self.NUM_CLASSES = None
        
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
    
    def load_dataset(self, dataset_file_path) -> None:
        """Loads the EAR dataset, normalizes, reshapes, and converts labels to one-hot encoding

        """

        print('Loading dataset files')
        self.train_x = np.load(dataset_file_path + 'train_x.npy', allow_pickle=True)
        self.train_y = np.load(dataset_file_path + 'train_y.npy', allow_pickle=True)
        self.test_x = np.load(dataset_file_path + 'test_x.npy', allow_pickle=True)
        self.test_y = np.load(dataset_file_path + 'test_y.npy', allow_pickle=True)
        print('Dataset loading complete')

        # normalize each spectrogram's values individually from 0.0 to 1.0
        self.train_x, self.test_x = map(lambda x: (x[:, :] - x[:, :].min()) / (x[:, :].max() - x[:, :].min()), [self.train_x, self.test_x])

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
        plt.imshow(self.train_x[0, 0, :, :], cmap='gray')
        plt.title("Training Set Ground Truth : {}".format(self.train_y[0, 0]))
        # first spectrogram in testing set
        plt.subplot(122)
        plt.imshow(self.test_x[0, 0, :, :], cmap='gray')
        plt.title("Test Set Ground Truth : {}".format(self.test_y[0, 0]))

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

    def create_network(self) -> None:
        """Create the sequential network model using Keras API
        
        """
        # Neural Network Structure
        #self.BATCH_SIZE = self.dataset.SAMPLES_PER_FOLD
        self.BATCH_SIZE = 240
        self.NUM_CLASSES = len(self.dataset.SAMPLE_CATEGORIES)

        # build the sequential network
        self.model = keras.Sequential()
        self.model.add(layers.Conv2D(24, kernel_size=(self.KERNEL_SIZE, self.KERNEL_SIZE), activation='relu', padding='same', input_shape=(self.dataset.SAMPLE_SHAPE[0], self.dataset.SAMPLE_SHAPE[1], 1)))
        self.model.add(layers.MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding ='same'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Conv2D(48, (self.KERNEL_SIZE, self.KERNEL_SIZE), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(4, 2), strides=(4, 2), padding ='same'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Conv2D(48, (self.KERNEL_SIZE, self.KERNEL_SIZE), activation='relu', padding='same'))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))

        # display model summary
        self.model.summary()

    def train_model(self) -> None:
        """Optimizes the model, sets up TensorBoard output, and trains the model

        """
        # optimize model using cross entropy and the 'Adam" optimizer
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        # setup tensorboard
        log_dir = self.LOG_DIR + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # train model
        train = self.model.fit(self.train_x, 
                               self.train_y_one_hot, 
                               batch_size=self.BATCH_SIZE, 
                               epochs=self.EPOCHS,
                               verbose=1,
                               validation_data=(self.test_x, self.test_y_one_hot),
                               callbacks=[tensorboard_callback])

if __name__ == "__main__":
    training_data_path = '/home/ihflores/EAR-UCI-Dataset/Spectrograms/train'
    validation_data_path = '/home/ihflores/EAR-UCI-Dataset/Spectrograms/validation'
    dataset_file_path = './dataset/'

    #dataset = EARDataset()
    #dataset.load(training_data_path, validation_data_path, dataset_file_path)

    network = Network()
    network.load_dataset(dataset_file_path)
    network.create_network()
    network.train_model()
