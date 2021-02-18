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
        
        # dataset variables
        self.num_classes = len(self.dataset.SAMPLE_CATEGORIES)
        
        # training data
        self.train_x = None
        self.train_y = None
        
        # validation data
        self.test_x = None
        self.test_y = None
        
        # one_hot labels
        self.train_y_one_hot = None
        self.test_y_one_hot = None
        
        # network model
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

    def create_network(
        self,
        kernel_size=5,
        conv_layers=[24, 48],
        pool_size=(4, 2),
        strides=(4, 2),
        dense_size=64,
        dropout_rate=None
    ) -> None:
        """Create the sequential network model using Keras API
        
        """
        # network parameters
        self.kernel_size = kernel_size
        self.conv_layers = conv_layers
        self.pool_size = pool_size
        self.strides = strides
        self.dense_size = dense_size

        # build the sequential network
        self.model = keras.Sequential()

        # add convolutional layers
        self.model.add(layers.Conv2D(conv_layers[0], kernel_size=(kernel_size, kernel_size), activation='relu', padding='same', input_shape=(self.dataset.SAMPLE_SHAPE[0], self.dataset.SAMPLE_SHAPE[1], 1)))
        self.model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding ='same'))
        if dropout_rate is not None:
            self.model.add(layers.Dropout(dropout_rate[0]))

        for i in range(1, len(conv_layers)):
            self.model.add(layers.Conv2D(conv_layers[i], (kernel_size, kernel_size), activation='relu', padding='same'))
            self.model.add(layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding ='same'))
            if dropout_rate is not None:
                self.model.add(layers.Dropout(dropout_rate[i]))
        
        self.model.add(layers.Flatten())

        # add dense layers
        self.model.add(layers.Dense(dense_size, activation='relu'))
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))

        # display model summary
        self.model.summary()

    def train_model(self, batch_size=160, epochs=1, log_name='') -> None:
        """Optimizes the model, sets up TensorBoard output, and trains the model

        """
        # training parameters
        self.batch_size = batch_size
        self.epochs = epochs

        # optimize model using cross entropy and the 'Adam" optimizer
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        # setup tensorboard
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if log_name != '':
            log_name += '_'
        log_dir = (
            self.LOG_DIR +
            log_name +
            timestamp            
        )
        self.model_name = log_name + timestamp
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # train model
        self.model.fit( self.train_x, 
                        self.train_y_one_hot, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        verbose=1,
                        validation_data=(self.test_x, self.test_y_one_hot),
                        callbacks=[tensorboard_callback])

    def save_model(self, save_path='./saved_models/', model_name=''):
        """Save the model.

        Args:
            save_path (str): saved model filepath. Defaults to './saved_models'.
        """
        print('Saving model')
        models.save_model(self.model, save_path + model_name)
        print('Model saved')

    def representative_data_gen(self):
        """Generator for test data used to adjust dynamic range of the quantizer.

        Yields:
            np.ndarray: training set samples
        """
        # iterate over 100 samples in training set
        for sample in tf.data.Dataset.from_tensor_slices(self.train_x).batch(1).take(100):
            yield [sample]

    
    def convert_to_tflite(self, model_path: str) -> None:
        """Convert model to tensorflow lite model.

        Args:
            model_path (str): saved model directory path
        """
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()

        # Save the model.
        with open(model_path + 'saved_model.tflite', 'wb') as f:
            f.write(tflite_model)


if __name__ == "__main__":
    # training_data_path = 'C:/Users/Ian/EAR-UCI-Dataset/Spectrograms/train'
    # validation_data_path = 'C:/Users/Ian/EAR-UCI-Dataset/Spectrograms/validation'
    
    dataset_file_path = r'C:\UCI\Senior Year\Winter_2021\159_senior_design'

    network = Network()
    network.load_dataset(dataset_file_path)
    network.create_network(
        kernel_size=3,
        conv_layers=[24, 48, 48],
        dropout_rate=[0.2, 0.2, 0.2],
        pool_size=(4, 4),
        strides=(4, 4),
        dense_size=32
    )
    network.train_model(
        batch_size=40,
        log_name='3conv_drop_2_small_batch_44pool_32dense_3k'
    )
    network.save_model(model_name=network.model_name)
    network.convert_to_tflite('./saved_models/' + network.model_name + '/')