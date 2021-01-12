"""Neural Network prediction

Neural Network inference
Current structure type: CNN

Author: Ian Flores
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

#Load the saved model
model_path = './saved_models'
model = load_model(model_path, compile = True)

#Insert samples
samples = []
use_samples = [12, 354, 2, 94, 123, 1003, 843, 253, 41, 2365]
prediction_path = 'C:/Users/Ian/EAR-UCI/dataset/'
prediction_x = np.load(prediction_path+'/test_x.npy', allow_pickle=True)
print(prediction_x.shape)
prediction_y = np.load(prediction_path+'/test_y.npy', allow_pickle=True)
for sample in use_samples:
    samples.append(prediction_x[0,sample])

samples = np.array(samples)
print(samples.shape)

#Generate predictions for samples
predictions = model.predict(samples)
print(predictions)