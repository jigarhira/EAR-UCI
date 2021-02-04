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
samples_x = []
samples_y = []

#use_samples = [12, 354, 2, 94, 123, 1003, 843, 253, 41, 456]
total = 100
use_samples = [x for x in range(total)]

prediction_path = './dataset'
prediction_x = np.load(prediction_path+'/test_x.npy', allow_pickle=True)
print(prediction_x.shape)
prediction_y = np.load(prediction_path+'/test_y.npy', allow_pickle=True)
print(prediction_y.shape)

for sample in use_samples:
    samples_x.append(prediction_x[sample])
    samples_y.append(prediction_y[sample])

samples_x = np.array(samples_x)
samples_y = np.array(samples_y)

for i, sample in enumerate(samples_x):
    samples_x[i] = (sample - sample.min()) / (sample.max() - sample.min())

samples_x = np.expand_dims(samples_x, axis=-1)
samples_y = np.expand_dims(samples_y, axis=-1)

print(samples_x.shape)
print(samples_y.shape)

#Generate predictions for samples
predictions = model.predict(samples_x, batch_size=1)
correct = 0

predictions = np.argmax(predictions, axis=1)
samples_y = samples_y.flatten().astype('int')
for x in range(len(use_samples)):
    if predictions[x]== samples_y[x]:
        correct = correct + 1
print(predictions)
print(samples_y)
print("accuracy = " + str(correct/total))