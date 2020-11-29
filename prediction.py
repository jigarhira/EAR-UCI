"""Neural Network prediction

Neural Network inference
Current structure type: CNN

Author: Ian Flores
"""

import tensorflow as tf
from tensorflow.keras.models import load_model

#Load the saved model
model_path = './saved_models'
model = load_model(model_path, compile = True)

#Insert samples
samples = []

#Generate predictions for samples
predictions = model.predict(samples)
print(predictions)