import keras
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

tf.random.set_seed(123)

# Test dataset
wp = np.array([[0.2,1.0], [0.2,0.925], [0.2,0.895]])
z = keras.backend.random_normal(shape=(wp.shape[0],20))

# Loading model test
pacvae = keras.models.load_model('./saved_model/pacvae')

# Prediction without loading model parameters
input = tf.concat([z, wp], axis=1)
current_ = pacvae.decoder.predict(input) # Input as [z, [baseTemp,eca]]
plt.plot(current_[0])
plt.plot(current_[1])
plt.plot(current_[2])
plt.show()