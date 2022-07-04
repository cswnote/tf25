import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

plt.imshow(img_train[0])

print(np.max(img_train))
print(np.min(img_train))

