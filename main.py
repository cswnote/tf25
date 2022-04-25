import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, LSTMCell, LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import cifar10

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

num_sample = 25
random_idxs = np.random.randint(x_train.shape[0], size=num_sample)

plt.figure(figsize=(12, 25))
for i, idx in enumerate(random_idxs):
    img = x_train[idx, :]
    label = y_train[idx]

    plt.subplot(5, int(len(random_idxs) / 5), i + 1)
    plt.imshow(img)
    plt.title('index: {}, label: {}'.format(idx, label))
plt.show()