import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

tf.random.set_seed(111)

(x_train_full, y_train_full), (x_test, y_test) = load_data(path='mnist.npz')

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=111)

num_x_train = x_train.shape[0]
num_x_val = x_val.shape[0]
num_x_test = x_test.shape[0]

print('all data: {}\tlabel: {}'.format(x_train_full.shape, y_train_full.shape))
print('train data: {}\tlabel: {}'.format(x_train.shape, y_train.shape))
print('validation data: {}\tlabel: {}'.format(x_val.shape, y_val.shape))
print('test data: {}\tlabel: {}'.format(x_test.shape, y_test.shape))


num_sample = 5
random_idxs = np.random.randint(60000, size=num_sample)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(random_idxs):
    img = x_train_full[idx, :]
    label = y_train_full[idx]

    plt.subplot(1, len(random_idxs), i+1)
    plt.imshow(img)
    plt.title('index: {}, label: {}'.format(idx, label))
plt.show()

print(np.max(x_train_full))
print(np.min(x_train_full))
