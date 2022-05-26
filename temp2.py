import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Input, InputLayer, Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.datasets.fashion_mnist import load_data
from keras.layers import Input, InputLayer, Flatten, Dense, Conv2D, MaxPool2D
# from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.datasets.fashion_mnist import load_data

from sklearn.model_selection import train_test_split

print(tf.__version__)

tf.random.set_seed(111)

(x_train_full, y_train_full), (x_test, y_test) = load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size = 0.3, random_state=111)

print('full data: {}\tlabel: {}'.format(x_train_full.shape, y_train_full.shape))
print('train data: {}\tlabel: {}'.format(x_train.shape, y_train.shape))
print('validation data: {}\tlabel: {}'.format(x_val.shape, y_val.shape))
print('test data: {}\tlabel: {}]'.format(x_test.shape, y_test.shape))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names[y_train[0]]

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

num_sample = 4

random_idx = np.random.randint(len(x_train_full), size=num_sample)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(random_idx):
  image = x_train_full[idx]
  label = y_train_full[idx]

  plt.subplot(1, len(random_idx), i + 1)
  plt.imshow(image)
  plt.title('index: {}, label: {}'.format(idx, class_names[label]))
plt.show()