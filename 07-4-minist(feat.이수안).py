import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model

from sklearn.model_selection import train_test_split

# testing, 테스트 중

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

# # show sample
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
print(y_test[0])

print('x max value =', np.max(x_train_full))
print('x min value =', np.min(x_train_full))

x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

print(y_train[0])
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
print(y_train[0])


model = Sequential([InputLayer(input_shape=(28, 28), name='input'),
                    Flatten(input_shape=[28, 28], name='flatten'),
                    Dense(100, activation='relu', name='dense1'),
                    Dense(64, activation='relu', name='dense2'),
                    Dense(32, activation='relu', name='dense3'),
                    Dense(10, activation='softmax', name='output')])
model.summary()
plot_model(model, show_shapes=True, to_file='main_model.png')

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_val, y_val))

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, loss, color='blue', label='train_loss')
ax1.plot(epochs, val_loss, color='red', label='val_loss')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Ephocs')
ax1.set_ylabel('Loss')
ax1.grid()
ax1.legend()

accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, accuracy, color='blue', label='train_accuracy')
ax2.plot(epochs, val_accuracy, color='red', label='val_accuracy')
ax2.set_title('Train and Validation Accuracy')
ax2.set_xlabel('Ephocs')
ax2.set_ylabel('Accuracy')
ax2.grid()
ax2.legend()

plt.show()