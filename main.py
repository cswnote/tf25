import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, LSTMCell, LSTM, Dropout, Conv2D, MaxPool2D, Conv3D, MaxPool3D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.datasets import cifar10

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(111)

(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(x_train_full.shape, x_test.shape)
print(y_train_full.shape, y_test.shape)



num_sample = 25
random_idxs = np.random.randint(x_train_full.shape[0], size=num_sample)
plt.figure(figsize=(14, 25))
for i, idx in enumerate(random_idxs):
    img = x_train_full[idx, :]
    # label = y_train[idx]
    label = class_names[y_train_full[idx][0]]

    plt.subplot(5, int(len(random_idxs) / 5), i + 1)
    plt.imshow(img)
    plt.title('index: {}, label: {}'.format(idx, label))
plt.show()

x_train_full = x_train_full / 255.

data_split = False

if data_split:
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=111)
else:
    x_train = x_train_full
    y_train = y_train_full

model = Sequential([InputLayer(input_shape=x_train.shape[1:], name='input'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
            ])

model.summary()
plot_model(model, to_file='main.jpg', show_shapes=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if data_split:
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))
else:
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))


record = history.history
print('train record: ', record['loss'][-1], record['accuracy'][-1])
print('validation record: ', record['val_loss'][-1], record['val_accuracy'][-1])

train_loss = record['loss']
train_accuracy = record['accuracy']
val_loss = record['val_loss']
val_accuracy = record['val_accuracy']

epochs = range(1, len(train_loss) + 1)
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, train_loss, color='blue', label='train_loss')
ax1.plot(epochs, val_loss, color='red', label='validation_loss')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid()
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, train_accuracy, color='blue', label='train_accuracy')
ax2.plot(epochs, val_accuracy, color='red', label='validation_accuracy')
ax2.set_title('Train and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid()
ax2.legend()

plt.show()