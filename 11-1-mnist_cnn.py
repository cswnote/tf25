import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.
x_train = x_train / 255.
print(x_train.shape)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

lr = 0.001
training_epochs = 12
batch_size = 128

model = Sequential([InputLayer(input_shape=x_train.shape[1:]),
                    Conv2D(filters=16, kernel_size=(3, 3), input_shape=x_train.shape[1:], activation='relu'),
                    MaxPool2D(pool_size=(2, 2)), # strides=2 가 기본인 것으로 보아, pool_size와 연동되는 듯
                    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
                    MaxPool2D(pool_size=(2, 2)),
                    Flatten(),
                    Dense(10, kernel_initializer='glorot_normal', activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics='accuracy')
model.summary()
plot_model(model, to_file='11-1-mnist_cnn.py.png')

model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

y_predicted = model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0] - 1)
print('index: ', random_index, 'actual y: ', np.argmax(y_test[random_index]),
      'predicted y:', np.argmax(y_predicted[random_index]))

evaluation = model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy: ', evaluation[1])