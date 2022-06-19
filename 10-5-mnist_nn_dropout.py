
port tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
import numpy as np
import random

random.seed(777)
lr = 0.001
batch_size = 100
training_epochs = 15
drop_rate = 0.3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    InputLayer(input_shape=x_train.shape[1:], name='input'),
    Flatten(input_shape=x_train.shape[1:], name='flatten'),
    Dense(512, kernel_initializer='glorot_normal', activation='relu', name='dense_1'),
    Dropout(drop_rate),
    Dense(512, kernel_initializer='glorot_normal', activation='relu', name='dense_2'),
    Dropout(drop_rate),
    Dense(512, kernel_initializer='glorot_normal', activation='relu', name='dense_3'),
    Dropout(drop_rate),
    Dense(512, kernel_initializer='glorot_normal', activation='relu', name='dense_4'),
    Dropout(drop_rate),
    Dense(10, kernel_initializer='glorot_normal', activation='softmax', name='dense_5'),
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='10-5-mnist_nn_dropout.png', show_shapes=True)

history = model.fit(x_train, y_train, epochs=training_epochs)

y_predicted = model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print('index: ', random_index,
          'actual y: ', np.argmax(y_test[random_index]),
          'predicted y: ', np.argmax(y_predicted[random_index]))

evaluation = model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy: ', evaluation[1])