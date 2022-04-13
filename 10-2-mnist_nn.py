import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model

import numpy as np
import random

random.seed(777)
lr = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    InputLayer(input_shape=x_train.shape[1:], name='input'),
    Flatten(input_shape=x_train.shape[1:], name='Flatten'),
    Dense(256, activation='relu', name='hidden_1'),
    Dense(nb_classes, activation='softmax', name='output')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='10-2-mnist_nn.png', show_shapes=True)

histroy = model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size)

y_predicted = model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0] - 1)
    print('index: ', random_index,
          "actual_y", np.argmax(y_test[random_index]),
          'predicted_y: ', np.argmax(y_predicted[random_index]))

evaluation = model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy: ', evaluation[1])