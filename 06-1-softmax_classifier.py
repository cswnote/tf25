import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
import numpy as np

x_raw = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_raw = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_data = np.array(x_raw, dtype=np.float32)
y_data = np.array(y_raw, dtype=np.float32)

print(x_data.shape)

nb_classes = 3

model = Sequential()
model.add(Input(shape=x_data.shape))
# use softmax activations: softmax = exp(logits) / reduce_sum(exp(logits), dim)
model.add(Dense(units=nb_classes, use_bias=True, activation='softmax'))

# use loss == categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.summary()
plot_model(model)

history = model.fit(x_data, y_data, epochs=2000)

print('--------------')
# Testing & One-hot encoding
a = model.predict(np.array([[1, 11, 7, 9]]))
# print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))
print(a, np.argmax(a, axis=1))

print('--------------')
b = model.predict(np.array([[1, 3, 4, 3]]))
print(b, np.argmax(b, axis=1))

print('--------------')
# or use argmax embedded method, predict_classes
c = model.predict(np.array([[1, 1, 0, 1]]))
c_onehot = model.predict_classes(np.array([[1, 1, 0, 1]]))
print(c, c_onehot)

print('--------------')
all = model.predict(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
all_onehot = model.predict_classes(np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]))
print(all, all_onehot)