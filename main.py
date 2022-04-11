import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Activation, Dense, Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Sequential, Model
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = Sequential([InputLayer(input_shape=x_data.shape[1:], name='input'),
                    Dense(10, activation='sigmoid', name='dense_0'),
                    Dense(10, activation='sigmoid', name='dense_1'),
                    Dense(10, activation='sigmoid', name='dense_2'),
                    Dense(10, activation='sigmoid', name='dense_3'),
                    Dense(1, activation='sigmoid', name='output')])
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='main.png')

history = model.fit(x_data, y_data, epochs=5000)

prediction = model.predict(x_data)
print('Prediction: \n', prediction)

score = model.evaluate(x_data, y_data)
print('Loss: ', score[0])
print('Accuracy: ', score[1])