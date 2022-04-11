import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = Sequential([InputLayer(input_shape=x_data.shape[1:], name='input'),
                    Dense(32, activation='sigmoid', name='dense_0'),
                    Dense(1, activation='sigmoid', name='output')])
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.summary()
plot_model(model)

history = model.fit(x_data, y_data, epochs=10000)

predictions = model.predict(x_data)
print('Prediction: \n', predictions)

score = model.evaluate(x_data, y_data)
print('Loss: ', score[0])
print('Accuracy: ', score[1])