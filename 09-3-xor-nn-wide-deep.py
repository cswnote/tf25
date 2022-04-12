import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten
from tensorflow.keras.utils import plot_model
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

print(x_data.shape)

model = Sequential([InputLayer(input_shape=x_data.shape[1], name='input'),
                    Dense(10, activation='sigmoid'),
                    Dense(10, activation='sigmoid'),
                    Dense(10, activation='sigmoid'),
                    Dense(10, activation='sigmoid'),
                    Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='09-3 model.png')

history = model.fit(x_data, y_data, epochs=5000)

predictions = model.predict(x_data)
print('Predictions: \n', predictions)

score = model.evaluate(x_data, y_data)
print("Accuracy: ", score[1])
print('----------------')
print(score)