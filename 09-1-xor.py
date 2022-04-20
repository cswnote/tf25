import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
import numpy as np


x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# model = Sequential()
# model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

print(x_data.shape)

model = Sequential([InputLayer(input_shape=x_data.shape[1:], name='input'),
                    Dense(1, activation='sigmoid', name='dense1')])
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='09-1-xor.jpg', show_shapes=True)

history = model.fit(x_data, y_data, epochs=1000)

predictions = model.predict(x_data)
print('Prediction: \n', predictions)

score = model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])