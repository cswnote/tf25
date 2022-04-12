import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = Sequential([
    InputLayer(input_shape=x_data.shape[1:], name='input'),
    Dense(2, activation='sigmoid', name='dense0'),
    Dense(1, activation='sigmoid', name='dense1')
])

model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='09-4 model.png')

# prepare callback
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# add callback param to fit()
history = model.fit(x_data, y_data, epochs=10000, callbacks=[tensorboard_callback])

predictions = model.predict(x_data)
print('Prediction: \n', predictions)

score = model.evaluate(x_data, y_data)
print('Loss: ', score[0])
print('Accuracy: ', score[1])

'''
at the end of the run, open terminal / command window
cd to the source directory
tensorboard --logdir logs/fit
read more on tensorboard: https://www.tensorflow.org/tensorboard/get_started
'''