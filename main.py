import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, LSTMCell, RNN, TimeDistributed
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential, Model
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell
y_data = [[1, 0, 2, 3, 3, 4]]  # ihello

num_classes = 5
input_dim = 5  # one-hot size, same as hidden_size to directly predict one-hot
sequence_length = 6  # |ihello| == 6
lr = 0.1

print(x_data)
x_data = to_categorical(x_data, num_classes=num_classes)
print(x_data)
print(x_data.shape)
y_data = to_categorical(y_data, num_classes=num_classes)
print(y_data.shape)

model = Sequential()

cell = LSTMCell(units=num_classes, input_shape=x_data.shape[1:])
model.add(RNN(cell=cell, return_sequences=True))
model.add(TimeDistributed(Dense(units=num_classes, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
# model.summary()
# plot_model(model, to_file='main.jpg', show_shapes=True)

model.fit(x_data, y_data, epochs=50)

model.summary()
plot_model(model, to_file='main.jpg', show_shapes=True)

predictions = model.predict(x_data)
print("=================================")
print(predictions)
print("=================================")
for i, prediction in enumerate(predictions):
    print(prediction)
    # print char using argmax, dict
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print("\tPredictions str: ", ''.join(result_str))