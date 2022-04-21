import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, Conv2D, LSTM, LSTMCell, TimeDistributed
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model
import numpy as np

sample = 'if you want you'
idx2char = list(set(sample))
print('idx2char: ', idx2char)
char2idx = {c: i for i , c in enumerate(idx2char)}
print('char2idx: ', char2idx)

# # hyper parameters
dic_size = len(char2idx) # RNN input size (one hot size)
hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size (RNN or softmax, etc.)
batch_size = 1
sequence_length = len(sample)-1 # number of lstm rollings (unit #)
lr = 0.1

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

x_one_hot_eager = tf.one_hot(x_data, num_classes)
print(x_one_hot_eager.shape)
x_one_hot_numpy = to_categorical(x_data, num_classes)
print(x_one_hot_numpy.shape)
y_one_hot_eager = to_categorical(y_data, num_classes)

model = Sequential([
    LSTM(units=num_classes, input_shape=x_one_hot_numpy.shape[1:], return_sequences=True),
    TimeDistributed(Dense(units=num_classes, activation='softmax'))
])
model.summary()
plot_model(model, to_file='main.jpg', show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
model.fit(x_one_hot_numpy, y_one_hot_eager, epochs=50)

predictions = model.predict(x_one_hot_numpy)

for i , prediction in enumerate(predictions):
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print('\tPredictions str: ', ''.join(result_str))
