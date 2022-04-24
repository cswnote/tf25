import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import InputLayer, Input, Flatten, Dense, TimeDistributed, LSTMCell, LSTM
from tensorflow.keras.models import Sequential, Model

import numpy as np


sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

print(sentence)

char_set = list(set(sentence))
char_dic = {key: i for i, key in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10 # Any arbitrary number
lr = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence)-sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1:i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[key] for key in x_str]
    y = [char_dic[key] for key in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

dataX = to_categorical(dataX, num_classes)
dataY = to_categorical(dataY, num_classes)

print(dataX.shape)
print(dataY.shape)

model = Sequential([
    LSTM(units=num_classes, input_shape=dataX.shape[1:], return_sequences=True),
    LSTM(units=num_classes, return_sequences=True), # 이 한 줄 실행에 따라 차이가 많이 남
    TimeDistributed(Dense(num_classes, activation='softmax'))
])
model.summary()
# plot_model(model, to_file='12-4-rnn_long_char.jpg', show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
model.fit(dataX, dataY, epochs=100)

results = model.predict(dataX)
print(results.shape)
prediceted_sentense = 'i'
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j == 0:  # print all for the first result to make a sentence
        # print(''.join([char_set[t] for t in index]), end='')
        for k in index:
            prediceted_sentense += char_set[k]
    else:
        # print(char_set[index[-1]], end='')
        # temp = char_set[index[-1]]
        # prediceted_sentense.append(char_set[index[-1]])
        prediceted_sentense += char_set[index[-1]]

print(sentence)
print(prediceted_sentense)

for i in range(len(sentence)):
    if sentence[i] != prediceted_sentense[i]:
        print(i, sentence[i], prediceted_sentense[i])