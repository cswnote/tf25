import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import InputLayer, Input, Flatten, Dense, TimeDistributed, LSTMCell, LSTM
from tensorflow.keras.models import Sequential, Model

import numpy as np


sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

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

