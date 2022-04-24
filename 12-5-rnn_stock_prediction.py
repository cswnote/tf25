import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import InputLayer, Input, Flatten, Dense, TimeDistributed, LSTMCell, LSTM
from tensorflow.keras.models import Sequential, Model

import numpy as np
import matplotlib.pyplot as plt
import os
# # activate at window
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# train Parameters
seq_length = 7
data_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

# train/test split
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

# # build datasets
def build_dataset(time_series, seq_length):
    dataX=[]
    dataY=[]
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i:i + seq_length, :]
        y = time_series[i + seq_length, [-1]]
        print(x, '->', y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)  # (505, 7, 5)
print(trainY.shape)

model = Sequential([
    LSTM(units=1, input_shape=trainX.shape[1:]),
    Dense(units=output_dim, activation='tanh')
])
model.summary()
plot_model(model, to_file='12-5-rnn_stock_prediction.jpg', show_shapes=True)

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
model.fit(trainX, trainY, epochs=iterations)


# Test step
test_predict = model.predict(testX)

# Plot predictions
plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()