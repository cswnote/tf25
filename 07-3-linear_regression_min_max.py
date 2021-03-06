import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, InputLayer, Dense, Flatten, Activation
import numpy as np

def min_max_scaler(data):
    a = np.min(data, 0)
    numerator = data - np.min(data, 0)
    b = np.max(data, 0)
    denominator = np.max(data,0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)


# very important. It does not work without it.
xy = min_max_scaler(xy)
print(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

model = Sequential()
model.add(Dense(units=1, input_shape=x_data.shape))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer=SGD(learning_rate=1e-5))
model.summary()

history = model.fit(x_data, y_data, epochs=1000)
predictions = model.predict(x_data)
score = model.evaluate(x_data, y_data)

print('Prediction: \n', predictions)
print('Cost: ', score)