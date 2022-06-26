import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, InputLayer, Dense, MaxPooling1D, Normalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.activations import relu

print(tf.__version__)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()

train_features = dataset.sample(frac=0.8, random_state=0)
train_labels = train_features.pop('MPG')
test_features = dataset.sample(frac=0.2, random_state=0)
test_labels = test_features.pop('MPG')

horsepower = train_features['Horsepower']

horsepower_normalizer = Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = Sequential([
    horsepower_normalizer,
    Dense(1)
])
horsepower_model.summary()

def build_and_compile_model(norm, lr):
    model = Sequential([
        norm,
        Dense(64, activation=relu),
        Dense(64, activation=relu),
        Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr))

    return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer, 0.1)
history = dnn_horsepower_model.fit(train_features['Horsepower'], train_labels, epochs=20, validation_split=0.2, verbose=0)




print('================')
