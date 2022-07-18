import pandas as pd
import numpy as np

# make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Normalization

print(tf.__version__)

import os
path = os.getcwd()
abalone_train = pd.read_csv(
    path + "/abalone/abalone_train.csv",
    # "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

# # Normalization. It is possible that normalize input type is 'dataframe'(is composed only numeric data)
normalize = Normalization()
normalize.adapt(abalone_features)

norm_abalone_model = Sequential([normalize,
                                 Dense(64),
                                 Dense(1)])
norm_abalone_model.compile(loss=keras.losses.MeanSquaredError(),
                           optimizer=keras.optimizers.Adam())
norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)

print("=======================")
print("=======================")