import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPool2D, Dropout, Embedding, GlobalAveragePooling1D, Activation, Normalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

print(tf.__version__ )

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

titanic_features_dict = {name: np.array(value)
                         for name, value in titanic_features.items()}

import itertools

def slices(features):
    for i in itertools.count():
        example = {name1: value1[i] for name1, value1 in features.items()}
        yield example

for example in slices(titanic_features_dict):
    for name2, value2 in example.items():
        print(f"{name2:19}: {value2}")
