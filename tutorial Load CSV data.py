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

inputs = {}
for name, column in titanic_features.items():
    print(name)
    print(column)
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1, ), name=name, dtype=dtype)

numeric_inputs = {name:input for name, input in inputs.items() if input.dtype==tf.float32}
x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = Normalization()

# titanic에서 dtype이 tf.float32인 열에 대한 normalize
# 단 열 이름을 titanic_features에서 구했으므로 survived는 빠짐
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    print(name, type(input.dtype))
    one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

print('================================')