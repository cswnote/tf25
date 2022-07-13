import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Normalization, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

# Create a symbolic input
input = keras.layers.Input(shape=(), dtype=tf.float32)

# Perform a calculation using the input
result = 2 * input + 1

# # one example of using Model layer, input is tf tensor
calc = Model(inputs=input, outputs=result)

# # change data type to tf datatype and then put it in dict
inputs = {}
for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    inputs[name] = keras.Input(shape=(1,), name=name, dtype=dtype)

# # extract numeric type data
numeric_inputs = {name:input for name, input in inputs.items() if input.dtype==tf.float32}

for key, value in numeric_inputs.items():
    print(value)

print('============')
print(list(numeric_inputs.values()))

# # maybe make list data to tensor layer
x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = keras.layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.apend(x)