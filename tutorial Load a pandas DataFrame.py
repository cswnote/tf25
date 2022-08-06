import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Normalization
from tensorflow.keras import Sequential, Input

print(tf.__version__)

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)

target = df.pop('target')

binary_feature_names = []
categorical_feature_names = []

for name, values in df.items():
    nunique = df[name].nunique()
    print(f"{name:10s}: {nunique}", end='\t')
    if nunique == 2:
        print(f"recommand to Binary")
        binary_feature_names.append(name)
    elif nunique <= 10:
        print(f"recommand to Categorical")
        categorical_feature_names.append(name)
    else:
        print(f"recommand to numerical")

inputs = {}
for name, column in df.items():
    if type(column[0]) == str:
        dtype = tf.string
    elif (name in categorical_feature_names or
          name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32

    inputs[name] = Input(shape=(), name=name, dtype=dtype)

