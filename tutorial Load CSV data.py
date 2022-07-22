import os

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

import pathlib

titanic_file_path = os.getcwd() + '/titanic/train.csv'

text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]

titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]

features = tf.io.decode_csv(lines, record_defaults=titanic_types)

for f in features:
    print(f"type: {f.dtype.name}, shape: {f.shape}")


