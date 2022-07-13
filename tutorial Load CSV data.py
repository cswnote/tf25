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

fonts_zip = tf.keras.utils.get_file(
    'fonts.zip',  "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True
)

import pathlib

font_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob('*.csv'))

print(font_csvs)

fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = 'fonts/*.csv',
    batch_size=1, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)

for features in fonts_ds.take(1):
    for i, (name, value) in enumerate(features.items()):
        print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")

print('=======')


import re

def make_images(features):
    image = [None]*400
    new_feats = {}

    for name, value in features.items():
        match = re.match('r(\d+)c(\d+)', name)
        if match:
            print(int(match.group(1)))
            print(int(match.group(2)))
            image[int(match.group(1))*20 + int(match.group(2))] = value
        else:
            new_feats[name] = value

    image = tf.stack(image, axis=0)
    image = tf.reshape(image, [20, 20, -1])
    new_feats['image'] = image

    return new_feats

fonts_image_ds = make_images(features)