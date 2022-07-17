import pathlib

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

import os

path = os.getcwd()

import pathlib
font_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob('*.csv'))
font_csvs[:10]

BATCH_SIZE = 2048
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = 'fonts/*.csv', batch_size=BATCH_SIZE,
    num_epochs=1,num_parallel_reads=100)

print('=================')
print('=================')