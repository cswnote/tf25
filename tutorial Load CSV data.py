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

font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)

num_font_features = len(font_line.split(','))
font_column_types = [str(), str()] + [float()] * (num_font_features - 2)

simple_font_ds = tf.data.experimental.CsvDataset(
    font_csvs, record_defaults=font_column_types, header=True)

font_files = tf.data.Dataset.list_files('fonts/*.csv')

def make_font_csv_ds(path):
    return tf.data.experimental.CsvDataset(
        path, record_defaults=font_column_types, header=True)

font_rows = font_files.interleave(make_font_csv_ds, cycle_length=3)

temp = font_rows.take(1)

print('=================')
print('=================')