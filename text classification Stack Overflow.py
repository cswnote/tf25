import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPool2D, Dropout, Embedding, GlobalAveragePooling1D, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy

import os
import re
import shutil
import string
import matplotlib.pyplot as plt

print('tehsorflow version is {}'.format(tf.__version__))

file='stack_overflow_16k'
url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

# # 뭔가 이상함, cache_subdir=file로 한 후 '' 해야 directory 구조가 원하는 대로 됨
dataset = tf.keras.utils.get_file(file, origin=url, untar=True, cache_dir='.' , cache_subdir='')

os.listdir(dataset)
