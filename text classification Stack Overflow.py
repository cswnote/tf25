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
dataset = tf.keras.utils.get_file(file, origin=url, untar=True, cache_dir='.' , cache_subdir=file)
dataset = tf.keras.utils.get_file(file, origin=url, untar=True, cache_dir='.' , cache_subdir='')

train_dir = os.path.join(dataset, 'train')
print(os.listdir(train_dir))

batch_size = 32
seed = 42

raw_train = tf.keras.utils.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)

a = next(iter(raw_train.take(1)))
print(a[0][0])
print(raw_train.class_names[a[1][0]])
print(a[1][0])

for text, label in raw_train.take(1):
    for i in range(1):
        print('=================================')
        print(raw_train.class_names[label[i]])
        print('=================================')
        print(text[i].numpy())
        print('=================================')
        print(text[i])
        print('=================================')

val_train = tf.keras.utils.text_dataset_from_directory(
    'stack_overflow_16k/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

test_dir = os.path.join(dataset + '/test')
test = tf.keras.utils.text_dataset_from_directory(test_dir, batch_size=batch_size)

sub_dir = os.listdir(train_dir)[0]
sample_file = os.path.join(train_dir, sub_dir)
sample_file = os.path.join(sample_file, os.listdir(sample_file)[0])

# max_features = 10000
# sequence_length = 250
#
# def custom_standardization(input_data):
#     lowercase = tf.strings.lower(input_data)
#     stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
#     return tf.strings.regex_replace(stripped_html,
#                                   '[%s]' % re.escape(string.punctuation),
#                                   '')
#
# vectorize_layer = tf.keras.layers.TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=max_features,
#     output_mode='int',
#     output_sequence_length=sequence_length)
#
# max_features = 10000
# sequence_length = 250
#
# vectorize_layer = tf.keras.layers.TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=max_features,
#     output_mode='int',
#     output_sequence_length=sequence_length)
#
# def vectorize_text(text, label):
#     text = tf.expand_dims(text, -1)
#     return vectorize_layer(text), label

real_y = list(test.map(lambda x, y: y))

test_y = []
for y in real_y:
  for i in y:
    test_y.append(i)

print('=======')
