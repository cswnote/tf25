import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization, Embedding

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

print(tf.__version__)

data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

dataset_dir = utils.get_file(origin=data_url,
                            untar=True,
                            cache_dir='/Users/rainyseason/winston/Workspace/python/Pycharm Project/tf25/stack_overflow/cache',
                            cache_subdir='/Users/rainyseason/winston/Workspace/python/Pycharm Project/tf25/stack_overflow')

dataset_dir = pathlib.Path(dataset_dir).parent

for item in list(dataset_dir.iterdir()):
    print(item)

train_dir = dataset_dir / 'train'
test_dir = dataset_dir / 'test'
train_py_dir = train_dir / 'python'

sample_file = train_py_dir / '0.txt'

with open(sample_file) as f:
    print(f.read())




print('===================')
print('===================')