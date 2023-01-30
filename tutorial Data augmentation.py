import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import platform


from tensorflow.keras import layers

if platform.platform() == 'Windows-10-10.0.19044-SP0':
    path = os.getcwd() + '/dataset/'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
    data_dir=path
)

num_classes = metadata.features['label'].num_classes
print("number of classes: {}".format(num_classes))

get_label_name = metadata.features['label'].int2str

temp = iter(train_ds)
image, label = next(temp)
_ = plt.imshow(image)
_ = plt.title(get_label_name[label])

print('========================')