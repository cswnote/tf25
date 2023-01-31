import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

# # Setup
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
)

num_classes = metadata.features['label'].num_classes

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
plt.imshow(image)
plt.title(get_label_name(label));

# # Resizing and rescaling
IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

result = resize_and_rescale(image)
_ = plt.imshow(result)





num_classes = metadata.features['label'].num_classes
print("number of classes: {}".format(num_classes))

get_label_name = metadata.features['label'].int2str

temp = iter(train_ds)
image, label = next(temp)
_ = plt.imshow(image)
_ = plt.title(get_label_name[label])

print('========================')
