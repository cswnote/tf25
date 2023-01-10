import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import platform

print(tf.__version__)

if platform.platform()[:3].lower() == 'win':
    cache_subdir = 'e:/workspace/Pycharm/tf25/dataset/'
elif platform.platform()[:3].lower() == 'mac':
    cache_subdir = '/Users/rainyseason/winston/Workspace/python/Pycharm Project/tf25/cats_and_dogs/'

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip',
                                      cache_subdir=cache_subdir,
                                      origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

batch_size = 32
img_size = (160, 160)
shuffle = True

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=False,
                                                            batch_size=batch_size,
                                                            image_size=img_size)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                              shuffle=shuffle,
                                                              batch_size=batch_size,
                                                              image_size=img_size)

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')

print('====================')
print('====================')
