import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Normalization, Dense

abalone_train = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
                           names=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                                 'Viscera weight', 'Shell weight', 'Age'])

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
print(abalone_features)