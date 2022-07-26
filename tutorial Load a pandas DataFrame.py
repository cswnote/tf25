import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer, Dense, Flatten, Normalization
from tensorflow.keras import Sequential

print(tf.__version__)

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


class MyModel(tf.keras.Model):
    def __init__(self):
        # Create all the internal layers in init.
        super().__init__(self)

        self.normalizer = Normalization(axis=-1)

        self.seq = Sequential([
                               self.normalizer,
                               Dense(10, activation='relu'),
                               Dense(10, activation='relu'),
                               Dense(1)
        ])

    def adapt(self, inputs):
        # Stack the inputs and `adapt` the normalization layer.
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)

    def call(self, inputs):
        # Stack the inputs
        inputs = stack_dict(inputs)
        # Run them through all layers.
        result = self.seq(inputs)

        return result


df = pd.read_csv(csv_file)
print('df')
print(df.head())
print(df.dtypes)
print()

target = df.pop('target')

numeric_features_name = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_features_name]
print('numeric_features')
print(numeric_features)
print()

print("numeric_features_tf is applied numeric_features to 'tf.conver_to_tensor'")
numeric_features_tf = tf.convert_to_tensor(numeric_features, target)
print(numeric_features_tf.numpy())
print()

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)
print()

print('normalizer numeric_features')
print(normalizer(numeric_features.iloc[:3]))
print()

print("numeric_dataset is applied (numeric_features, target) to 'Dataset.from_tensor_slices'")
numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))
for row in numeric_dataset.take(3):
    print(row)
print()

print("numeric_dict_ds: apply 'Dataset.from_tenor_slices on (dict(numeric_features), target)")
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))
for row in numeric_dict_ds.take(3):
    print(row)
print()








print("======================================")