import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer, Dense, Flatten, Normalization

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)

target = df.pop('target')

numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)

def get_basic_model():
    model = tf.keras.Sequential([
                                 normalizer,
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

# model = get_basic_model()
# model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
    print(row)

values = []
for key in sorted(dict(numeric_features)):
    values.append(tf.cast(dict(numeric_features)[key], tf.float32))

print("==========================")