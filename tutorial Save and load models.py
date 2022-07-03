import os.path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=train_images.shape[1:]),
        Dropout(0.2),
        Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


if __name__ == '__main__':

    print(tf.__version__)

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    checkpoint_path = 'training_1/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    model = create_model()
    model.summary()

    model.fit(train_images, train_labels, epochs=10,
                  validation_data=(test_images, test_labels), callbacks=[cp_callback])