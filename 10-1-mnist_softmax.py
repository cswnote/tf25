import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Sequential


learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing data
x_train, x_test = x_train / 255.0, x_test / 255.0

# change data shape
# print(x_train.shape)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
# print(x_train.shape[1:])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([InputLayer(input_shape=x_train.shape[1:], name='input'),
                    Flatten(input_shape=x_train.shape[1:], name='Flatten'),
                    Dense(10, activation='softmax', name='output')])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
model.summary()
plot_model(model, to_file='10-1-minist_softmax.png')

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

predictions = model.predict(x_test)
print('Prediction: \n', predictions)
x_train
score = model.evaluate(x_train, y_train)
print('Accuracy: ', score[1])
