import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


# try different learning_rate
learning_rate = 65535  # ? it works too hahaha
# learning_rate = 0.1
# learning_rate = 1e-10  # small learning rate won't work either

model = Sequential()
# model.add(Input(shape=(8, 3)))    # why error???
model.add(Dense(units=3, input_dim=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=learning_rate), metrics=['accuracy'])

model.summary()

model.fit(x_data, y_data, epochs=1000)

#predict
pred1 = model.predict(x_test)
print(pred1)
pred2 = model.predict_classes(x_test)
print(pred2)

# Calculate the accuracy
accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(accuracy[1])