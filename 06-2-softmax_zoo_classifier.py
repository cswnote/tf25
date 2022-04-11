import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model, to_categorical
import numpy as np


# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7

y_one_hot = to_categorical(y_data) # 기존 코드에는 y_data 뒤에 nb_classes를 넣었으나 안 넣어도 됨
print("one_hot:", y_one_hot)

model = Sequential()
model.add(InputLayer(input_shape=x_data.shape))
model.add(Dense(units=nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])
model.summary()

history = model.fit(x_data, y_one_hot, epochs=1000)

# Single data test
test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]]) # expected prediction == 3 (feathers)
pred = model.predict(test_data)
print(pred.shape)
print(pred[0])
print(np.argmax(pred, axis=-1))
# print(tf.model.predict(test_data), tf.model.predict_classes(test_data)) # 더 이상 지원하지 않으므로 위 형식으로

temp = y_data.flatten()

# Full x_data test
pred = model.predict_classes(x_data)
# pred = model.predict(x_data)  # 아래 실행시 결과 차이가 남, 특히 p == int(y) 에서
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))