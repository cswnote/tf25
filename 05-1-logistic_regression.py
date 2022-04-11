import tensorflow as tf

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

print(tf.shape(x_data))

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
# 위 1행을 아래 4행으로 바꾸려 했으나 안됨, 출력 shape 정의가 안되서어 인가 ㅠㅜ
# tf.model.add(tf.keras.layers.Input(shape=(2, ))) # shape을 (2, ) 혹은 (6, 2) 으로 하면 돌아는 감, 결과는 엉망
# tf.model.add(tf.keras.layers.Dense(300, activation='sigmoid'))
# tf.model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
# tf.model.add(tf.keras.layers.Dense(10, activation='sigmoid'))


# use sigmoid activation for 0~1 problem
tf.model.add(tf.keras.layers.Activation('sigmoid'))

''' 
better result with loss function == 'binary_crossentropy', try 'mse' for yourself
adding accuracy metric to get accuracy report during training
'''

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=5000)

print("Accuracy: ", history.history['accuracy'][-1])