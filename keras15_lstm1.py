import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])
y = array([4,5,6,7,8])
print(x.shape) # (5, 3)
print(y.shape) # (5,)

# x : 3차원 구성으로 reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
# x = x.reshape(5, 3, 1)


model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(3,1)))
# LSTM input_shape(열, 몇 개씩 자를지) ; 행 무시, 열 3, 1개씩 잘라서
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

from keras.callbacks import EarlyStopping, TensorBoard
# early_stopping = EarlyStopping(monitor='loss', patience=55, mode='auto')
tb_hist = TensorBoard(log_dir='./graph',
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

model.fit(x, y, epochs = 400, batch_size=1, callbacks=[tb_hist])

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_input = array([6,7,8])  #(3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)
