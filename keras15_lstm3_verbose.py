import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1)))
# LSTM input_shape(열, 몇 개씩 자를지) ; 행 무시, 열 3, 1개씩 잘라서
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

# model.summary()

# 3. 모델 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

model.fit(x, y, epochs = 100, batch_size=1,verbose=0)
# verbose = 0 ; 훈련과정을 보여주지 않겠다.
# verbose = 1 ; 훈련과정을 보여주겠다. (default)
# verbose = 2 ; loss, mae 값만 노출(간결)
# verbose = 3~ ; epochs 만 노출

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_input = array([[6.5,7.5,8.5], [50,60,70], [70,80,90], [100,110,120]])  #(3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
