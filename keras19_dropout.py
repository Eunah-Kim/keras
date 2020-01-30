import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Dropout, BatchNormalization

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)

###### BatchNormalization을 하는 이유  ######
# 모델에서 조절되는 가중치 값을 정규화함 > 가중치 값이 계산학 쉽게 정리 > 성능^
# Dropout과 함께 썼을 때 성능이 나빠질 수 있음 > 함께 사용 비권장
# 단, GAN의 경우 Dropout과 BatchNormalization을 함께 쓰는 경우가 많음


# 2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) # return_sequences=True 시, 3차원 데이터로 반환
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dropout(0.3))
# model.add(BatchNormalization())
model.add(Dense(5, activation='linear'))
model.add(Dense(1))

model.summary()

# 3. 모델 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

model.fit(x, y, epochs = 100, batch_size=1)

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_input = array([[6.5,7.5,8.5], [50,60,70], [70,80,90], [100,110,120]])  #(3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
