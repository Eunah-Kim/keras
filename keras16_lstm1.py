import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1),
               return_sequences=True)) # return_sequences=True 시, 3차원 데이터로 반환
model.add(LSTM(2, activation='relu', return_sequences=True))
model.add(LSTM(3, activation='relu', return_sequences=True))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(LSTM(5, activation='relu', return_sequences=True))
model.add(LSTM(6, activation='relu', return_sequences=True))
model.add(LSTM(7, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(9, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=False))
model.add(Dense(5, activation='linear'))
model.add(Dense(1))

model.summary()

# 3. 모델 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=55, mode='auto')
# monitor : 지켜볼 평가지표 ; loss, mae, acc ..
# patience : 반복 허용 횟수
# mode : 'auto', 'min', 'max'
model.fit(x, y, epochs = 2000, batch_size=1,verbose=1,
          callbacks=[early_stopping])

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_input = array([[6.5,7.5,8.5], [50,60,70], [70,80,90], [100,110,120]])  #(3,) -> (1, 3) -> (1, 3, 1)
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
