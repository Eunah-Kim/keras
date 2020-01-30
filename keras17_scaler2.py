from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000], [30000, 40000, 50000],
           [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

print(x.shape)
print(y.shape)



from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(x.shape[0], x.shape[1], 1)

'''
scaler1 = MinMaxScaler()
# 최소값:0, 최대값:1, 0~1로 변환
# 이상치가 있다면 변환값이 매우 좁은 범위에 모임
scaler1.fit(x)
x1 = scaler1.transform(x)
print(x1)

scaler2 = StandardScaler()
# 평균이 0, 표준편차가 1이 되조록 변환
# 이상치가 있다면 평균, 표준편차에 영향 O
scaler2.fit(x)
x2 = scaler2.transform(x)
print(x2)

scaler3 = MaxAbsScaler()
# 최대 절대값과 0이 각각 1, 0이 되도록 변환
# MinMax와 유사, 이상치 영향 O
scaler3.fit(x)
x3 = scaler3.transform(x)
print(x3)

scaler4 = RobustScaler()
# 중앙값(median)과 IQR(interquartile range) 사용
# 아웃라이어의 영향을 최소화
scaler4.fit(x)
x4 = scaler4.transform(x)
print(x4)
'''

# 2. Train, Test split
x_train = x[:10, :]
x_test = x[10:,:]
y_train = y[:10]
y_test = y[10:]

# 3. Dense 모델 구현

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, input_shape=(3,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1))

# 4. 모델 훈련

model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])

model.fit(x_train, y_train, epochs = 100, batch_size=1)

# 5. 결과 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print(loss, mae)

x_input = array([[250, 260, 270]])
x_input = x_input.reshape(1,3,1)
y_predict_1 = model.predict(x_input)
print("y_predict_1: ", y_predict_1)

y_predict = model.predict(x_test)
print(y_predict)

#R2 구하기
from sklearn.metrics import r2_score
r2_pred = r2_score(y_test, y_predict)
print("r2: ", r2_pred)

