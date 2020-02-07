import numpy as np
import pandas as pd

samsung = np.load('./samsung/data/samsung1.npy')

print(samsung) 
print(samsung.shape) #(430,5)

# x, y 데이터 정의
def split_xy5(dataset, time_steps, y_column):
    X, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):
            break
        tem_x = dataset[i:x_end_number, : ]
        tem_y = dataset[x_end_number:y_end_number, 3]
        X.append(tem_x)
        y.append(tem_y)
    return np.array(X), np.array(y)

x, y = split_xy5(samsung, 5, 1)
print(x.shape)  #(425,5,5)
print(y.shape)  #(425, 1)
print(x[0,:], "\n", y[0])


# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3, shuffle=False)

print(x_train.shape)  #(297,5,5)
print(x_test.shape)  #(128,5,5)


## 데이터 전처리
# StandardScaler

# 3차원 -> 2차원
x_train = np.reshape(x_train,
            (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test,
            (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# x_train = x_train.reshape(297,25)
# x_test = x_test.reshape(128,25)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0,:])

# 2차원 -> 3차원
x_train_scaled = x_train_scaled.reshape(297,25,1)
x_test_scaled = x_test_scaled.reshape(128,25,1)

# 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

model = Sequential()
model.add(LSTM(120, activation='relu', input_shape=(25,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

model.fit(x_train_scaled, y_train, epochs=100, batch_size=25, 
          validation_split=0.25)

loss, mae = model.evaluate(x_test_scaled, y_test, batch_size=1)
print("mae: ", mae)


# x_pred의 가장 최근 날짜 5행을 추출 (1/31~2/6 데이터)
def x_pred(sequence,n_step):
    seq_x = list() 
    for i in range(len(sequence)):
        end_ix = i + n_step
        if end_ix > len(sequence):
            break
        seq_x=sequence[i:end_ix,:]
    return seq_x

x_pred =np.array(x_pred(samsung,5))
print(x_pred.shape)


# scaler 적용
x_pred = x_pred.reshape(1,25)
x_pred_scaled = scaler.transform(x_pred)
x_pred_scaled = x_pred_scaled.reshape(1,25,1)
# print(x_pred_scaled)

# 주가 예측
y_predict = model.predict(x_pred_scaled)
print("2/7 삼성전자 예측 종가: ", y_predict)

# RMSE 계산
y_test = [np.float(60100)]  # 2/7 pm 1:00 정답 주가
print("2/7 pm 1 기준 삼성 주가 : 60,100원")
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("2/7 pm 1 기준 종가 RMSE: ", RMSE(y_test, y_predict))


#-------  train_test_split를 적용했을 때 결과  -------
# 2/7 삼성전자 예측 종가:  [[60034.844]]
# 2/7 pm 1 기준 삼성 주가 : 60,100원
# 2/7 pm 1 기준 종가 RMSE:  65.15625