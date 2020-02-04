import pandas as pd
from pandas import to_numeric
import numpy as np
from numpy import array

samsung = pd.read_csv('samsung.csv', encoding='euc-kr')
kospi = pd.read_csv('kospi200.csv', encoding='euc-kr')

samsung = samsung.iloc[:,1:]
kospi = kospi.iloc[:,1:]

for column in samsung.columns[:]:
    samsung[column] = samsung[column].map(lambda x: int(x.replace(',', '')))
samsung = samsung.sort_index(ascending=False)

for column in kospi.columns[4:]:
    kospi[column] = kospi[column].map(lambda x: float(x.replace(',', '')))
kospi = kospi.sort_index(ascending=False)

samsung.info()
kospi.info()

samsung = samsung.astype(float)
# kospi = kospi.astype(float)

samsung.info()


def split_sequence3(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        # 다항식 연산
        seq_x, seq_y = sequence[i:end_ix, : ], sequence[end_ix, : ]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


samsung_x = np.array(samsung)
kospi_x = np.array(kospi)
print(samsung_x.shape) #(426,1)
print(kospi_x.shape) #(426,1)

n_steps = 5
x1, y1 = split_sequence3(samsung_x, n_steps)
x2, y2 = split_sequence3(kospi_x, n_steps)

# y 데이터 종가만 추출   
y1 = y1[:,3:4]
y2 = y2[:,3:4]

for i in range(len(x1)):
    print(x1[i], y1[i])

for i in range(len(x2)):
    print(x2[i], y2[i])

print(x1.shape) #(421, 5, 5)
print(y1.shape) #(421, 1)
print(x2.shape) #(421, 5, 5)
print(y2.shape) #(421, 1)


# 앙상블 LSTM 모델 만들기
x1 = x1.reshape(421, 5, 5)
y1 = y1.reshape(421, 1)
x2 = x2.reshape(421, 5, 5)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

input1 = Input(shape=(5,5))
dense1 = LSTM(60, activation = 'relu')(input1)
dense1 = Dense(40)(dense1)
output1 = Dense(30)(dense1)

input2 = Input(shape=(5,5))
dense2 = LSTM(30, activation = 'relu')(input2)
dense2 = Dense(20)(dense2)
output2 = Dense(10)(dense2)
# 앙상블 시 dense의 수, node의 수가 달라도 상관없음

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(20)(merge1)
middle2 = Dense(10)(middle1)
output = Dense(1)(middle2)

# 마지막에 함수형 모델 정의(input, output layer 명시, 여러 개일 경우 list로 연결)
model = Model(inputs = [input1, input2], outputs = output)

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit([x1,x2], y1, epochs=100, batch_size=40, validation_split=0.25)

loss, mae = model.evaluate([x1,x2], y1, batch_size=1)
print(loss, mae)


def x_pred(sequence,n_step):
    seq_x = list() 
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x=sequence[i:end_ix,:]
    return seq_x
 
x_pred1 =x_pred(samsung_x,5)
x_pred2 = x_pred(kospi_x,5)
x_pred1 = x_pred1.reshape(1,5,5)
x_pred2 = x_pred2.reshape(1,5,5)
print(x_pred1)
print(x_pred2)

x_pred = [x_pred1, x_pred2]
# print(x_pred.shape)
y_predict = model.predict(x_pred)
print(y_predict)


y_test = [np.float(57200)]
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))