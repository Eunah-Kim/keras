import pandas as pd
from pandas import to_numeric
import numpy as np
from numpy import array

samsung = pd.read_csv('samsung.csv', encoding='euc-kr')

for column in samsung.columns[1:]:
    samsung[column] = samsung[column].map(lambda x: int(x.replace(',', '')))
samsung = samsung.sort_index(ascending=False)

samsung = samsung.iloc[:,1:]
samsung = samsung.astype(float)
print(samsung.info())

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
print(samsung_x.shape) #(426,1)


n_steps = 5
x, y = split_sequence3(samsung_x, n_steps)

# y 데이터 종가만 추출   
y = y[:,3:4]

for i in range(len(x)):
    print(x[i], y[i])

print(x.shape) #(421, 5, 5)
print(y.shape) #(421, 1)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# DNN 모델 만들기
x = x.reshape(421, 5, 5)

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

model = Sequential()
model.add(LSTM(60, activation = 'relu', input_shape=(5,5)))
model.add(BatchNormalization())
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=200, batch_size=40, validation_split=0.25)

loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

def x_pred(sequence,n_step):
    seq_x = list() 
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x=sequence[i:end_ix,:]
    return seq_x
 
x_pred =x_pred(samsung_x,5)
x_pred = x_pred.reshape(1,5,5)
print(x_pred)

y_predict = model.predict(x_pred)
print(y_predict)

y_test = [np.float(57200)]
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))
