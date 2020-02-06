import numpy as np
import pandas as pd

samsung = np.load('./samsung/data/samsung.npy')
kospi200 = np.load('./samsung/data/kospi200.npy')

# print(samsung)
# print(samsung.shape)
# print(kospi200)
# print(kospi200.shape)

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

x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, random_state=1, test_size = 0.3, shuffle=False)

print(x1_train.shape)  #(294,5,5)
print(x1_test.shape)  #(127,5,5)
print(x2_train.shape)  #(294,5,5)
print(x2_test.shape)  #(127,5,5)

# 데이터 전처리
# StandardScaler

# 3차원 -> 2차원
x1_train = np.reshape(x1_train,
            (x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2]))
x1_test = np.reshape(x1_test,
            (x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2]))
x2_train = np.reshape(x2_train,
            (x2_train.shape[0], x2_train.shape[1]*x2_train.shape[2]))
x2_test = np.reshape(x2_test,
            (x2_test.shape[0], x2_test.shape[1]*x2_test.shape[2]))


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)
print(x2_test_scaled)

# 2차원 -> 3차원
x1_train_scaled = x1_train_scaled.reshape(294,5,5)
x1_test_scaled = x1_test_scaled.reshape(127,5,5)
x2_train_scaled = x2_train_scaled.reshape(294,5,5)
x2_test_scaled = x2_test_scaled.reshape(127,5,5)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, BatchNormalization, Input

input1 = Input(shape=(5,5))
dense1 = LSTM(60)(input1)
dense1 = Dense(40)(dense1)
dense1 = Dense(20)(dense1)
output1 = Dense(20)(dense1)

input2 = Input(shape=(5,5))
dense2 = LSTM(35)(input2)
dense2 = Dense(25)(dense2)
output2 = Dense(15)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output)


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit([x1_train_scaled, x2_train_scaled], 
          y1_train, epochs=100, batch_size=20, 
          validation_split=0.25)

loss, mae = model.evaluate([x1_test_scaled, x2_test_scaled], 
                           y1_test, batch_size=1)
print(loss, mae)

y_pred = model.predict([x1_test_scaled, x2_test_scaled])
# print(y_pred)

for i in range(5):
    print('종가: ', y1_test[i], '/ 예측가: ', y_pred[i])

