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

x, y = split_xy5(samsung, 5, 1)
print(x.shape)
print(y.shape)
print(x[0,:], "\n", y[0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3, shuffle=False)

print(x_train.shape)  #(294,5,5)
print(x_test.shape)  #(127,5,5)


# 데이터 전처리
# StandardScaler

# 3차원 -> 2차원
x_train = np.reshape(x_train,
            (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test,
            (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# x_train = x_train.reshape(294,25)
# x_test = x_test.reshape(127,25)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0,:])

# # 2차원 -> 3차원
# x_train = x_train_scaled.reshape(294,5,5)
# x_test = x_test_scaled.reshape(127,5,5)


from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(25,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train_scaled, y_train, epochs=100, batch_size=20, 
          validation_split=0.25)

loss, mae = model.evaluate(x_test_scaled, y_test, batch_size=1)
print(loss, mae)

y_pred = model.predict(x_test_scaled)
# print(y_pred)

for i in range(5):
    print('종가: ', y_test[i], '/ 예측가: ', y_pred[i])

