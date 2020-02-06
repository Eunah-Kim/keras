from numpy import array

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

dataset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_steps = 3

x, y = split_sequence(dataset, n_steps)

# print(x)
# print(y)

for i in range(len(x)):
    print(x[i], y[i])
    
x = x.reshape(x.shape[0], x.shape[1], 1)
    
# DNN 모델 구성하기
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(60, activation = 'relu', input_shape=(3,1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=200, batch_size=1)

loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_pred = array([[90, 100, 110]])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
x_pred = model.predict(x_pred)
print(x_pred)
