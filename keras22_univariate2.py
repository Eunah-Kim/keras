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
    
# DNN 모델 구성하기
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(50, input_shape=(3,)))
model.add(Dense(40))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=100, batch_size=1)

loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_pred = array([[90, 100, 110]])
x_pred = model.predict(x_pred)
print(x_pred)
