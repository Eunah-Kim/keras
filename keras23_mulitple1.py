from numpy import array

def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        # 다항식 연산
        seq_x, seq_y = sequence[i:end_ix,:-1], sequence[end_ix-1,-1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape)
print(in_seq2.shape)
print(out_seq.shape)

in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

print(in_seq1.shape)
print(in_seq2.shape)
print(out_seq.shape)

from numpy import hstack
dataset = hstack((in_seq1, in_seq2, out_seq))
n_steps = 3
# print(dataset)

x, y = split_sequence2(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])
    
print(x.shape)
print(y.shape)

'''
# LSTM 모델 구성하기
x = x.reshape(8, 6, 1)

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(60, activation = 'relu', input_shape=(6,1)))
model.add(Dense(100, activation='relu'))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=400, batch_size=1)

loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_pred = array([[90, 50],
                [100,105],
                [110,115]])
# print(x_pred.shape)
x_pred = x_pred.reshape(1,6, 1)
x_pred = model.predict(x_pred)
print(x_pred)
'''
# DNN 모델 만들기
x = x.reshape(8, 6)

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape=(6,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x, y, epochs=100, batch_size=1)

loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

x_pred = array([[90, 50],
                [100,105],
                [110,115]])
# print(x_pred.shape)
x_pred = x_pred.reshape(1,6)
x_pred = model.predict(x_pred)
print(x_pred)

