# 1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# x_train = np.array(range(1,61))
# y_train = np.array(range(1,61))
# x_val = np.array(range(61, 81))
# y_val = np.array(range(61, 81))
# x_test = np.array(range(81,101))
# y_test = np.array(range(81,101))

x_train = x[:60]
y_train = y[:60]
x_val = x[60:80]
y_val = y[60:80]
x_test = x[80:]
y_test = y[80:]


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=1))
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])
# metrics : 실행결과를 보여줌
  # 회귀 > mse/mae(Rmse/Rmae)... , 분류 > acc

model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_data=(x_val, y_val))

# 4. 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('mae: ', mae)

# aaa = model.predict(x_test)
# print(aaa)

x_prd = np.array([101,102,103])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

'''
bbb = model.predict(x)
print(bbb)
'''

