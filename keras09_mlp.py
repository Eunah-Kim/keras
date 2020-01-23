# 1. 데이터
import numpy as np
x = np.array([range(1, 11), range(11,21)])
y = np.array([range(1, 11), range(11,21)])

print(x.shape)  # (2, 10)
print(y.shape)  # (2, 10)

# x = x.reshape(10, 2)
# y = y.reshape(10, 2)

x = x.transpose()
y = y.transpose()

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, shuffle=False, random_state=66)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, shuffle=False, random_state=66)


print(x_train)
print(x_val)
print(x_test)


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=2))
model.add(Dense(5, input_shape=(2,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(2))

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])
# metrics : 실행결과를 보여줌
  # 회귀 > mse/mae(Rmse/Rmae)... , 분류 > acc

model.fit(x_train, y_train, epochs = 100, batch_size=1, 
          validation_data=(x_val, y_val))

# 4. 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('mae: ', mae)
'''
x_prd = np.array([101,102,103])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

#RMSE 구하기
y_predict = model.predict(x_test, batch_size=1)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


#R2 구하기
from sklearn.metrics import r2_score
r2_pred = r2_score(y_test, y_predict)
print("r2: ", r2_pred)
'''