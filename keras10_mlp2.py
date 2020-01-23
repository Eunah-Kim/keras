# 1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101,201), range(301, 401)])
y = np.array([range(101, 201)])
# y2 = np.array(range(101, 201))

print(x.shape)  # (3, 100)
print(y.shape)  # (1, 100)
# print(y2.shape) # (100,)

x = x.transpose()  # np.transpose(x)
y = y.transpose()  # np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, shuffle=False, random_state=66)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, shuffle=False, random_state=66)

# print(x_train)
# print(x_val)
# print(x_test)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=3))
  # input_dim = 2 ; 열이 2개 ; 벡터가 2개
model.add(Dense(64, input_shape=(3,)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

'''
# 안 좋은 모델 만들기
model.add(Dense(3, input_shape=(3,)))
model.add(Dense(100))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
'''

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])

model.fit(x_train, y_train, epochs = 100, batch_size=20, 
          validation_data=(x_val, y_val))

# 4. 평가 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('mae: ', mae)

x_prd = np.array([[201,202,203], [204,205,206], [207,208,209]])
x_prd = np.transpose(x_prd)
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
