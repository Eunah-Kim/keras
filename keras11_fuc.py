# 1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101,201), range(301, 401)])
y = np.array([range(101, 201)])

print(x.shape) 
print(y.shape) 

x = np.transpose(x)
y = np.transpose(y)

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
from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

# input1 = Input(shape=(3,))
# x = Dense(5)(input1)
# x = Dense(2)(x)
# x = Dense(3)(x)
# output1 = Dense(1)(x)

# 마지막에 함수형 모델 정의(input, output layer 명시)
model = Model(inputs = input1, outputs = output1)

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])

model.fit(x_train, y_train, epochs = 100, batch_size=1, 
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
