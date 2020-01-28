# 1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101,201), range(301, 401)])
y1 = np.array([range(101, 201)])

x2 = np.array([range(1001, 1101), range(1101,1201), range(1301, 1401)])
# y2 = np.array([range(101, 201)])


x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.4, shuffle=False, random_state=66)
# x_val, x_test, y_val, y_test = train_test_split(
#     x_test, y_test, test_size=0.5, shuffle=False, random_state=66)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, test_size=0.4, shuffle=False, random_state=66)
x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(
    x1_test, x2_test, y1_test, test_size=0.5, shuffle=False, random_state=66)

print(x1_train.shape)
print(x1_val.shape)
print(x1_test.shape)
print(x2_train.shape)
print(x2_val.shape)
print(x2_test.shape)
print(y1_train.shape)
print(y1_val.shape)
print(y1_test.shape)


# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,))
dense21 = Dense(7)(input2)
dense23 = Dense(4)(dense21)
output2 = Dense(5)(dense23)
# 앙상블 시 dense의 수, node의 수가 달라도 상관없음

from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

# 마지막에 함수형 모델 정의(input, output layer 명시, 여러 개일 경우 list로 연결)
model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])

# 2개 이상의 input >> list로 연결
model.fit([x1_train, x2_train], y1_train, epochs = 100, batch_size=1, 
          validation_data=([x1_val, x2_val], y1_val))

# 4. 평가 예측
loss, mae = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('mae: ', mae)

x1_prd = np.array([[201,202,203], [204,205,206], [207,208,209]])
x2_prd = np.array([[211,212,213], [214,215,216], [217,218,219]])
x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)
aaa = model.predict([x1_prd, x2_prd], batch_size=1)
print(aaa)

#RMSE 구하기
y_predict = model.predict([x1_test, x2_test], batch_size=1)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y_predict))
print("RMSE: ", RMSE(y1_test, y_predict))


#R2 구하기
from sklearn.metrics import r2_score
r2_pred = r2_score(y1_test, y_predict)
print("r2: ", r2_pred)
