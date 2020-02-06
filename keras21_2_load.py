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

# 2. 모델 불러오기
from keras.models import load_model, Model
from keras.layers import Dense

# 모델을 불러오고 추가로 Dense를 연결 할 시 dense layer명이 중복됨
# 이를 해결하기 위해 Dense층 내 이름(name)을 설정해줌

model = load_model("./save/savetest01.h5")

# Sequential로 연결하기
model.add(Dense(5, name='1'))
model.add(Dense(5, name='2'))
model.add(Dense(1, name='3'))


# # 함수형으로 연결하기
# x = model.output
# x = Dense(5, name='1')(x)
# x = Dense(5, name='2')(x)
# x = Dense(1, name='3')(x)
# model = Model(inputs=model.input, outputs=x)

# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])

model.fit(x_train, y_train, epochs = 1000, batch_size=20, 
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
