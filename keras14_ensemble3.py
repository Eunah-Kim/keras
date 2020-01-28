# 1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101,201), range(301, 401)])

# y1 = np.array([range(101, 201)])

y1 = np.array([range(1, 101), range(101,201), range(301, 401)])
y2 = np.array([range(1001, 1101), range(1101,1201), range(1301, 1401)])
y3 = np.array([range(1, 101), range(101,201), range(301, 401)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.4, shuffle=False, random_state=66)
# x_val, x_test, y_val, y_test = train_test_split(
#     x_test, y_test, test_size=0.5, shuffle=False, random_state=66)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1, y1, y2, y3, test_size=0.4, shuffle=False, random_state=66)
x1_val, x1_test, y1_val, y1_test, y2_val, y2_test, y3_val, y3_test = train_test_split(
    x1_test, y1_test, y2_test, y3_test, test_size=0.5, shuffle=False, random_state=66)
# shuffle = True로 설정했을 시, x와 y의 순서쌍은 변하지 않는다.

print(y3_train.shape)
print(y3_val.shape)
print(y3_test.shape)


# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(3)(dense1)
output1 = Dense(4)(dense2)

# merege 필요 없음

output_1 = Dense(2)(output1)
output_1 = Dense(3)(output_1)   # 1번째 output 모델
   
output_2 = Dense(4)(output1)
output_2 = Dense(4)(output_2)
output_2 = Dense(3)(output_2)   # 2번째 output 모델

output_3 = Dense(5)(output1)
output_3 = Dense(3)(output_3)   # 3번째 output 모델
# output 모델 수 = 결과 y의 수(y1, y2, y3 ; 3개)
# output 모델의 마지막 노드수 = y의 column 수

# 마지막에 함수형 모델 정의(input, output layer 명시, 여러 개일 경우 list로 연결)
model = Model(inputs = input1, outputs = [output_1, output_2, output_3])

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mae'])

# 2개 이상의 input >> list로 연결
model.fit(x1_train, [y1_train, y2_train, y3_train], epochs = 100, batch_size=1, 
          validation_data=(x1_val, [y1_val, y2_val, y3_val]))


# 4. 평가 예측
aaa = model.evaluate(x1_test, [y1_test, y2_test, y3_test], batch_size=1)
# loss, mse 반환값이 여러 개 > 리스트로 나타내기
print('aaa: ', aaa)

x1_prd = np.array([[201,202,203], [204,205,206], [207,208,209]])
x1_prd = np.transpose(x1_prd)
predict_bbb = model.predict(x1_prd, batch_size=1)
print(predict_bbb)

#RMSE 구하기
y_predict = model.predict(x1_test, batch_size=1)
# print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse1 = RMSE(y1_test, y_predict[0])
rmse2 = RMSE(y2_test, y_predict[1])
rmse3 = RMSE(y3_test, y_predict[2])

print("RMSE1: ", rmse1, "RMSE2: ", rmse2, "RMSE3: ", rmse3)

rmse = (rmse1+rmse2+rmse3)/3
print("RMSE: ", rmse)

#RMSE 구하기2
y1_predict, y2_predict, y3_predict = model.predict(x1_test, batch_size=1)

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
RMSE = (RMSE1+RMSE2+RMSE3)/3
print("RMSE: ", RMSE)


#R2 구하기
from sklearn.metrics import r2_score
r2_pred1 = r2_score(y1_test, y_predict[0])
r2_pred2 = r2_score(y2_test, y_predict[1])
r2_pred3 = r2_score(y3_test, y_predict[2])
r2_pred = (r2_pred1 + r2_pred2 + r2_pred3)/3
print("r2: ", r2_pred)
