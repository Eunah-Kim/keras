# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)
# print(y.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim=1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

'''
# 3. 훈련
model.compile(loss='mse', optimizer = 'adam', 
              metrics=['mse'])
# metrics : 실행결과를 보여줌
  # 회귀 > mse/mae(Rmse/Rmae)... , 분류 > acc

model.fit(x, y, epochs = 1000, batch_size=1)

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print('mae: ', mae)

x_prd = np.array([11,12,13])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

bbb = model.predict(x)
print(bbb)

'''

