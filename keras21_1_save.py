
# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=3))
model.add(Dense(64, input_shape=(3,)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()

model.save('./save/savetest01.h5')
print('저장 완료')
