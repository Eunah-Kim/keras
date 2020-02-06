from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)  #(50000, 32, 32, 3)
print(y_train.shape)  #(50000, 1)


# x_train, x_test 전처리
# 데이터 타입 : float / 최대값인 255로 나눠서 0~1 사이의 값으로 바꿈
x_train = x_train.reshape(x_train.shape[0],32*32*3).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],32*32*3).astype('float32')/255

# 분류문제이기 때문에 y데이터를 categorical로 변경
#  > one-hot encoding으로 10개의 값으로 변경됨
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)


model = Sequential()
model.add(Dense(248, activation = 'relu', input_shape=(32*32*3,)))
model.add(Dense(248, activation = 'relu'))
model.add(Dense(126, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 다중분류 loss='categorical_crossentropy', metrics=[accuracy']
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=20)
model.fit(x_train, y_train, validation_split=0.2,
          epochs=100, batch_size=500, verbose=1,
          callbacks = [early_stopping])

acc = model.evaluate(x_test, y_test)

print(acc)
# [1.5447493600845337, 0.4916999936103821]
