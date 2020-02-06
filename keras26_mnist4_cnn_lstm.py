from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, LSTM, Reshape
from keras.callbacks import EarlyStopping
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  #(60000,28,28)
print(y_train.shape)  #(60000,)

# x_train, x_test 전처리
# 데이터 타입 : float / 최대값인 255로 나눠서 0~1 사이의 값으로 바꿈
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255

# 분류문제이기 때문에 y데이터를 categorical로 변경
#  > one-hot encoding으로 10개의 값으로 변경됨
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)


model = Sequential()
model.add(Conv2D(16, (2,2), strides=(2,2), padding='same',
                 activation = 'relu',
                input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2,2))
# model.add(Conv2D(16, (2,2), activation='relu', padding='same'))
model.add(Reshape((28,28)))
model.add(LSTM(4, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 다중분류 loss='categorical_crossentropy', metrics=[accuracy']
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=20)
model.fit(x_train, y_train, validation_split=0.2,
          epochs=100, batch_size=1000, verbose=1,
          callbacks = [early_stopping])

acc = model.evaluate(x_test, y_test)

print(acc)
# [0.7599361300468445, 0.7519999742507935]
