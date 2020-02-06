from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, LSTM
from keras.callbacks import EarlyStopping
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  #(60000,28,28)
print(y_train.shape)  #(60000,)

# x_train, x_test 전처리
x_train = x_train.reshape(x_train.shape[0],28*28).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28*28).astype('float32')/255

# 분류문제이기 때문에 y데이터를 categorical로 변경
#  > one-hot encoding으로 10개의 값으로 변경됨
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)


model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape=(784,)))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 다중분류 loss='categorical_crossentropy', metrics=[accuracy']
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'])

early_stopping = EarlyStopping(monitor='loss', patience=20)

hist = model.fit(x_train, y_train, validation_split=0.2,
                epochs=100, batch_size=100, verbose=1,
                callbacks = [early_stopping])


print(hist.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])


import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()


acc = model.evaluate(x_test, y_test)

print(acc)
# [0.2109908364421206, 0.9743000268936157]
