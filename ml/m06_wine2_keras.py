import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=';',
                   encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop("quality", axis=1)

'''
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse=False)
y = y.reshape(len(y),1)
onehot.fit(y)
y = onehot.transform(y)
print(y.shape)
'''

x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, train_size=0.8, shuffle=True )

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape) #(3918,11)
print(y_train.shape) #(3918,10)


# 모델 정의
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(11,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 학습
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 평가하기
y_pred = model.predict(x_test)
print(y_pred.shape)
# print(y_pred)

loss, acc = model.evaluate(x_test, y_test)
print("\n 정답률: ", acc)
