import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# 1. 데이터 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8',
                        names=['a','b','c','d','y']) #, header=None)

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"]
x = iris_data.loc[:,["a","b","c","d"]]

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# y = y.replace("Iris-setosa",0)
# y = y.replace("Iris-virginica",1)
# y = y.replace("Iris-versicolor",2)

x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, train_size=0.8, shuffle=True )

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train)

# 모델 정의
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

# 모델 학습
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=10)

# 평가하기
y_pred = model.predict(x_test)
print(y_pred.shape)
# y_pred.reshape(30,3,1)
# y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

loss, acc = model.evaluate(x_test, y_test)
print("\n 정답률: ", acc)
