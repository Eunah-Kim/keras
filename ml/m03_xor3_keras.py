from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. 데이터
x_train = np.array([[0,0],[1,0],[0,1],[1,1]])
y_train = np.array([0,1,1,0])

# 2. 모델
# model = LinearSVC()
# model = KNeighborsClassifier(n_neighbors=1)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid'))

# 3. 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size=1)

# 4. 평가예측
x_test = np.array([[0,0],[1,0],[0,1],[1,1]])
y_test = np.array([0,1,1,0])
y_predict = model.predict(x_test)
loss, acc_score = model.evaluate(x_test, y_test)

print(x_test, "의 예측결과: ", y_predict)
print("acc = ", acc_score)