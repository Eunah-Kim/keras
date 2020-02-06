# LinearSVC, KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 모델의 설정
model = KNeighborsClassifier(n_neighbors=7)

# 모델 실행
model.fit(x_train, y_train)

# 결과 출력
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('\n Accuracy: %.4f' %acc)

