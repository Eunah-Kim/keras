from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(28*28,), name = 'input')
    x = Dense(512, activation='relu', name='hidden')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden')(inputs)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

# keras에서 scikit_learn 사용하기
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# Scikit_learn을 쓸 수 있는 kears 함수형 모델 만들기
model = KerasClassifier(build_fn=build_network, verbose=1)
# KerasClassifier(build_fn=함수형모델, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
# n_iter = 작업횟수, n_jobs = cpu 병렬처리 개수, cv= cross_validation
# n_jobs = -1 ;  있는거 다 쓰겠다.
search.fit(x_train, y_train)

print(search.best_params_)
# {'optimizer': 'adam', 'keep_prob': 0.1, 'batch_size': 20}