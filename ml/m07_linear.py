from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()

x = np.array(boston.data)
y = np.array(boston.target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state = 1)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

lr = LinearRegression()
lr.fit(x_train, y_train)
print("LinearRegression의 score: ", lr.score(x_test, y_test))

ridge = Ridge()
ridge.fit(x_train, y_train)
print("Ridge의 score: ", ridge.score(x_test, y_test))

lasso = Lasso()
lasso.fit(x_train, y_train)
print("Lasso의 score: ", lasso.score(x_test, y_test))