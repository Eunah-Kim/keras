import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC


warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, train_size=0.8, shuffle=True)

# 그리드 서치에서 사용할 매개 변수 ---(*1)
parameter = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.001,0.0001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001,0.0001]}
]

#그리드 서치 ---(*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
model =GridSearchCV(SVC(), parameter, cv=kfold_cv)
# 
model.fit(x_train, y_train)
print("최적의 매개 변수 - ", model.best_estimator_)


# 최적의 매개 변수로 평가하기 ---(*3)
y_pred = model.predict(x_test)
print("최종 정답률: ", accuracy_score(y_test, y_pred))