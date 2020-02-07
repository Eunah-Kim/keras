import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]


x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, train_size=0.8, shuffle=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Pipeline 모델 정의
# preprocessing이 포함된 모델임
# preprocessing;MinMax, model;SVC 을 한번에 정의
pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])

pipe.fit(x_train, y_train)

print("테스트 점수: ", pipe.score(x_test, y_test))
