import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators


warnings.filterwarnings('ignore')

iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, train_size=0.8, shuffle=True)

warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")
# type_filer="regresser"

print(allAlgorithms)
print(len(allAlgorithms))
print(type(allAlgorithms))

for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    clf = algorithm()
    
    # 학습하고 평가하기
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name, "의 정답률; ", accuracy_score(y_test, y_pred))
    
'''
31
<class 'list'>
AdaBoostClassifier 의 정답률;  1.0
BaggingClassifier 의 정답률;  1.0
BernoulliNB 의 정답률;  0.23333333333333334
CalibratedClassifierCV 의 정답률;  0.9333333333333333
ComplementNB 의 정답률;  0.5333333333333333
DecisionTreeClassifier 의 정답률;  1.0
ExtraTreeClassifier 의 정답률;  0.9333333333333333
ExtraTreesClassifier 의 정답률;  0.9666666666666667
GaussianNB 의 정답률;  0.9666666666666667
GaussianProcessClassifier 의 정답률;  1.0
GradientBoostingClassifier 의 정답률;  1.0
KNeighborsClassifier 의 정답률;  1.0
LabelPropagation 의 정답률;  1.0
LabelSpreading 의 정답률;  1.0
LinearDiscriminantAnalysis 의 정답률;  1.0
LinearSVC 의 정답률;  0.9666666666666667
LogisticRegression 의 정답률;  0.9333333333333333
LogisticRegressionCV 의 정답률;  1.0
MLPClassifier 의 정답률;  0.9333333333333333
MultinomialNB 의 정답률;  0.5333333333333333
NearestCentroid 의 정답률;  0.9666666666666667
NuSVC 의 정답률;  1.0
PassiveAggressiveClassifier 의 정답률;  0.6
Perceptron 의 정답률;  0.5333333333333333
QuadraticDiscriminantAnalysis 의 정답률;  1.0
RadiusNeighborsClassifier 의 정답률;  0.9666666666666667
RandomForestClassifier 의 정답률;  1.0
RidgeClassifier 의 정답률;  0.8333333333333334
RidgeClassifierCV 의 정답률;  0.8333333333333334
SGDClassifier 의 정답률;  0.8666666666666667
SVC 의 정답률;  1.0
'''