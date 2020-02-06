import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators


warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# classofoer 알고리즘 모두 추출하기---(*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

kfold_cv = KFold(n_splits=5, shuffle=True)
for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기 ---(*2)
    clf = algorithm()
    
    if hasattr(clf, "score"):
        # hasattr(clf, "score"); score가 있는 모델 사용하겠다
        scores = cross_val_score(clf, x, y, cv=kfold_cv)
        # cross_val_score 자체에 fit이 포함됨
        # cv = kfold_cv에 정의된 만큼 cross_val하겠다 (5번 split)
        avg_score = 0
        for i in scores:
            avg_score += i
        avg_score = avg_score/len(scores)
        print(name, "의 정답률 = ")
        print(avg_score)
        # 5개의 score를 확인할 수 있음
        # 어떤 모델이 좋은지 확인할 수 있음