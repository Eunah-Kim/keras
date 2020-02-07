from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, Y_train)

print("훈련 세트 정확도: {:.3f}".format(xgb.score(X_train, Y_train)))
print("테스트 세트 정확도: {:.3f}".format(xgb.score(X_test, Y_test)))

print("특성 중요도 \n", xgb.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
    
plot_feature_importances_cancer(xgb)
plt.show()