from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# 기존의 열 2개만 남기는 것이 아닌 두개의 열로 압축함.
# 실제 column이 아닌 것이 핵심!!!!!!
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("원본 데이터 형태: ", X_scaled.shape)
print("축소된 데이터 형태: ", X_pca.shape)