import numpy as np
import pandas as pd

df1 = pd.read_csv('./samsung/samsung.csv', index_col=0,
                  header = 0, encoding='cp949', sep=',')
print(df1.info)
# print(df1.shape)

df2 = pd.read_excel('./samsung/삼성전자 0203-0206.xlsx', index_col=0,
                  header = 0, encoding='cp949', sep=',')

print(df2.info)
# print(df2.shape)

# samsung(~2월3일)
for i in range(len(df1.index)):
    for j in range(len(df1.iloc[i])):
        df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))

# samsung 추가 데이터
for i in range(len(df2.index)):
    df2.iloc[i,4] = int(df2.iloc[i,4].replace(',', ''))
    
# loc vs iloc : loc; 컬럼명, iloc;컬럼위치

df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_index(axis=0, ascending=[True])
print(df2)

# df1, df2 합치기
samsung = df1.append(df2)
print(samsung.shape)

# pandas에서 numpy로 바꾸기
samsung = samsung.values

print(type(samsung))
print(samsung.shape)

# numpy 파일 'samsung1.npy'로 저장하기
np.save('./samsung/data/samsung1.npy', arr=samsung)
