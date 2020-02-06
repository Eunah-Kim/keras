import numpy as np
import pandas as pd

df1 = pd.read_csv('./samsung/samsung.csv', index_col=0,
                  header = 0, encoding='cp949', sep=',')
# index_col : index로 지정할 열
# header : 칼럼명 위치된 행
# sep=',' : 행 사이를 ,로 나눔

# print(df1)
# print(df1.shape)

df2 = pd.read_csv('./samsung/kospi200.csv', index_col=0,
                  header = 0, encoding='cp949', sep=',')

# print(df2)
# print(df2.shape)

# samsung
for i in range(len(df1.index)):
    for j in range(len(df1.iloc[i])):
        df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))

# kospi200의 거래량.
for i in range(len(df2.index)):
    df2.iloc[i,4] = int(df2.iloc[i,4].replace(',', ''))
    
# loc vs iloc : loc; 컬럼명, iloc;컬럼위치

df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])

# print(df2)

# pandas에서 numpy로 바꾸기
df1 = df1.values
df2 = df2.values

print(type(df1), type(df2))
print(df1.shape, df2.shape)

# numpy 파일 저장하기
np.save('./samsung/data/samsung.npy', arr=df1)
np.save('./samsung/data/kospi200.npy', arr=df2)