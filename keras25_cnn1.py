from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2,2), padding='valid',
                input_shape=(5, 5, 1)))
# Conv2d(output 수, kernel, strid, padding, input_shape)
# kernel = 자를 픽셀 크기
# padding='valid': 패딩 적용X (4,4,7)
# padding='same' : 기존 이미지와 같은 크기로 넘겨주기 위해 
#                  기존 이미지 주변을 0으로 채움 (5,5,7)
# padding을 하는 이유 : 가장자리에 있는 데이터의 손실을 막기 위해
model.add(MaxPooling2D(2,2))
# (2, 2)필터를 적용해 가장 큰 픽셀 값을 반환
# strids=2(필터크기)처럼 적용됨
model.add(Flatten())
# 3차원 이미지를 펴서 1차원으로 전달
model.add(Dense(1))
# Dense로 연결
model.summary()


# strides / 
model = Sequential()
model.add(Conv2D(7, (2,2), strides=(2,2), padding='valid',
                input_shape=(5, 5, 1)))
# strides(1,1)==strides=1 ; default
# strides(2,2)==strides=2 적용시, 소수점 아래 버림
model.add(Conv2D(100, (2,2), padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(1))

model.summary()
