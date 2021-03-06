##### 선형회귀 #####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#보스턴 주택 데이터셋
from sklearn import datasets
housing = datasets.load_boston()

#key 값 확인
housing.keys()

#pandas dataframe으로 변환
data = pd.DataFrame(housing['data'], columns=housing['feature_names'])
target = pd.DataFrame(housing['target'], columns=['Target'])

#데이터 셋 크기
print(data.shape)
print(target.shape)

#데이터 프레임 결합 - data 와 target
df = pd.concat([data, target], axis=1)
df.head(3)

#기본정보 확인 
df.info

#결측값 확인
df.isnull().sum()

#상관관계 분석
df_corr = df.corr()

plt.figure(figsize=(10,10))
sns.set(font_scale=0.8)
sns.heatmap(df_corr, annot=True, cbar=False) #히트맵
plt.show()

#target 변수와 상관관계가 높은 순으로 출력
corr_order = df.corr().loc[:'LSTAT','Target'].abs().sort_values(ascending=False)
corr_order

#시각화로 분석할 피터 선택 추출
plot_cols = ['Target','LSTAT','RM','PTRATIO','INDUS']
plot_df = df.loc[:,plot_cols]
plot_df.head()

#regplot으로 선형회귀선 표시
plt.figure(figsize=(10,10))
for idx, col in enumerate(plot_cols[1:]):
  ax1 = plt.subplot(2,2,idx+1)
  sns.regplot(x=col, y=plot_cols[0], data=plot_df, ax=ax1)
plt.show()

#피처 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df_scaled = df.iloc[:,:-1] #마지막열인 target을 제외하고 스케일러를 돌림
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

#스케일링 변환된 값을 데이터프레임에 반영
df.iloc[:,:-1] = df_scaled[:,:]
df.head()

#학습데이터와 테스트데이터 분할
from sklearn.model_selection import train_test_split
x_data = df.loc[:,['LSTAT','RM']]
y_data = df.loc[:,'Target']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=12)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#선형회귀모형
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

print('회귀계수(기울기):', np.round(lr.coef_, 1))
print('상수항(절편):', np.round(lr.intercept_, 1))

#예측값 저정
y_test_pred = lr.predict(x_test)

#예측값과 실제값의 분포
plt.figure(figsize=(10,5)) #표 크기 지정
plt.scatter(x_test['LSTAT'], y_test, label='y_test') #파란점, 실제값
plt.scatter(x_test['LSTAT'], y_test_pred, c='r', label='y_pred') #빨간점, 예측값
plt.legend(loc='best')
plt.show()

#성능 평가 - MSE 사용
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(x_train)

train_mse = mean_squared_error(y_train, y_train_pred) #훈련 데이터의 평가 점수
print('Train MSE:%.4f' % train_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print('Test MSE:%.4f' % test_mse)

##### 로버스트 회귀 #####

df = pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRARIO','B','LSTAT','MEDV']
df.head()

#선형회귀

from sklearn.linear_model import LinearRegression
slr = LinearRegression(fit_intercept=True)
X = df[['RM']].values
y = df[['MEDV']].values
slr.fit(X, y)
print('기울기: %.3f' % slr.coef_[0])
print('절편: %.3f' % slr.intercept_)

import matplotlib.pyplot as plt
plt.scatter(X, y, c='steelblue', edgecolors='white', s=70)
plt.plot(X, slr.predict(X), color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

plt.show()

#RANSAC
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, loss='absolute_loss', residual_threshold=5.0, random_state=0)

#선형 회귀분석 수행
ransac.fit(X, y)

#데이터들이 오차 범위 내에 있는지 여부를 저장한 배열을 가져와서 저장
inlier_mask = ransac.inlier_mask_

#배열의 값을 반대로 만들어서 저장
outlier_mask = np.logical_not(inlier_mask)

#그래프를 그릴 범위 설정
line_X = np.arange(3, 10, 1)

#그릴 범위에 해당하는 데이터의 예측값 가져오기
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

#실제 데이터를 산점도로 표현
plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')

#예측 모델을 선 그래프로 표현
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

print('기울기: %.3f' % ransac.estimator_.coef_[0])
print('절편: %.3f' % ransac.estimator_.intercept_)
plt.show()

##### 로지스틱 회귀 #####

x = np.linspace(-10, 10)
print('input\n', x)

def sigmoid(x):
  return 1 / (1+np.exp(-x))
 
output = sigmoid(x)
print('output\n', output)

threshold = 0.5
print(len(output))

clfied_1 = output[output>=0.5]
clfied_0 = output[output<0.5]

plt.plot(x, output, color='r', linewidth=3, zorder=0)

plt.scatter(x, output, c=[clfied_1, clfied_0])

plt.axhline(0.5, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#데이터 로드
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer

#데이터 분포 확인
#전처리 - 정규화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)
print('압축 전의 shape \n', data_scaled.shape)

#전처리 - feature 압축
#feature가 여러개라 단일 변수에 대한 분포를 확인하기 힘들기 때문에 여러 변수들의 특성을 고려해 압축한 하나의 피처로 만듦
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(data_scaled)
pca_data = pca.transform(data_scaled)
print('압축 후의 shape \n', pca_data.shape)

#시각화
plt.scatter(pca_data, cancer.target)
plt.xlabel('Pressured Features as One')
plt.ylabel('Distribution of Breast Cancer')
plt.show()

#로지스틱 회귀를 이용하여 학습 및 예측 수행
from sklearn.linear_model import LogisticRegression

#전처리 - train/test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(pca_data, cancer.target, test_size=0.3, random_state=0)

#모델 학습 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_preds = lr_clf.predict(X_test)

#predict_proba로 모델이 예측한 확률값 반환
lr_proba = lr_clf.predict_proba(X_test)

#주어진 데이터의 원래 분포
plt.scatter(pca_data, cancer.target)

#주황색: logistic regression이 분류한 값
plt.scatter(X_test, lr_preds)

#초록색: logistic regression이 예측한 확률 
plt.scatter(X_test, lr_proba[:,1])

plt.show()

#accuracy와 roc_auc 측정
from sklearn.metrics import accuracy_score, roc_auc_score

print('accuracy: {:0.3f}'.format(accuracy_score(y_test, lr_preds)))
print('roc_auc: {:0.3f}'.format(roc_auc_score(y_test, lr_preds)))

##### 단일 입력 로지스틱 회귀 #####

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

model = Sequential()
#입력 1개를 받아 출력 1개를 리턴하는 선형 회귀 레이어를 생성
model.add(Dense(input_dim = 1, units = 1))
#선형 회귀의 출력값을 시그모이드에 연결
model.add(Activation('sigmoid'))
#크로스 엔트로피를 비용함수로 설정해 경사하강법으로 학습
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['binary_accuracy'])

#데이터 생성
X = np.array([-2,-1.5,-1,1.25,1.62,2])
Y = np.array([0,0,0,1,1,1])

#모델 학습 - 300번의 반복 학습 통해 최적의 w와 b찾기
model.fit(X, Y, epochs=300, verbose=0)

model.predict([-2,-1.5,-1,1.25,1.62,2])

#시그모이드 특성상 왼쪽 극한의 값은 0, 오른쪽 극한의 값은 1로 수렴
model.predict([-1000,1000])

model.summary()

model.layers[0].weights

model.layers[0].get_weights()

##### 소프트맥스 회귀 #####

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

#손으로 쓴 숫자 MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('train data(count, row, column : ' + str(X_train.shape))
print('test data(count, row, column : ' + str(X_test.shape))

X_train[0]

#X_train[0]을 통해 데이터 샘플을 보면, 
#각 픽셀이 0부터 255까지의 값을 가지고 있다는 것을 알 수 있음.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('train target (count) : ' + str(y_train.shape))
print('test target (count) : ' + str(y_test.shape))

print('sample from train : ' + str(y_train[0]))
print('sample from test : ' + str(y_test[0]))

#행과 열 구분 없이 단순히 784(28*28) 길이의 배열로 데이터 단순화
input_dim = 784
X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#One-hot encoding 
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(y_train[0])

#소프트맥스 구현
model = Sequential()
model.add(Dense(input_dim = input_dim, units=10, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=2048, epochs=100, verbose=0)

model.predict(X_train)[0]

score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])

model.summary()

model.layers[0].weights

