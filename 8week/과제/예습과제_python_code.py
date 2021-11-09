#교차검증의 필요성(1)
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
iris = load_iris()
print('iris 데이터 레이블 파악 \n{}'.format(iris.target))

#교차 검증의 필요성(2)
logreg = LogisticRegression()
kfold = KFold(n_splits=3)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print(scores)

#Stratified K-Fold

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

#데이터 생성 과정
make_class = make_classification(n_samples=500, #표본 수
                                n_features=3, # 형성하는 피처의 수
                                n_redundant=0, # 독립 변수간 성형 조합으로 나타내는 성분 수
                                n_informative=2, # 독립 변수 중 종속 변수와 상관 관계있는 성분 수
                                n_classes=3, # 종속 변수 클래스 수
                                n_clusters_per_class=1, # 클래스 당 클러스터 수 - 종속변수 간 산포도 인접 여부
                                random_state=1)
data = pd.DataFrame(make_class[0], columns=range(make_class[0].shape[1]))
data['target'] = make_class[1]
data.head()

#KFold를 사용할 시 결과를 살펴보자 (3번째 fold만 확인해보자)
kfold = KFold(n_splits=3, random_state=15, shuffle=True)
splits = kfold.split(data, data['target'])
print(f'타겟 데이터 비율\n{data["target"].value_counts() / len(data)}\n\n')
for n, (train_index, test_index) in enumerate(splits):
      if n == 2:
              print(f'{n+1} 번째 fold 학습 데이터: {np.round(len(train_index) / (len(train_index)+len(test_index)),2)}'+
                   f'\t테스트 데이터 비율: {np.round(len(test_index) / (len(train_index)+len(test_index)),2)}\n학습데이터에 있는 타겟 비율\n'+ 
                   f'{data.iloc[test_index,3].value_counts() / len(data.iloc[test_index,3])}\n테스트 데이터에 있는 타겟 비율\n'+
                   f'{data.iloc[train_index,3].value_counts() / len(data.iloc[train_index, 3])}\n\n')

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
splits = kfold.split(data, data['target'])
print(f'타겟 데이터 비율\n{data["target"].value_counts() / len(data)}\n\n')
#3번째 fold만 확인
for n, (train_index, test_index) in enumerate(splits):
      if n==2:
              print(f'{n+1}번째 fold 학습 데이터: {np.round(len(train_index) / (len(train_index)+len(test_index)),2)}'+
                   f'\t테스트 데이터 비율: {np.round(len(test_index) / (len(train_index)+len(test_index)),2)}/n학습데이터에 있는 타겟 비율\n'+
                   f'{data.iloc[test_index,3].value_counts() / len(data.iloc[test_index,3])}\n테스트 데이터에 있는 타겟 비율\n'+
                   f'{data.iloc[train_index,3].value_counts() / len(data.iloc[train_index,3])}\n\n')

#Cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()
dt = DecisionTreeClassifier(random_state=42)

data = iris_data.data
label = iris_data.target

#성능 지표는 정확도(accuracy), 교차 검증 세트는 3개
scores = cross_val_score(dt, data, label, scoring='accuracy', cv=3)
print('교차 검증별 정확도: ', np.round(scores,2))
print('평균 검증 정확도: ', np.round(np.mean(scores),2))

#GridSearch

import sklearn.datasets as data
import pandas as pd
#breast_cancer 데이터 셋
x = data.load_breast_cancer()
cancer = pd.DataFrame(data=x.data, columns=x.feature_names)
cancer['target'] = x.target
#양성과 음성이 타겟인 data

#GridSearchcv() 실습
#의사 결정 나무로 진행
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#학습 데이터와 테스트 데이터 분할
X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

#의사결정나무를 이용한 분류 모델
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

#학습 후 X_train에 대해 predict하고 예측값 저장
y_pred = dt_clf.predict(X_test)

from sklearn.model_selection import GridSearchCV

#매번 달라지지 않도록 random_state 지정
dt_clf = DecisionTreeClassifier(random_state=10)

#의사결정 나무의 하이퍼 파라미터를 딕셔너리 형태로 정의
parameters = {'max_depth' : [3,5,7],
             'min_samples_split' : [3,5]}

grid_dt = GridSearchCV(dt_clf, #estimator 객체,
                      param_grid=parameters,
                      cv=5,
                      refit=True)
#refit을 통해 하이퍼 파라미터 조합 중 최적의 하이퍼 파라미터를 찾아서 재학습 시킨다. 
#하지만 default가 true이므로 생략 가능

grid_dt.fit(X_train, y_train)

#어떤 값들이 저장되어 있는지 확인해보자
grid_dt.cv_results_.keys()

result = pd.DataFrame(grid_dt.cv_results_)
result[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score',
       'split1_test_score']]
#split0, 1의 경우 각 폴드 세트에서 테스트한 성능의 수치
#중요한 부분은 mean_test_score과 rank_test_score

#최적의 하이퍼 파라미터 조합만 확인
estimator = grid_dt.best_estimator_
pred = estimator.predict(X_test)

#분류에 대한 평가 지표인 정확도를 사용하자
from sklearn.metrics import accuracy_score
print('테스트 셋 정확도: ', accuracy_score(y_test, pred))

#RandomSearchCV

import sklearn.datasets as data
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms

#breast_cancer 데이터 셋
x = data.load_breast_cancer()
cancer = pd.DataFrame(data=x.data, columns=x.feature_names)
cancer['target'] = x.target
#트레인 데이터와 테스트 데이터 분할
X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, 
                                                      test_size=0.3,
                                                      random_state=100)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

max_depths = range(1,5,1)
min_samples_split = range(2,12,1)
min_samples_leaf = range(1,100,2)
param_grid = {
    'max_depth': max_depths,
    'min_samples_split': min_samples_split,
    'min_samples_leaf' : min_samples_leaf
}

import time
start = time.time()
random_dtc = RandomizedSearchCV(DecisionTreeClassifier(), param_grid, random_state=1, n_iter=100, cv=5, n_jobs=-1)
random_dtc.fit(X_train_std, y_train)
print('It takes %s minutes' % ((time.time() - start) / 60))
print('랜덤 서치를 통해서 찾은 최적 하이퍼 파라미터 조합:\n', random_dtc.best_params_)

!pip3 install bayesian-optimization

from bayes_opt import BayesianOptimization
import numpy as np

#블랙박스 함수에 대한 정의
def target(x):
    return np.exp(-(x-5)**2)+np.exp(-(2*x-2)**2)+4/(x-3)**2

#x라는 하이퍼 파라미터를 설정하여 target function의 최적화 된 점을 찾는다고 생각하자
bayes_optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=0)
bayes_optimizer.maximize(init_points=2, n_iter=14, acq='ei')

#실습(1)

import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv('penguins.txt')

data.head()

print('데이터 크기', data.shape)
print('데이터 내용', data.columns)

data.info()

print('펭귄 종류별 수')
data['species'].value_counts().plot(kind='bar')
data['species'].value_counts()

#데이터 결츨치 확인
data.isnull().sum()

#결측치 채우기
col_missing = ['bill_depth_mm','bill_length_mm','flipper_length_mm','body_mass_g']
for column in col_missing:
    data[column].fillna(data[column].median(), inplace=True)

data['sex'] = data['sex'].fillna('MALE')

#EDA
sns.pairplot(data, hue='species', height=4)
plt.show()

#남자는 1여자는 0으로 변환
lb_sex = LabelEncoder()
data['sex'] = lb_sex.fit_transform(data['sex'])
data['sex'][:5]

lb_island = LabelEncoder()
data['island'] = lb_island.fit_transform(data['island'])

lb_species = LabelEncoder()
data['species'] = lb_species.fit_transform(data['species'])
print('종 인코딩:')
for i, j in enumerate(lb_species.classes_):
    print(j, '->', i)

sns.boxplot(x='species', y='bill_length_mm', data=data)
plt.show()
#bill_lengh_mm을 사용하면 Adelie종 구별할 수 있을 것이다.

sns.violinplot(x='species', y='bill_depth_mm', data=data)
plt.show()
#bill_depth_mm을 사용하면 Gentoo를 구별할 수 있을 것이다.

#bill_length, bill_depth feature를 한번에 scatterplot으로 확인
sns.FacetGrid(data, hue='species', height=8)\
   .map(plt.scatter, 'bill_length_mm', 'bill_depth_mm')\
   .add_legend()

sns.boxplot(x='species', y='flipper_length_mm', data=data)
plt.show()
#flipper_length_mm도 Gentoo종을 파악하는 데에 중요한 feature

sns.violinplot(x='species', y='body_mass_g', data=data)
plt.show()
#body_mass_g도 마찬가지

sns.FacetGrid(data, hue='species', height=8)\
   .map(plt.scatter, 'body_mass_g', 'flipper_length_mm')\
   .add_legend()
#두 feature로 Gentoo종 구별 가능

#전처리_Feture

#피쳐와 타겟의 구분 및 학습, 테스트 데이터 분할
y = data['species']
x = data.drop('species', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

print(x_train.shape)
print(y_train.shape)

#피쳐 스케일링
sc = StandardScaler()
sc.fit(x_test)
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

#GridSearchCV()

svc = SVC(random_state=10)
param_grid = {'C': [0.1, 1, 10, 100],
             'gamma': [1, 0.1, 0.01, 0.001]}

import time
start = time.time()
grid_search = GridSearchCV(svc, param_grid, refit=True)
grid_search.fit(x_train, y_train)
print('It takes %s minutes' % ((time.time() - start) / 60))

print(grid_search.best_params_)

#그리드 서치 성능 평가 
grid_predictions = grid_search.predict(x_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

#RandomSearchCV()

rand_list = {'C': stats.uniform(2, 13),
             'gamma': stats.uniform(0.1, 1)}

import time
start = time.time()
rand_search = RandomizedSearchCV(svc, param_distributions=rand_list,
                                 n_iter=20, n_jobs=4, cv=3, random_state=10,
                                 scoring='accuracy')
rand_search.fit(x_train, y_train)
print('It takes %s minutes' % ((time.time() - start) / 60))

#랜덤 서치 성능 평가
rand_predictions = rand_search.predict(x_test)
print(confusion_matrix(y_test, rand_predictions))
print(classification_report(y_test, rand_predictions))

# Commented out IPython magic to ensure Python compatibility.
#실습(2)

#data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#visualization
import matplotlib.pyplot as plt 
import seaborn as sns
# %matplotlib inline
from pandas.plotting import parallel_coordinates

#preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#grid search
from sklearn.model_selection import GridSearchCV

#random search
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

#evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, dtype='category')
y = y.cat.rename_categories(iris.target_names)
df['species'] = y
df.head(5)

df.describe()

df.groupby('species').size()

#시각화

sns.pairplot(df, hue='species')
plt.show()

sns.distplot(df[df.species!='setosa']['petal length (cm)'], hist=True, rug=True, label='setosa')
sns.distplot(df[df.species=='setosa']['petal length (cm)'], hist=True, rug=True, label='others')
plt.legend()
plt.show()
#petal length 하나의 변수만으로 setosa와 다른 종들 쉽게 분류 가능할 것으로 보인다.

sns.distplot(df[df.species=='virginica']['petal length (cm)'], hist=True, rug=True, label='virginica')
sns.distplot(df[df.species=='versicolor']['petal length (cm)'], hist=True, rug=True, label='versicolor')
plt.legend()
plt.show()
#virginica와 versicolor는 petal length만으로 완전히 분류하기 어려워 보인다.

parallel_coordinates(df, 'species')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size':15}, frameon=True, shadow=True, facecolor='white', edgecolor='black')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=42)

ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns)
print(X_train.mean())
print(X_train.var())
X_test = ss.transform(X_test)

def print_metrics(model, feature, target):
    scores = cross_val_score(model, feature, target, cv=5)
    print('*** Cross val score *** \n   {}'.format(scores))
    print('\n*** Mean Accuracy *** \n   {:.7}'.format(scores.mean()))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
print_metrics(dt, X_train, y_train)

#그리드 서치
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(2,5,1),
          'min_samples_split': range(1,100,5)}

import time
start = time.time()
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
dt_gs.fit(X_train, y_train)
dt_gd_pred = dt_gs.predict(X_test)

print(dt_gs.best_params_)
print('그리드 서치(학습):{0:.4f}'.format(dt_gs.best_score_))
print('그리드 서치(테스트):{0:.4f}'.format(accuracy_score(y_test, dt_gd_pred)))
print('It takes %s minutes' % ((time.time() - start)/60))

#랜덤 서치
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(1,6),
          'min_samples_split': randint(1,100)}

import time
start = time.time()
dt_rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=300, n_jobs=-1, random_state=42)
dt_rs.fit(X_train, y_train)
dt_rs_pred = dt_rs.predict(X_test)

print(dt_rs.best_params_)
print('랜덤 서치(학습):{0:.4f}'.format(dt_rs.best_score_))
print('랜덤 서치(테스트):{0:.4f}'.format(accuracy_score(y_test, dt_rs_pred)))
print('It takes %s minutes' % ((time.time() - start)/60))

#베이지안 최적화
!pip install scikit-optimize

space = {'max_depth': np.arange(1,6,1),
         'min_impurity_decrease': np.arange(0.00005,0.002,0.0001),
         'min_samples_split': np.arange(1,100,2)}

from skopt import BayesSearchCV
BO = BayesSearchCV(DecisionTreeClassifier(random_state=42),
                   search_spaces=space,
                   n_jobs=1,
                   cv=5,
                   n_iter=20,
                   scoring='accuracy',
                   verbose=0,
                   random_state=42)

BO.fit(X_train, y_train)

print('정확도 with 베이지안 최적화(학습):', BO.best_score_)
print(BO.best_params_)
dt_BO_pred = BO.predict(X_test)
print('정확도 with 베이지안 최적화(테스트):{0:.4f}'.format(accuracy_score(y_test, dt_BO_pred)))
