import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

titanic = sns.load_dataset('titanic')
titanic.isnull().sum()

sns.heatmap(titanic.isnull(), cbar = False)

sns.heatmap(titanic.isnull(), cbar = True)

!pip install missingno
import missingno as msno

msno.heatmap(titanic)

msno.bar(titanic)

msno.matrix(titanic)

msno.dendrogram(titanic)

titanic_delete = titanic.dropna()
titanic_delete

titanic_delete2 = titanic.dropna(axis = 0, how = 'any') 
titanic_delete2.head()

titanic_delete_ae = titanic.dropna(subset = ['age','embarked'], how = 'any', axis = 0)
titanic_delete_ae

dict = {'First':[72, np.nan, np.nan, 95], 'Second':[np.nan, np.nan, np.nan, np.nan],
       'Third':[65, np.nan, 76, 88], 'Fourth':[np.nan, np.nan, np.nan, 90]}

df = pd.DataFrame(dict)
df

df.dropna(how = 'all')

df.dropna(thresh = 1)

df.dropna(thresh = 2)

df.dropna(thresh = 3)

df.dropna(thresh = 4)

titanic_delete7 = titanic.dropna(thresh = 14)
titanic_delete7

titanic_delete3 = titanic.dropna(axis = 1)
titanic_delete3

titanic_delete5 = titanic.dropna(axis = 1, subset = [0,1,2,42,258])
titanic_delete5

titanic_delete4 = titanic.dropna(axis = 1, thresh = 500)
titanic_delete4

print(":: 변경 전 ::")
print(df)

print('\n')

print(":: 변경 후 ::")
print(df.fillna(0))

titanic.fillna(method = 'pad') #(= method = 'ffill')

titanic.fillna(method = 'bfill')

print(titanic['embark_town'].value_counts())
titanic['embark_town'].fillna('Southampton').isnull().sum()

titanic['age'] = titanic['age'].replace(to_replace = np.nan, value = titanic['age'].mean())
titanic

df2 = pd.DataFrame({'c1':[5,2,3,5,4], 'c2':['a','a','b','a','c'], 'c3':[5,7,5,5,4]},
                  index = ['num1','num2','num3','num4','num5'])
df2

df2.duplicated()

df2.duplicated(['c2','c3'])

df2_1 = df2.drop_duplicates()
df2_1

df2_3 = df2.drop_duplicates(subset = ['c2'], keep = 'first')
df2_3

df2_2 = df2.drop_duplicates(subset = None, keep = 'first')
df2_2

df2_3 = df2.drop_duplicates(subset = None, keep = 'last')
df2_3

df2_4 = df2.drop_duplicates(subset = None, keep = False)
df2_4

arr = [6,2,5,7,8,9,1,1,5,4,3,8,3,2]

result1 = set(arr)
print(result1)

result2 = list(result1)
print(result2)

result = []

for value in arr:
    if value not in result:
        result.append(value)
        
print(result)

result1 = dict.fromkeys(arr)
print(result1)

result2 = list(dict.fromkeys(arr))
print(result2)

!pip install -U imbalanced-learn
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *

df = sns.load_dataset('titanic')

X = df[['pclass', 'sibsp', 'parch','fare']]
y = df['survived']

print('the shape of X:', X.shape)
print('the shape of y:', y.shape)

print('counts of label "1":', sum(y==1))
print('counts of label "0":', sum(y==0))

X_resampled, y_resampled = RandomUnderSampler(random_state = 0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))

X_resampled, y_resampled = TomekLinks().fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))

X_resampled, y_resampled = CondensedNearestNeighbour(random_state=0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))

X_resampled, y_resampled = RandomOverSampler(random_state = 0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))

X_resampled, y_resampled = SMOTE(random_state = 777).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))

X_resampled, y_resampled = ADASYN(random_state = 0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))

e, c = pd.factorize(df['class'])

e

c

df['new_label'] = e

df[['class', 'new_label']].head()

from sklearn.preprocessing import LabelEncoder

sports = ['축구', '농구', '야구', '배구']
encoder = LabelEncoder()
encoder.fit(sports)
labels = encoder.transform(sports)
print(labels)

df['class'].head(10)

df = sns.load_dataset('titanic')
encoder = LabelEncoder()
t_label = encoder.fit_transform(df['class'])

print(t_label)

print('\nLabel mapping:')
for i, j in enumerate(encoder.classes_):
    print(j, '->', i)

df['new_label'] = t_label

df[['class','new_label']].head(10)

df = sns.load_dataset('titanic')
encoder = LabelEncoder()
t_label = encoder.fit_transform(df['class'])

print(t_label)

print(encoder.classes_)

encoder.inverse_transform(t_label)

map_class = {'First':1, 'Second':2, 'Third':3}
df[['class_label']] = df[['class']].applymap(map_class.get)

df[['class','class_label']].head()

cla = pd.get_dummies(df['class'])

cla

clb = pd.get_dummies(df)

clb.head()

from sklearn.preprocessing import OneHotEncoder

df = sns.load_dataset('titanic')

o_encoder = OneHotEncoder()

o_encoder.fit(df[['class']])
oh = o_encoder.transform(df[['class']])
print(oh)

oh = oh.toarray()
print(oh)

from sklearn.preprocessing import OneHotEncoder

df = sns.load_dataset('titanic')

o_encoder = OneHotEncoder(sparse = False)

o_encoder.fit(df[['class']])
oh = o_encoder.transform(df[['class']])
print(oh)

oh = pd.DataFrame(oh)
oh.columns = o_encoder.get_feature_names()
df = pd.concat([df, oh], axis = 1)
df = df.drop(columns = ['class'])

df[['x0_First', 'x0_Second','x0_Third']].tail()

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

x = np.arange(10)

x[9] = 1000

x.mean(), x.std()

scaled = standard_scaler.fit_transform(x.reshape(-1,1))

print('평균:', x.mean(), '\n분산:', x.std())

print('표준화한 결과')
print('평균:', scaled.mean(), '\n분산:', scaled.std())

print('평균이 0에 근사함을 알 수 있다.')
print('평균:', round(scaled.mean(),2), '\n분산:', scaled.std())

from sklearn.preprocessing import MinMaxScaler

x = np.arange(10)
scaled = MinMaxScaler().fit_transform(x.reshape(-1,1))

print('평균:', scaled.mean(), '\n분산:', scaled.std())

print('이상치 추가 후')
x[9] = 1000
scaled = MinMaxScaler().fit_transform(x.reshape(-1,1))

print('평균:', scaled.mean(), '\n분산:', scaled.std())

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('feature 평균')
print(iris_df.mean())
print('feature 분산')
print(iris_df.var())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 평균')
print(iris_df_scaled.mean())
print('feature 분산')
print(iris_df_scaled.var())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 최솟값')
print(iris_df_scaled.min())
print('feature 최댓값')
print(iris_df_scaled.max())

train_array = np.arange(0,11).reshape(-1,1)

test_array = np.arange(0,6).reshape(-1,1)

scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)

print('train_array:', np.round(train_array.reshape(-1),2))
print('scaling 이후의 train_array:', np.round(train_scaled.reshape(-1),2))

scaler.fit(test_array)
test_scaled = scaler.transform(test_array)

print('test_array:', np.round(test_array.reshape(-1),2))
print('scaling 이후의 test_array:', np.round(test_scaled.reshape(-1),2))

train = pd.read_csv('http://bit.ly/fc-ml-titanic')
train.head()

train.info()

train.isnull().sum(axis = 0)

train.groupby(['Survived','Sex'])['Age'].mean()

age_filled = train.groupby(['Survived', 'Sex'])['Age'].transform(lambda x:x.fillna(x.mean()))
age_filled

scaler = MinMaxScaler()
train_age = age_filled.values.reshape(-1,1)

scaler.fit(train_age)
age_scaled = scaler.transform(train_age)
train['AgeScaled'] = pd.Series(age_scaled.reshape(-1))

train.head()

scaler = StandardScaler()
train_fare = train['Fare'].values.reshape(-1,1)
scaler.fit(train_fare)
fare_scaled = scaler.transform(train_fare)
train['FareScaled'] = pd.Series(fare_scaled.reshape(-1))

print('Age의 정규화 적용 전 후 평균과 분산 비교')
print(train[['Age','AgeScaled']].mean())
print(train[['Age','AgeScaled']].var())
print('\nFare의 표준화 적용 전 후 평균과 분산 비교')
print(train[['Fare','FareScaled']].mean())
print(train[['Fare','FareScaled']].var())

train.head()
