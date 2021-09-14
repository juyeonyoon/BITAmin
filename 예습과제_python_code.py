#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


titanic = sns.load_dataset('titanic')
titanic.isnull().sum()


# In[3]:


sns.heatmap(titanic.isnull(), cbar = False)


# In[4]:


sns.heatmap(titanic.isnull(), cbar = True)


# In[5]:


get_ipython().system('pip install missingno')
import missingno as msno


# In[6]:


msno.heatmap(titanic)


# In[7]:


msno.bar(titanic)


# In[8]:


msno.matrix(titanic)


# In[9]:


msno.dendrogram(titanic)


# In[10]:


titanic_delete = titanic.dropna()
titanic_delete


# In[11]:


titanic_delete2 = titanic.dropna(axis = 0, how = 'any') 
titanic_delete2.head()


# In[12]:


titanic_delete_ae = titanic.dropna(subset = ['age','embarked'], how = 'any', axis = 0)
titanic_delete_ae


# In[13]:


dict = {'First':[72, np.nan, np.nan, 95], 'Second':[np.nan, np.nan, np.nan, np.nan],
       'Third':[65, np.nan, 76, 88], 'Fourth':[np.nan, np.nan, np.nan, 90]}

df = pd.DataFrame(dict)
df


# In[14]:


df.dropna(how = 'all')


# In[15]:


df.dropna(thresh = 1)


# In[16]:


df.dropna(thresh = 2)


# In[17]:


df.dropna(thresh = 3)


# In[18]:


df.dropna(thresh = 4)


# In[19]:


titanic_delete7 = titanic.dropna(thresh = 14)
titanic_delete7


# In[20]:


titanic_delete3 = titanic.dropna(axis = 1)
titanic_delete3


# In[21]:


titanic_delete5 = titanic.dropna(axis = 1, subset = [0,1,2,42,258])
titanic_delete5


# In[22]:


titanic_delete4 = titanic.dropna(axis = 1, thresh = 500)
titanic_delete4


# In[23]:


print(":: 변경 전 ::")
print(df)

print('\n')

print(":: 변경 후 ::")
print(df.fillna(0))


# In[24]:


titanic.fillna(method = 'pad') #(= method = 'ffill')


# In[25]:


titanic.fillna(method = 'bfill')


# In[26]:


print(titanic['embark_town'].value_counts())
titanic['embark_town'].fillna('Southampton').isnull().sum()


# In[27]:


titanic['age'] = titanic['age'].replace(to_replace = np.nan, value = titanic['age'].mean())
titanic


# In[28]:


df2 = pd.DataFrame({'c1':[5,2,3,5,4], 'c2':['a','a','b','a','c'], 'c3':[5,7,5,5,4]},
                  index = ['num1','num2','num3','num4','num5'])
df2


# In[29]:


df2.duplicated()


# In[30]:


df2.duplicated(['c2','c3'])


# In[31]:


df2_1 = df2.drop_duplicates()
df2_1


# In[32]:


df2_3 = df2.drop_duplicates(subset = ['c2'], keep = 'first')
df2_3


# In[33]:


df2_2 = df2.drop_duplicates(subset = None, keep = 'first')
df2_2


# In[34]:


df2_3 = df2.drop_duplicates(subset = None, keep = 'last')
df2_3


# In[35]:


df2_4 = df2.drop_duplicates(subset = None, keep = False)
df2_4


# In[36]:


arr = [6,2,5,7,8,9,1,1,5,4,3,8,3,2]

result1 = set(arr)
print(result1)

result2 = list(result1)
print(result2)


# In[37]:


result = []

for value in arr:
    if value not in result:
        result.append(value)
        
print(result)


# In[38]:


result1 = dict.fromkeys(arr)
print(result1)

result2 = list(dict.fromkeys(arr))
print(result2)


# In[39]:


get_ipython().system('pip install -U imbalanced-learn')
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *


# In[40]:


df = sns.load_dataset('titanic')


# In[41]:


X = df[['pclass', 'sibsp', 'parch','fare']]
y = df['survived']

print('the shape of X:', X.shape)
print('the shape of y:', y.shape)


# In[42]:


print('counts of label "1":', sum(y==1))
print('counts of label "0":', sum(y==0))


# In[43]:


X_resampled, y_resampled = RandomUnderSampler(random_state = 0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))


# In[44]:


X_resampled, y_resampled = TomekLinks().fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))


# In[45]:


X_resampled, y_resampled = CondensedNearestNeighbour(random_state=0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))


# In[46]:


X_resampled, y_resampled = RandomOverSampler(random_state = 0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))


# In[47]:


X_resampled, y_resampled = SMOTE(random_state = 777).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))


# In[48]:


X_resampled, y_resampled = ADASYN(random_state = 0).fit_resample(X,y)

print('the shape of X_resampled:', X_resampled.shape)
print('the shape of y_resampled:', y_resampled.shape)
print('\n')
print('counts of label "1":', sum(y_resampled == 1))
print('counts of label "0":', sum(y_resampled == 0))


# In[49]:


e, c = pd.factorize(df['class'])


# In[50]:


e


# In[51]:


c


# In[52]:


df['new_label'] = e


# In[53]:


df[['class', 'new_label']].head()


# In[54]:


from sklearn.preprocessing import LabelEncoder


# In[56]:


sports = ['축구', '농구', '야구', '배구']
encoder = LabelEncoder()
encoder.fit(sports)
labels = encoder.transform(sports)
print(labels)


# In[58]:


df['class'].head(10)


# In[59]:


df = sns.load_dataset('titanic')
encoder = LabelEncoder()
t_label = encoder.fit_transform(df['class'])


# In[60]:


print(t_label)


# In[62]:


print('\nLabel mapping:')
for i, j in enumerate(encoder.classes_):
    print(j, '->', i)


# In[63]:


df['new_label'] = t_label


# In[64]:


df[['class','new_label']].head(10)


# In[65]:


df = sns.load_dataset('titanic')
encoder = LabelEncoder()
t_label = encoder.fit_transform(df['class'])


# In[66]:


print(t_label)


# In[67]:


print(encoder.classes_)


# In[68]:


encoder.inverse_transform(t_label)


# In[69]:


map_class = {'First':1, 'Second':2, 'Third':3}
df[['class_label']] = df[['class']].applymap(map_class.get)


# In[71]:


df[['class','class_label']].head()


# In[72]:


cla = pd.get_dummies(df['class'])


# In[73]:


cla


# In[74]:


clb = pd.get_dummies(df)


# In[75]:


clb.head()


# In[76]:


from sklearn.preprocessing import OneHotEncoder

df = sns.load_dataset('titanic')

o_encoder = OneHotEncoder()

o_encoder.fit(df[['class']])
oh = o_encoder.transform(df[['class']])
print(oh)


# In[77]:


oh = oh.toarray()
print(oh)


# In[78]:


from sklearn.preprocessing import OneHotEncoder

df = sns.load_dataset('titanic')

o_encoder = OneHotEncoder(sparse = False)

o_encoder.fit(df[['class']])
oh = o_encoder.transform(df[['class']])
print(oh)


# In[80]:


oh = pd.DataFrame(oh)
oh.columns = o_encoder.get_feature_names()
df = pd.concat([df, oh], axis = 1)
df = df.drop(columns = ['class'])


# In[81]:


df[['x0_First', 'x0_Second','x0_Third']].tail()


# In[82]:


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()


# In[84]:


x = np.arange(10)

x[9] = 1000


# In[85]:


x.mean(), x.std()


# In[87]:


scaled = standard_scaler.fit_transform(x.reshape(-1,1))


# In[88]:


print('평균:', x.mean(), '\n분산:', x.std())


# In[89]:


print('표준화한 결과')
print('평균:', scaled.mean(), '\n분산:', scaled.std())


# In[90]:


print('평균이 0에 근사함을 알 수 있다.')
print('평균:', round(scaled.mean(),2), '\n분산:', scaled.std())


# In[91]:


from sklearn.preprocessing import MinMaxScaler

x = np.arange(10)
scaled = MinMaxScaler().fit_transform(x.reshape(-1,1))

print('평균:', scaled.mean(), '\n분산:', scaled.std())

print('이상치 추가 후')
x[9] = 1000
scaled = MinMaxScaler().fit_transform(x.reshape(-1,1))

print('평균:', scaled.mean(), '\n분산:', scaled.std())


# In[93]:


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


# In[94]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)


# In[95]:


iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 평균')
print(iris_df_scaled.mean())
print('feature 분산')
print(iris_df_scaled.var())


# In[96]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 최솟값')
print(iris_df_scaled.min())
print('feature 최댓값')
print(iris_df_scaled.max())


# In[98]:


train_array = np.arange(0,11).reshape(-1,1)

test_array = np.arange(0,6).reshape(-1,1)


# In[99]:


scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)

print('train_array:', np.round(train_array.reshape(-1),2))
print('scaling 이후의 train_array:', np.round(train_scaled.reshape(-1),2))


# In[100]:


scaler.fit(test_array)
test_scaled = scaler.transform(test_array)

print('test_array:', np.round(test_array.reshape(-1),2))
print('scaling 이후의 test_array:', np.round(test_scaled.reshape(-1),2))


# In[102]:


train = pd.read_csv('http://bit.ly/fc-ml-titanic')
train.head()


# In[103]:


train.info()


# In[106]:


train.isnull().sum(axis = 0)


# In[107]:


train.groupby(['Survived','Sex'])['Age'].mean()


# In[108]:


age_filled = train.groupby(['Survived', 'Sex'])['Age'].transform(lambda x:x.fillna(x.mean()))
age_filled


# In[110]:


scaler = MinMaxScaler()
train_age = age_filled.values.reshape(-1,1)


# In[111]:


scaler.fit(train_age)
age_scaled = scaler.transform(train_age)
train['AgeScaled'] = pd.Series(age_scaled.reshape(-1))


# In[112]:


train.head()


# In[114]:


scaler = StandardScaler()
train_fare = train['Fare'].values.reshape(-1,1)
scaler.fit(train_fare)
fare_scaled = scaler.transform(train_fare)
train['FareScaled'] = pd.Series(fare_scaled.reshape(-1))


# In[115]:


print('Age의 정규화 적용 전 후 평균과 분산 비교')
print(train[['Age','AgeScaled']].mean())
print(train[['Age','AgeScaled']].var())
print('\nFare의 표준화 적용 전 후 평균과 분산 비교')
print(train[['Fare','FareScaled']].mean())
print(train[['Fare','FareScaled']].var())


# In[116]:


train.head()

