# 1. 계층형 군집화 문제

 sklean에서 제공하는 load_wine dataset 활용
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

#wine data 불러오기
from sklearn.datasets import load_wine

wine = load_wine()
wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
wine_df['target'] = wine['target']

wine_df.head()

"""### 문제 1-1. Code 작성하고, wine 데이터 특징 파악"""

# 결측치 확인

#Code

# target 분포 확인 

#Code

"""#### 3개의 feature만 선택 해, 계층형 군집화 통해 분류해보려고 한다. """

#3개의 feature 선택

from sklearn.ensemble import RandomForestClassifier

X_train = wine_df.iloc[:,:-1]
y_train = wine_df.iloc[:,-1]

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
sorted_i = rf.feature_importances_.argsort()
sns.barplot(x=rf.feature_importances_[sorted_i], y=wine_df.columns[sorted_i])
plt.show()

"""### 문제 1-2. 3개의 feture 선택해 Code 작성하시오"""

#Code 작성
#3개 feature 선택 

sel_data =

sel_data.head()

"""### 문제 1-3. 다음 코드를 작성하시오"""

#Code작성
#StandScaler()를 이용해 sel_data를 feture scaling 

scaler = 
scaled_X = 
print(scaled_X)

"""#### 계층형 군집화에는 다양한 likage method 존재한다. 어떤 method가 좋을지 확인해보자
#### (ward, average, single, complete)
"""

from sklearn.metrics import adjusted_rand_score
# adjusted_rand_score은 예측 및 실제 클러스터링에서 동일하거나 다른 클러스터에 
# 할당된 쌍을 계산하여 두 클러스터링 간의 유사성을 계산합니다. 

clustering_ari = []

linkage_settings = ['ward', 'average', 'single', 'complete']
for method in linkage_settings:
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    agg.fit(scaled_X)
    
    assignments_scaled_X = agg.labels_
    clustering_ari.append(adjusted_rand_score(y_train, assignments_scaled_X))

d = {'likage':linkage_settings, 'score': clustering_ari}
pd.DataFrame(d)

df_X = pd.DataFrame(scaled_X, columns=sel_data.columns)
scaled_data = pd.concat([df_X, wine_df['target']], axis = 1)
scaled_data.head()

"""### 문제 1-4. method를 선택해 clustering 하고 Dendrogram으로 나타내보자"""

#Code 작성
#제일 높은 score을 가진 method 선택 , metric='euclidean'

clusters = linkage(scaled_data, )
clusters.shape

#Code 작성 
#Dendrogram으로 나타내기

"""#### 클러스터링 결과를 확인해보자"""

from scipy.cluster.hierarchy import fcluster

cut_tree = fcluster(clusters, t=15, criterion='distance')
cut_tree
labels = scaled_data['target']

df = pd.DataFrame({'pred':cut_tree, 'labels':labels})
con_mat = pd.crosstab(df['pred'], df['labels'])
con_mat

"""### 문제 1-5. 위의 코드를 통해 나온 결과를 서술하시오. (어떻게 나왔는지 간단하게 써주시면 됩니다.)

#### 시각화를 통해 클러스터링 결과를 확인해보자
"""

scaled_data['cluster'] = cut_tree

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(scaled_data.iloc[:,0],
          scaled_data.iloc[:,1],
          scaled_data.iloc[:,2],
          c = scaled_data['cluster'],
          edgecolor='k')

ax.set_title('cluster visualization')
ax.set_xlabel('1st feature')
ax.set_ylabel('2nd feature')
ax.set_zlabel('3rd feature')
