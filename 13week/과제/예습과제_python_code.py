import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(data_df.head())

lr_clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=10000)
knn_clf = KNeighborsClassifier(n_neighbors=8)
rf_cf = RandomForestClassifier(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

vo_soft_clf = VotingClassifier(estimators=[('LR', lr_clf), ('KNN', knn_clf), ('RF', rf_cf)], voting='soft')

vo_soft_clf.fit(X_train, y_train)
pred_s = vo_soft_clf.predict(X_test)

vo_hard_clf = VotingClassifier(estimators=[('LR', lr_clf), ('KNN', knn_clf), ('RF', rf_cf)], voting='hard')

vo_hard_clf.fit(X_train, y_train)
pred_h = vo_hard_clf.predict(X_test)

print('Soft Voting 분류기 정확도', accuracy_score(y_test, pred_s))
print('Hard Votinf 분류기 정확도', accuracy_score(y_test, pred_h))

classifiers = [lr_clf, knn_clf, rf_cf]
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))

from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(5)

mnist = datasets.load_digits()
features, labels = mnist.data, mnist.target

def cross_validation(classifier, features, labels):
    cv_scores = []
    
    for i in range(10):
        scores = cross_val_score(classifier, features, labels, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        
    return cv_scores

np.random.seed(5)
dt_cv_scores = cross_validation(tree.DecisionTreeClassifier(), features, labels)
dt_cv_scores

np.mean(dt_cv_scores)

np.random.seed(5)
rf_cv_scores = cross_validation(RandomForestClassifier(), features, labels)
rf_cv_scores

np.mean(rf_cv_scores)

cv_list = [['random_forest', rf_cv_scores], ['decision_tree', dt_cv_scores]]
df = pd.DataFrame.from_dict(dict(cv_list))
df

df.plot()

feature_name_df = pd.read_csv('features.txt', sep='\s+', header=None, names=['column_index', 'column_name'])

print(feature_name_df.shape)
feature_name_df.groupby('column_name').count().sort_values('column_index')

def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x: x[0]+'_'+str(x[1])
                                                                                              if x[1]>0 else x[0], axis=1)
    
    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)
    return new_feature_name_df

def get_human_dataset():
    feature_name_df = pd.read_csv('features.txt', sep=' ', header=None, names=['column_index', 'column_name'])
    
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    
    X_train = pd.read_csv('X_train.txt', sep='\s+', names=feature_name)
    X_test = pd.read_csv('X_test.txt', sep='\s+', names=feature_name)
    
    y_train = pd.read_csv('y_train.txt', sep='\s+ ', header=None, names=['action'])
    y_test = pd.read_csv('y_test.txt', sep='\s+', header=None, names=['action'])
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_human_dataset()

X_train.shape, y_train.shape

X_test.shape, y_test.shape

X_train.head()

y_train

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as spd
import warnings
warnings.filterwarnings('ignore')

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy))

from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators': [50],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [8, 12, 18],
    'min_samples_split' : [8, 16, 20]
}

rf_clf = RandomForestClassifier(random_state=0, n_jobs=1)

grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

rf_clf1 = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=18, 
                                min_samples_split=8, random_state=0)
rf_clf1.fit(X_train, y_train)
pred = rf_clf1.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()

!pip install xgboost

import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/hmkim312/datas/main/HAR/features.txt'

feature_name_df = pd.read_csv(url, sep='\s+', header=None, names=['columns_index','column_name'])
feature_name = feature_name_df.iloc[:,1].values.tolist()
X_train = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/X_train.txt',sep='\s+', header=None)
X_test = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/X_test.txt', sep='\s+', header=None)
y_train = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/y_train.txt', 
                      sep='\s+', header=None, names=['action'])
y_test = pd.read_csv('https://raw.githubusercontent.com/hmkim312/datas/main/HAR/y_test.txt', 
                      sep='\s+', header=None, names=['action'])
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.columns = feature_name
X_test.columns = feature_name
X_train.head()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import time
import warnings

warnings.filterwarnings('ignore')
start_time = time.time()

clf = AdaBoostClassifier(random_state=10)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print('ACC: ', accuracy_score(y_test,pred))
print('AdaBoost 수행 시간: {:.1}초'.format(time.time()-start_time))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
import warnings

warnings.filterwarnings('ignore')
start_time = time.time()

tree_model = DecisionTreeClassifier(max_depth=20)
clf = AdaBoostClassifier(base_estimator=tree_model, n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print('ACC: ', accuracy_score(y_test, pred))
print('hyper parameter 수정 후 AdaBoost 수행 시간: {:.1f}초'.format(time.time()-start_time))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=13)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)

print('ACC: ', accuracy_score(y_test, gb_pred))
print('GBM 수행 시간: {:.1f}초'.format(time.time()-start_time))

from xgboost import XGBClassifier

start_time = time.time()
xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=3)
xgb.fit(X_train.values, y_train)
print('Acc: ', accuracy_score(y_test, xgb.predict(X_test.values)))
print('XGBoost 수행 시간: {:.1f}초'.format(time.time()-start_time))

from xgboost import XGBClassifier

evals = [(X_test.values, y_test)]

start_time = time.time()
xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, max_depth=3)
xgb.fit(X_train.values, y_train, early_stopping_rounds=5, eval_set=evals)
print('Acc: ', accuracy_score(y_test, xgb.predict(X_test.values)))
print('XGBoost 수행 시간: {:.1f}초'.format(time.time()-start_time))

!pip install lightgbm

from lightgbm import LGBMClassifier

start_time = time.time()
lgbm = LGBMClassifier(n_estimator=400)
lgbm.fit(X_train.values, y_train, early_stopping_rounds=10, eval_set=evals)
print('Acc: ', accuracy_score(y_test, lgbm.predict(X_test.values)))
print('LightGBM 수행 시간: {:.1f}초'.format(time.time()-start_time))
