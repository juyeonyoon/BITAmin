pip install graphviz

from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

#DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(random_state=156)

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)

dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='tree.dot', class_names=iris_data.target_names, 
               feature_names=iris_data.feature_names, impurity=True, filled=True)

import graphviz

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#max_depth = 3인 경우 
dt_clf = DecisionTreeClassifier(random_state=156, max_depth=3)

dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='tree.dot', class_names=iris_data.target_names, 
               feature_names=iris_data.feature_names, impurity=True, filled=True)

import graphviz

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#min_samples_split = 4인경우
dt_clf = DecisionTreeClassifier(random_state=156, min_samples_split=4)

dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='tree.dot', class_names=iris_data.target_names,
               feature_names=iris_data.feature_names, impurity=True, filled=True)

import graphviz

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#min_samples_leaf = 4인 경우
dt_clf = DecisionTreeClassifier(random_state=156, min_samples_leaf=4)

dt_clf.fit(X_train, y_train)

export_graphviz(dt_clf, out_file='tree.dot', class_names=iris_data.target_names,
               feature_names=iris_data.feature_names, impurity=True, filled=True)

import graphviz

with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#실습

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

titanic_data = pd.read_csv('titanic_data_clean.csv')

titanic_data

titanic_data.info()

t_features = titanic_data[titanic_data.columns[:-1]]
t_target = titanic_data[titanic_data.columns[-1]]

#Sex 변수: female -> 0, male -> 1
t_features['Sex'] = t_features.Sex.map({'female':0, 'male':1})

#Pclass 변수: One-hot Encoding
t_features = pd.get_dummies(data=t_features, columns=['Pclass'], prefix='Pclass')

train_features, test_features, train_target, test_target = train_test_split(t_features, t_target, test_size=0.2,
                                                                           random_state=2021, stratify=t_target)

pd.DataFrame(train_target)['Survived'].value_counts()

#Random Under Sampling

import sklearn

x_shuffled = sklearn.utils.shuffle(train_features, random_state=2021)
y_shuffled = sklearn.utils.shuffle(train_target, random_state=2021)

import imblearn
from imblearn.under_sampling import RandomUnderSampler
train_features_us, train_target_us = RandomUnderSampler(random_state=2021).fit_resample(x_shuffled, y_shuffled)

pd.DataFrame(train_target_us)['Survived'].value_counts()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', random_state=2021)

tree_fit = tree.fit(train_features_us, train_target_us)

from sklearn.tree import export_graphviz

feature_names = train_features_us.columns.tolist()

target_name = np.array(['Dead', 'Survive'])

dot_data = export_graphviz(tree, 
                          filled=True,
                          rounded=True,
                          class_names=target_name,
                          feature_names=feature_names)

graphviz.Source(dot_data)

#교차 검증을 통한 성능 평가
scores = cross_validate(estimator=tree,
                        X=train_features_us,
                        y=train_target_us,
                        scoring=['accuracy'],
                        cv=10,
                        n_jobs=1,
                        return_train_score=False)

print('CV accuracy: %s' % scores['test_accuracy'])
print('CV accuracy(Mean): %.3f (std: %.3f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))

y_pred = tree.predict(test_features)
y_pred

#오차행렬
confmat = pd.DataFrame(confusion_matrix(test_target, y_pred), 
                      index=['True[0]', 'True[1]'],
                      columns=['Predict[0]', 'Predict[1]'])

confmat

#분류 모델 평가 지표(정확도, 정밀도, 재현율, F1-score, AUC)

print('정확도 accuracy: %.3f' % accuracy_score(test_target, y_pred))
print('정밀도 precision: %.3f' % precision_score(y_true=test_target, y_pred=y_pred))
print('재현율 recall: %.3f' % recall_score(y_true=test_target, y_pred=y_pred))
print('F1-score: %.3f' % f1_score(y_true=test_target, y_pred=y_pred))
print('AUC: %.3f' % roc_auc_score(test_target, y_pred))

#ROC 커브

fpr, tpr, thresholds = roc_curve(test_target, tree.predict_proba(test_features)[:,1])

plt.plot(fpr, tpr, '--', label='Decision Tree')
plt.plot([0,1], [0,1], 'k--', label='random guess')
plt.plot([fpr], [tpr], 'r--', ms=10)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()

#파라미터 튜닝

dt_clf = DecisionTreeClassifier(random_state=2021)

print('DecisionTreeClassifier 기본 하이퍼 파라미터 : \n', dt_clf.get_params())

#max_depth
param_range1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#min_sample_leaf
param_range2 = [10, 20, 30, 40, 50]

#criterion(불순도 계산 방법)
param_range3 = ['gini', 'entropy']

param_grid = [{'max_depth': param_range1,
              'min_samples_leaf': param_range2,
              'criterion': param_range3}]

gs = GridSearchCV(estimator=dt_clf,
                 param_grid=param_grid,
                 scoring='accuracy',
                 cv=10,
                 n_jobs=1)

gs = gs.fit(train_features_us, train_target_us)

print('GridSearchCV 최고 평균 정확도 수치 : {0:.4f}'.format(gs.best_score_))
print('GridSearchCV 최적 하이퍼 파라미터 : ', gs.best_params_)

#최적의 모델 선택
best_tree = gs.best_estimator_
best_tree.fit(train_features_us, train_target_us)

y_pred = best_tree.predict(test_features)
y_pred

confmat = pd.DataFrame(confusion_matrix(test_target, y_pred), index=['True[0]', 'True[1]'],
                      columns=['Predict[0]', 'Predict[1]'])

confmat

print('정확도 accuracy: %.3f' % accuracy_score(test_target, y_pred))
print('정밀도 precision: %.3f' % precision_score(y_true=test_target, y_pred=y_pred))
print('재현율 recall: %.3f' % recall_score(y_true=test_target, y_pred=y_pred))
print('F1-score: %.3f' % f1_score(y_true=test_target, y_pred=y_pred))
print('AUC: %.3f' % roc_auc_score(test_target, y_pred))

fpr, tpr, thresholds = roc_curve(test_target, tree.predict_proba(test_features)[:,1])

plt.plot(fpr, tpr, '--', label='Decision Tree')
plt.plot([0,1], [0,1], 'k--', label='random guess')
plt.plot([fpr], [tpr], 'r--', ms=10)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()

feature_names = train_features_us.columns.tolist()

target_names = np.array(['Dead','Survive'])

dot_data_best = export_graphviz(best_tree, 
                                filled=True,
                                rounded=True,
                                class_names=target_name,
                                feature_names=feature_names)

graphviz.Source(dot_data_best)

#feature importance

feature_importance_values = best_tree.feature_importances_
feature_importances = pd.Series(feature_importance_values, index=train_features_us.columns)

feature_top5 = feature_importances.sort_values(ascending=False)[:5]

plt.figure(figsize=[8,6])
plt.title('Feature Importances Top 5')
sns.barplot(x=feature_top5, y=feature_top5.index)
plt.show()

feature_importances.sort_values(ascending=False)
