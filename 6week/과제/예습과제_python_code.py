import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors

iris = sns.load_dataset('iris')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

iris[features] = scaler.fit_transform(iris[features])

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(iris, test_size = 0.3, random_state = 1)

from sklearn.model_selection import cross_val_score

k_range = np.arange(1, 80)
k_scores = []

for k in k_range:
    knn = neighbors.KNeighborsClassifier(k)
    scores = cross_val_score(knn, df_train[features],
                            df_train['species'], cv = 10,
                            scoring = 'accuracy')
    k_scores.append(scores.mean())
    
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()

result = pd.DataFrame()
result['k'] = k_range
result['accuracy'] = k_scores
result = result.sort_values(by = 'accuracy', ascending = False).reset_index(drop = True)
result.head()

classifier = neighbors.KNeighborsClassifier(result['k'][0])
classifier.fit(df_train[features], df_train['species'])
pred = classifier.predict(df_test[features])

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(df_test['species'], pred))
print(accuracy_score(df_test['species'], pred))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

X, Y = make_circles(n_samples = 500, noise = 0.02)

plt.scatter(X[:, 0],X[:,1], c = Y, marker = '.')
plt.show()

X1 = X[:,0].reshape((-1,1))
X2 = X[:,1].reshape((-1,1))
X3 = (X1**2 + X2**2)
X = np.hstack((X, X3))

fig = plt.figure()
axes = fig.add_subplot(111, projection = '3d')
axes.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True)
plt.show()

from sklearn import svm

svc = svm.SVC(kernel = 'linear')
svc.fit(X, Y)
w = svc.coef_
b = svc.intercept_

x1 = X[:,0].reshape((-1,1))
x2 = X[:,1].reshape((-1,1))
x1, x2 = np.meshgrid(x1, x2)
x3 = -(w[0][0]*x1 +  w[0][1]*x2 + b) / w[0][2]

fig = plt.figure()
axes2 = fig.add_subplot(111, projection = '3d')
axes2.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True)
axes1 = fig.gca(projection = '3d')
axes1.plot_surface(x1, x2, x3, alpha = 0.01)
plt.show()

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_stat.csv')

print(df.shape)
print('-----------------------------------')
df.head()

df.Pos.value_counts()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

sns.lmplot('STL','2P',data = df,fit_reg = False,
          scatter_kws = {'s':150}, 
          markers = ['o','x'],
          hue = 'Pos')

plt.title('STL and 2P in 2d plane')

sns.lmplot('AST','2P',data = df,fit_reg = False,
          scatter_kws = {'s':150},
          markers = ['o','x'],
          hue = 'Pos')

plt.title('AST and 2P in 2d plane')

sns.lmplot('BLK','3P',data=df, fit_reg=False,
          scatter_kws={'s':150},
          markers=['o','x'],
          hue = 'Pos')

plt.title('BLK and 3P in 2d plane')

sns.lmplot('TRB','3P',data = df, fit_reg=False,
          scatter_kws = {'s':150},
          markers = ['o','x'],
          hue = 'Pos')

plt.title('TRB and 3P in 2d plane')

df.drop(['2P','AST','STL'], axis = 1, inplace = True)
df.head()

df.drop('TRB', axis = 1, inplace = True)
df.head()

X = df[['3P','BLK']]
Y = df[['Pos']]

from sklearn.svm import SVC
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate

svm_clf_l = SVC(kernel = 'linear')

scores_l = cross_val_score(svm_clf_l, X, Y, cv = 5)
scores_l

print('???????????? ??????:', scores_l.mean())
print(pd.DataFrame(cross_validate(svm_clf_l, X, Y, cv = 5)))

svm_clf_r = SVC(kernel = 'rbf')

scores_r = cross_val_score(svm_clf_r, X, Y, cv = 5)
scores_r

print('???????????? ??????:', scores_r.mean())
print(pd.DataFrame(cross_validate(svm_clf_r, X, Y, cv = 5)))

from  sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.2, random_state=100)

train.shape
train

test.shape

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import numpy as np

def svc_param_selection(X, y, nfolds):
    svm_parameters = [
        {'kernel':['rbf'],
        'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
        'C': [0.01,0.1,1,10,100,1000]
        }
    ]
    clf = GridSearchCV(SVC(), svm_parameters, cv = 10)
    clf.fit(X_train, y_train.values.ravel())
    print(clf.best_params_)
    
    return clf

X_train = train[['3P','BLK']]
y_train = train[['Pos']]

clf = svc_param_selection(X_train, y_train.values.ravel(), 10)

C_candidates = []
C_candidates.append(clf.best_params_['C']*0.01)
C_candidates.append(clf.best_params_['C'])
C_candidates.append(clf.best_params_['C']*100)

gamma_candidates = []
gamma_candidates.append(clf.best_params_['gamma']*0.01)
gamma_candidates.append(clf.best_params_['gamma'])
gamma_candidates.append(clf.best_params_['gamma']*100)

X = train[['3P','BLK']]
Y = train['Pos'].tolist()

position = []
for gt in Y:
    if gt == 'C':
        position.append(0)
    else:
        position.append(1)

classifiers = []
for C in C_candidates:
    for gamma in gamma_candidates:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X,Y)
        classifiers.append((C,gamma,clf))
        
plt.figure(figsize = (18, 18))
xx, yy = np.meshgrid(np.linspace(0,4,100), np.linspace(0,4,100))

for (k, (C, gamma, clf)) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.subplot(len(C_candidates), len(gamma_candidates), k+1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size = 'medium')
    
    plt.pcolormesh(xx, yy, -Z, cmap = plt.cm.RdBu)
    plt.scatter(X['3P'], X['BLK'], c = position, cmap = plt.cm.RdBu_r, edgecolors = 'k')

X_test = test[['3P', 'BLK']]

Y_test = test[['Pos']]

y_true, y_pred = Y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))
print()
print('accuracy :' + str(accuracy_score(y_true, y_pred)))

comperison = pd.DataFrame({'prediction':y_pred, 'ground_truth':y_true.values.ravel()})
comperison
