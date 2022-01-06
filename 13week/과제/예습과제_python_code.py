import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
X = iris.data
y = iris.target

pd.DataFrame(X, columns=iris.feature_names)

from sklearn.decomposition import PCA

X_reduced = PCA(n_components=3).fit_transform(X)
pd.DataFrame(X_reduced, columns=['1_?', '2_?', '3_?'])

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)
X4D = pca.inverse_transform(X_reduced)
pd.DataFrame(X4D, columns=iris.feature_names)

#주성분
print('주성분 : \n', pca.components_)

#표현 분산
print('표현 분산 : \n', pca.explained_variance_ratio_)

ratio = pca.explained_variance_ratio_

df_v = pd.DataFrame(ratio, index=['PC1', 'PC2', 'PC3'], columns=['V_ration'])
plt.pie(df_v['V_ration'], labels=df_v.index, autopct='%.2f%%')
plt.title('explained_variance_ratio')
plt.show()

fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor='k',
    s=40)
ax.set_title('First three PCA directions')
ax.set_xlabel('1st eigenvector')
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel('2nd eigenvector')
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel('3rd eigenvector')
ax.w_zaxis.set_ticklabels([])
plt.show()

rnd_pca = PCA(n_components=3, svd_solver='randomized', random_state=42)
X_reduced_rnd = rnd_pca.fit_transform(X)

from sklearn.decomposition import IncrementalPCA
import numpy as np

n_batches = 10
inc_pca = IncrementalPCA(n_components=3)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced_inc = inc_pca.transform(X)

from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001, coef0=1, fit_inverse_transform=True)

X_reduced_lin = lin_pca.fit_transform(X)
X_reduced_rbf = rbf_pca.fit_transform(X)
X_reduced_sig = sig_pca.fit_transform(X)

import seaborn as sns

sns.scatterplot(x=X_reduced_lin[:,0], y=X_reduced_lin[:,1], hue=y)
plt.xlabel('components_1')
plt.ylabel('components_2')
plt.show()

sns.scatterplot(x=X_reduced_rbf[:, 0], y=X_reduced_rbf[:, 1], hue=y)
plt.xlabel('components_1')
plt.ylabel('components_2')
plt.show()

sns.scatterplot(x=X_reduced_sig[:, 0], y=X_reduced_sig[:, 1], hue=y)
plt.xlabel('components_1')
plt.ylabel('components_2')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression(solver='lbfgs'))
])

param_grid = [{
    'kpca__gamma': np.linspace(0.03, 0.05, 10),
    'kpca__kernel' : ['rbf', 'sigmoid']
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433, 
                   fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

from sklearn.metrics import mean_squared_error

mean_squared_error(X, X_preimage)

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X_reduced_lda = lda.transform(X)

sns.scatterplot(x=X_reduced_lda[:,0], y=X_reduced_lda[:,1], hue=y)
plt.xlabel('components_1')
plt.ylabel('components_2')
plt.show()

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

N = 100
rv1 = stats.multivariate_normal([0, 0], [[0.7, 0.0], [0.0, 0.7]])
rv2 = stats.multivariate_normal([1, 1], [[0.8, 0.2], [0.2, 0.8]])
rv3 = stats.multivariate_normal([-1, 1], [[0.8, 0.2], [0.2, 0.8]])
np.random.seed(0)
X1 = rv1.rvs(N)
X2 = rv2.rvs(N)
X3 = rv3.rvs(N)
y1 = np.zeros(N)
y2 = np.ones(N)
y3 = 2*np.ones(N)
X = np.vstack([X1, X2, X3])
y = np.hstack([y1, y2, y3])

plt.scatter(X1[:, 0], X1[:, 1], alpha=0.8, s=50, marker='o', color='r', label='class 1')
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.8, s=50, marker='s', color='g', label='class 2')
plt.scatter(X3[:, 0], X3[:, 1], alpha=0.8, s=50, marker='x', color='b', label='class 3')
plt.xlim(-5, 5)
plt.ylim(-4, 5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)

qda.priors_

qda.means_

qda.covariance_[0]

qda.covariance_[1]

qda.covariance_[2]

import matplotlib as mpl
import seaborn as sns

x1min, x1max = -5, 5
x2min, x2max = -4, 5
XX1, XX2 = np.meshgrid(np.arange(x1min, x1max, (x1max-x1min)/1000),
                      np.arange(x2min, x2max, (x2max-x2min)/1000))
YY = np.reshape(qda.predict(np.array([XX1.ravel(), XX2.ravel()]).T), XX1.shape)
cmap = mpl.colors.ListedColormap(sns.color_palette(['r','g','b']).as_hex())
plt.contourf(XX1, XX2, YY, cmap=cmap, alpha=0.5)
plt.scatter(X1[:, 0], X1[:, 1], alpha=0.8, s=50, marker='o', color='r', label='class 1')
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.8, s=50, marker='s', color='g', label='class 2')
plt.scatter(X3[:, 0], X3[:, 1], alpha=0.8, s=50, marker='x', color='b', label='class 3')
plt.xlim(x1min, x1max)
plt.ylim(x2min, x2max)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Result of QDA')
plt.legend()
plt.show()

!pip install pydataset

import pandas as pd
from pydataset import data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns

df = data('Wages1')
df.head()

df.info()

df.describe()

sns.countplot(df['sex'])

fig = plt.figure()
fig, axs = plt.subplots(figsize=(15,5), ncols=3)
sns.set(font_scale=1.4)
sns.distplot(df['exper'], color='black', ax=axs[0])
sns.distplot(df['school'], color='black', ax=axs[1])
sns.distplot(df['wage'], color='black', ax=axs[2])

sex = pd.get_dummies(df['sex'])
df.drop(['sex'], axis=1, inplace=True)
df = pd.concat([df, sex], axis=1)
df.head()

corrmat = df.corr(method='pearson')
f, ax = plt.subplots(figsize=(8, 8))
sns.set(font_scale=1.2)
sns.heatmap(round(corrmat,2),
           vmax=1., square=True,
           cmap='gist_gray', annot=True)

X = df[['exper', 'school', 'wage']]
y = df['male']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=50)

lda = LinearDiscriminantAnalysis()
model_lda = lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

cm = confusion_matrix(y_test, y_pred_lda)
ax = plt.subplots(figsize=(5,5))
with sns.axes_style('white'):
    sns.heatmap(cm, cbar=False, square=True, annot=True, fmt='g', linewidths=2.5,
               xticklabels={'female','male'}, yticklabels={'female','male'})
plt.ylabel('True'); plt.xlabel('Predict')

round(accuracy_score(y_test, y_pred_lda), 4)

print(classification_report(y_test, y_pred_lda))

qda = QuadraticDiscriminantAnalysis()
model_qda = qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

cm = confusion_matrix(y_test, y_pred_qda)
ax = plt.subplots(figsize=(5,5))
with sns.axes_style('white'):
    sns.heatmap(cm, cbar=False, square=True, annot=True, fmt='g', linewidths=2.5,
               xticklabels={'female','male'}, yticklabels={'female','male'})
plt.ylabel('True'); plt.xlabel('Predict')

round(accuracy_score(y_test, y_pred_qda), 4)

print(classification_report(y_test, y_pred_qda))

false_positive_rate_qda, true_positive_rate_qda, thresholds = roc_curve(y_test, y_pred_qda)
roc_auc_qda = auc(false_positive_rate_qda, true_positive_rate_qda)
false_positive_rate_lda, true_positive_rate_lda, thresholds = roc_curve(y_test, y_pred_lda)
roc_auc_lda = auc(false_positive_rate_lda, true_positive_rate_lda)

plt.figure(figsize=(6,6))
plt.title('Receiver Operating Characteristics')
plt.plot(false_positive_rate_lda, true_positive_rate_lda,
        color='green', label='LDA AUC = {:.2f}'.format(roc_auc_lda))
plt.plot(false_positive_rate_qda, true_positive_rate_qda,
        color='red', label='QDA AUC = {:.2f}'.format(roc_auc_qda))
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
