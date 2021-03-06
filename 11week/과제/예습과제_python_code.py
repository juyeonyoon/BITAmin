import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.target = df.target.map({0:'setosa', 1:'versicolor', 2:'virginica'})

df.head()

setosa = df[df.target=='setosa']
versicolor = df[df.target=='versicolor']
virginica = df[df.target=='virginica']

import matplotlib.pyplot as plt
fig = plt.figure()

#setosa의 분포
fig, ax = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

setosa['sepal length (cm)'].plot(kind='hist', ax=ax[0,0])
setosa['sepal length (cm)'].plot(kind='kde', ax=ax[0,0], 
                                secondary_y=True,
                                title='sepal length (cm) distribution',
                                figsize=(8,4))

setosa['sepal width (cm)'].plot(kind='hist', ax=ax[0,1],)
setosa['sepal width (cm)'].plot(kind='kde', ax=ax[0,1], 
                                secondary_y=True,
                                title='sepal width (cm) distribution',
                                figsize=(8,4))

setosa['petal length (cm)'].plot(kind='hist', ax=ax[1,0])
setosa['petal length (cm)'].plot(kind='kde', ax=ax[1,0], 
                                secondary_y=True,
                                title='petal length (cm) distribution',
                                figsize=(8,4))

setosa['petal width (cm)'].plot(kind='hist', ax=ax[1,1])
setosa['petal width (cm)'].plot(kind='kde', ax=ax[1,1], 
                                secondary_y=True,
                                title='petal width (cm) distribution',
                                figsize=(8,4))

#versicolor의 분포
fig, ax = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

versicolor['sepal length (cm)'].plot(kind='hist', ax=ax[0,0])
versicolor['sepal length (cm)'].plot(kind='kde', ax=ax[0,0], 
                                secondary_y=True,
                                title='sepal length (cm) distribution',
                                figsize=(8,4))

versicolor['sepal width (cm)'].plot(kind='hist', ax=ax[0,1],)
versicolor['sepal width (cm)'].plot(kind='kde', ax=ax[0,1], 
                                secondary_y=True,
                                title='sepal width (cm) distribution',
                                figsize=(8,4))

versicolor['petal length (cm)'].plot(kind='hist', ax=ax[1,0])
versicolor['petal length (cm)'].plot(kind='kde', ax=ax[1,0], 
                                secondary_y=True,
                                title='petal length (cm) distribution',
                                figsize=(8,4))

versicolor['petal width (cm)'].plot(kind='hist', ax=ax[1,1])
versicolor['petal width (cm)'].plot(kind='kde', ax=ax[1,1], 
                                secondary_y=True,
                                title='petal width (cm) distribution',
                                figsize=(8,4))

#virginica의 분포
fig, ax = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

virginica['sepal length (cm)'].plot(kind='hist', ax=ax[0,0])
virginica['sepal length (cm)'].plot(kind='kde', ax=ax[0,0], 
                                secondary_y=True,
                                title='sepal length (cm) distribution',
                                figsize=(8,4))

virginica['sepal width (cm)'].plot(kind='hist', ax=ax[0,1],)
virginica['sepal width (cm)'].plot(kind='kde', ax=ax[0,1], 
                                secondary_y=True,
                                title='sepal width (cm) distribution',
                                figsize=(8,4))

virginica['petal length (cm)'].plot(kind='hist', ax=ax[1,0])
virginica['petal length (cm)'].plot(kind='kde', ax=ax[1,0], 
                                secondary_y=True,
                                title='petal length (cm) distribution',
                                figsize=(8,4))

virginica['petal width (cm)'].plot(kind='hist', ax=ax[1,1])
virginica['petal width (cm)'].plot(kind='kde', ax=ax[1,1], 
                                secondary_y=True,
                                title='petal width (cm) distribution',
                                figsize=(8,4))

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(metrics.classification_report(y_test, pred))

from sklearn.metrics import ConfusionMatrixDisplay
cm = metrics.confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=load_iris().target_names).plot()

accuracy_score(y_test, pred)

#그리드서치
params_nb = {'var_smoothing':[1,0.5,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001, 0.00000001,0.000000001]}

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(GaussianNB(priors=[1/3,1/3,1/3]), param_grid=params_nb,
                   cv=list(StratifiedKFold(n_splits=5).split(X_train, y_train)),
                   n_jobs=1)
grid.fit(X_train, y_train)

print('best param:', grid.best_estimator_)
print('best score:', grid.best_score_)
best_nb = grid.best_estimator_

best_nb.fit(X_train, y_train)

new_pred = best_nb.predict(X_test)
print(metrics.classification_report(y_test, new_pred))

from sklearn.metrics import ConfusionMatrixDisplay
cm = metrics.confusion_matrix(y_test, new_pred)
ConfusionMatrixDisplay(cm, display_labels=load_iris().target_names).plot()

accuracy_score(y_test, pred)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

X = np.array([
    [0,1,1,0],
    [1,1,1,1],
    [1,1,1,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,1,0],
    [1,0,1,1],
    [0,1,1,0]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

model_bern = BernoulliNB().fit(X, y)

model_bern.classes_

model_bern.class_count_

np.exp(model_bern.class_log_prior_)

fc = model_bern.feature_count_
fc

fc / np.repeat(model_bern.class_count_[:,np.newaxis], 4, axis=1)

model_bern.alpha

theta = np.exp(model_bern.feature_log_prob_)
theta

x_new = np.array([1,1,0,0])

model_bern.predict_proba([x_new])

x_new = np.array([0,0,1,1])

model_bern.predict_proba([x_new])

email_list = [
    {'email title': 'free game only today', 'spam': True},
    {'email title': 'limited time offer only today', 'spam': True},
    {'email title': 'cheapest flight deal', 'spam': True},
    {'email title': 'today flight schedule', 'spam': True},
    {'email title': 'today meeting schedule', 'spam': False},
    {'email title': 'Competition winners announced', 'spam': False},
    {'email title': 'Notification of Change of Schedule', 'spam': False},
    {'email title': 'your credit card statement', 'spam': False}
]

df = pd.DataFrame(email_list)
df.info()

df['label'] = df['spam'].map({True:1, False:0})
df

df_x = df['email title']
df_y = df['label']

cv = CountVectorizer(binary=True)
x_traincv = cv.fit_transform(df_x)
x_traincv

cv.get_feature_names_out()

encoded_input = x_traincv.toarray()
encoded_input

cv.inverse_transform(encoded_input)[0]

bnb = BernoulliNB()
y_train = df_y.astype('int')
bnb.fit(x_traincv, y_train)

test_email_list = [
    {'email title': 'limited free game only today', 'spam': True},
    {'email title': 'today flight schedule', 'spam': True},
    {'email title': 'cheapest game catalogue', 'spam': True},
    {'email title': 'hey traveler free flight deal', 'spam': True},
    {'email title': 'free flight offer', 'spam': False},
    {'email title': 'Competition winners attached', 'spam': False},
    {'email title': 'Notification of Change of announced', 'spam': False},
    {'email title': 'your credit card offer only today', 'spam': False}
]

test_df = pd.DataFrame(test_email_list)
test_df['label'] = test_df['spam'].map({True:1, False:0})
test_x = test_df['email title']
test_y = test_df['label']
x_testcv = cv.transform(test_x)

predictions = bnb.predict(x_testcv)
predictions

accuracy_score(test_y, predictions)
