# < 2주차 예습 >

# matplotlib

**막대그래프**

import matplotlib.pyplot as plt
import pandas as pd

labels = ['Kor', 'Maths', 'Eng', 'Sci']
scores = [80,90,75,100]
plt.bar(labels, scores) #가로형 막대그래프
plt.show()

plt.barh(labels, scores) #세로형 막대그래프
plt.show()

plt.bar(labels, scores, align = 'center', color = 'green', alpha = 0.5)

plt.xticks(labels, fontsize = 15, rotation = 30)
plt.yticks(fontsize = 15, rotation = 30)

plt.xlabel('Subject', fontsize = 15)
plt.ylabel('Score', fontsize = 15)
plt.title('Scores for midterm exam', fontsize = 20)

plt.show()

plt.barh(labels, scores, align = 'center', color = 'red', alpha = 0.5)

plt.xticks(fontsize = 15, rotation = 30)
plt.yticks(labels, fontsize = 15, rotation = 30)

plt.xlabel('Score', fontsize = 15)
plt.ylabel('Subject', fontsize = 15)
plt.title('Scores for midterm exam', fontsize = 20)

plt.show()

data = [['A', 90, 100, 80], ['B', 70, 90, 85], ['C', 100, 95, 75], ['D', 80, 90,95]]
df = pd.DataFrame(data, columns = ['Name','Kor','Maths','Eng'])
df.plot(x = 'Name', y = ['Kor','Maths','Eng'], kind = 'bar', figsize = (9,8))
plt.show()

df.plot(x = 'Name', y = ['Kor','Maths','Eng'], kind = 'barh', figsize = (9,8))
plt.show()

employees = ['Rudra','Alok','Prince','Nayan','Reman']
earnings = {'January':[10,20,15,18,14], 'Faburary':[20,13,10,18,15], 'March':[20,20,10,15,18]}
df = pd.DataFrame(earnings, index = employees)
df.plot(kind = 'bar', stacked = True, figsize = (10,8))
plt.legend(loc = 'lower left')
plt.show()

"""**파이 차트**"""

labels = ['Dog', 'Cat', 'Fish', 'Others']
sizes = [45, 30, 15, 10]

plt.pie(sizes)
plt.show()

plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=(0.2,0,0,0), shadow=True, startangle=90)
plt.title('Animals', fontsize = 15)
plt.show()

ratio = [34, 32,16,18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
explode = [0.05, 0.05, 0.05, 0.05]
colors = ['red', 'yellow', 'green', 'purple']
wedgeprops = {'width':0.7, 'edgecolor': 'w', 'linewidth':5}

plt.pie(ratio, labels=labels, autopct='%1.1f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)

"""**선 그래프**"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.arange(0,10, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

year = [2015, 2016, 2017, 2018, 2019]
goals = [52, 59, 54, 51, 50]
plt.plot(year, goals, color = 'green', marker = 'o', linestyle = 'solid')

plt.title('Number of goals in 5 years')
plt.xlabel('Year')
plt.ylabel('Number of goals')

plt.xticks(rotation = 50)
plt.show()

year = [2015, 2016, 2017, 2018, 2019]
goals = [52, 59, 54, 51, 50]
plt.plot(year, goals, color = 'green', marker = 'o', linestyle = 'solid')

plt.title('Number of goals in 5 years')
plt.xlabel('Year')
plt.ylabel('Number of goals')

plt.xticks(rotation = 50)
plt.ylim(40,60)
plt.xlim(2014,2020)
plt.show()

year = [2015, 2016, 2017, 2018, 2019]
goals = [52, 59, 54, 51, 50]
goals2 = [57, 55, 53, 49, 52]
plt.plot(year, goals, color = 'green', marker = 'o', linestyle = 'solid')
plt.plot(year, goals2, color = 'blue', marker = 'o', linestyle = 'solid')

plt.title('Number of goals in 5 years')
plt.xlabel('Year')
plt.ylabel('Number of goals')

plt.xticks(rotation = 50)
plt.show()

year = [2015, 2016, 2017, 2018, 2019]
goals = [52, 59, 54, 51, 50]
goals2 = [57, 55, 53, 49, 52]
plt.plot(year, goals, color = 'green', marker = 'o', linestyle = 'dotted', label = ('messi'))
plt.plot(year, goals2, color = 'blue', marker = 'o', linestyle = 'dotted', label = ('honaldo'))

plt.legend(loc = 'upper right')

plt.title('Number of goals in 5 years')
plt.xlabel('Year')
plt.ylabel('Number of goals')

plt.xticks(rotation = 50)
plt.show()

"""**히스토그램**"""

N = 10000
x = np.random.randn(N)

plt.hist(x, bins = 30)
plt.show()

plt.hist(x, bins = 30, density=True, cumulative=True)
plt.yticks(fontsize = 15)

plt.show()

plt.hist(x, bins=30, color='green', alpha=0.5)
plt.hist(x+5, bins=60, color='orange',alpha=0.5)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(['histogram1','histogram2'], fontsize=15)
plt.show()

"""
**박스플롯**

"""

spread = np.random.rand(50)*100
center = np.ones(25)*50
flier_high = np.random.rand(10)*100+100
flier_low = np.random.rand(10)*-100
data = np.concatenate((spread, center, flier_high, flier_low))

plt.boxplot(data)
plt.show()

plt.boxplot(data, vert = False)
plt.show()

spread = np.random.rand(50)*100
center1 = np.ones(25)*50
center2 = np.ones(25)*50
flier_high = np.random.rand(10)*100+100
flier_low = np.random.rand(10)*-100
data1 = np.concatenate((spread, center1, flier_high, flier_low))
data2 = np.concatenate((spread, center2, flier_high, flier_low))
data = [data1, data2, data2[::5]]

plt.boxplot(data)
plt.show()

"""**산점도**"""

x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x,y)
plt.show()

import seaborn as sns
tips = sns.load_dataset('tips')

plt.title('correlation of total_bill and tip', fontsize = 15)
plt.scatter(x=tips['total_bill'], y = tips['tip'], c = 'coral', s = 10, alpha = 0.7)
plt.show()

"""# seborn

**데이터셋 불러오기**
"""

import seaborn as sns
 
titanic = sns.load_dataset('titanic')
titanic

"""**막대그래프**"""

labels = ['Kor', 'Maths', 'Eng', 'Sci']
scores = [80, 90, 75, 100]

sns.barplot(x = labels, y = scores, palette='YlGnBu')

sns.barplot(x = scores, y = labels, palette='YlGnBu')

sns.barplot(x = 'sex', y = 'survived', hue = None, data = titanic, palette='YlGnBu')

sns.countplot(x = 'class', data = titanic)

sns.countplot(x = 'class', data = titanic, hue = 'who')

sns.countplot(x = 'class', data = titanic, hue = 'who', dodge = False)

"""**박스플롯**"""

tips = sns.load_dataset('tips')

sns.boxplot(x = 'day', y = 'total_bill', data = tips)

sns.boxplot(x = 'day', y = 'total_bill', data = tips, hue = 'smoker')

"""**바이올린플롯**"""

tips = sns.load_dataset('tips')
sns.violinplot(x = tips['total_bill'])

sns.violinplot(data = tips, x = 'day', y = 'total_bill')

sns.violinplot(data = tips, x = 'day', y = 'total_bill', hue = 'sex', inner = 'quartile')

sns.violinplot(data = tips, x = 'day', y = 'total_bill', hue = 'sex', split = True, inner = 'quartile')

"""**pairplot**"""

iris = sns.load_dataset('iris')
sns.pairplot(iris)

"""**히트맵**"""

uniform_data = np.random.rand(10,12)
sns.heatmap(uniform_data, annot = True)

flights = sns.load_dataset('flights')
flights_pivot = flights.pivot('month', 'year', 'passengers')
sns.heatmap(flights_pivot, cmap='YlGn', annot=True, fmt='d', linewidths=3)

titanic = sns.load_dataset('titanic')
titanic_corr = titanic.corr()
sns.heatmap(titanic_corr, cmap = 'Blues', annot = True)

"""# plotly

**사용할 데이터 불러오기**
"""

pip install plotly

import plotly.express as px

df1 = px.data.gapminder()
df1.head()

df3 = px.data.tips()
df3. head()

df4 = px.data.election()
df4.head()

"""**선 그래프**"""

df = px.data.gapminder().query("country=='Korea, Rep.'")
fig = px.line(df, x = 'year', y = 'lifeExp', title = 'Life expectancy in Korea', color_discrete_sequence = ['red'])
fig.show()

df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x = 'year', y = 'lifeExp', color = 'country')
fig.show()

df_asia = px.data.gapminder().query("continent == 'Asia'")
print(df_asia.head())
fig = px.line(df_asia, x = 'year', y = 'lifeExp', color = 'country', title = '아시아 국가들의 시간에 따른 평균수명')
fig.show()

"""**막대 그래프**"""

df = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.bar(df, x = 'year', y = 'pop', title = '대한미눅 연도별 인구변화')
fig.show()

df = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.bar(df, x = 'year', y = 'pop', title = 'Life expectancy in Korea', hover_data=['lifeExp', 'gdpPercap'], color = 'lifeExp', labels = {'pop':'population of Korea'}, height=400)
fig.show()

df = px.data.tips()
fig = px.bar(df, x = 'sex', y = 'total_bill', color = 'smoker', barmode='group', height=400)
fig.show()

df = px.data.tips()
fig = px.bar(df, x = 'sex', y = 'total_bill', color = 'smoker', barmode='group', facet_row='time', 
             facet_col='day', category_orders={'day':['Thur','Fri','Sat','Sun'], 'time':['Lunch','Dinner']})
fig.show()

"""**막대그래프(animation)**"""

df = px.data.gapminder()
fig = px.bar(df, x = 'continent', y = 'pop', color = 'continent', animation_frame='year', animation_group='country', range_y = [0,4000000000])
fig.show()

"""**산점도**"""

iris = px.data.iris()
print(iris.head())
fig = px.scatter(iris, x='petal_width', y='petal_length', color='species', size='sepal_length', hover_data=['sepal_width'])
fig.show()

tips = px.data.tips()
fig = px.scatter(tips, x = 'total_bill', y = 'tip', color='sex', facet_row='day')
fig.show()

"""**버블차트**"""

df = px.data.gapminder()
fig = px.scatter(df.query('year==2007'), x = 'gdpPercap', y = 'lifeExp', title = 'gdp에 따른 평균수명', size = 'pop', color = 'continent', hover_name = 'country', log_x = True, size_max = 60)
fig.show()

"""**버블차트(animation)**"""

df = px.data.gapminder()
px.scatter(df, x = 'gdpPercap', y = 'lifeExp', animation_frame = 'year', animation_group='country', size='pop', color='continent', 
           hover_name='country', log_x=True, size_max=55, range_x=[100, 100000], range_y = [25,90])

"""**파이차트**"""

df = px.data.gapminder().query('year==2007').query("continent=='Asia'")
fig = px.pie(df, values = 'pop', names='country', title='아시아 국가별 인구비율')
fig.show()

"""**히스토그램**"""

df = px.data.tips()
fig = px.histogram(df, x = 'total_bill', title = 'Total Bill', nbins=20, histnorm='probability density', 
                   labels={'total_bill':'Total Bill'}, opacity=0.7, color_discrete_sequence=['deepskyblue'])
fig.show()

df = px.data.tips()
fig = px.histogram(df, x = 'total_bill', title = 'Total Bill(sex)', nbins=20, histnorm='probability density', 
                   labels={'total_bill':'Total Bill'}, opacity=0.7, color='sex', color_discrete_sequence=['magenta','deepskyblue'])
fig.show()

"""**상자그림(boxplot)**"""

df = px.data.tips()
fig = px.box(df, x='day', y='total_bill')
fig.show()

df = px.data.tips()
fig = px.box(df, x='day', y='total_bill', points='all')
fig.show()

"""**바이올린 그래프**"""

df = px.data.tips()
fig = px.violin(df, y='tip', x='smoker', color='sex', box=True, points='all', hover_data=df.columns)
fig.show()

"""**radar chart**"""

df = pd.DataFrame(dict(r = [1,5,2,2,3], theta=['processing cost', 'mechanical properties', 'chemical stability', 'thermal stability', 'device integration']))
fig = px.line_polar(df, r = 'r', theta = 'theta', line_close=True)
fig.update_traces(fill='toself')
fig.show()

"""**funnel chart**"""

stages = ['Website visit', 'Downloads', 'Potential customers', 'Requested price', 'invoice sent']
df_mtl = pd.DataFrame(dict(number = [39,27.4,20.6,11,3], stage = stages))
df_mtl['office'] = 'Montreal'
df_toronto = pd.DataFrame(dict(number=[52,36,18,14,5], stage = stages))
df_toronto['office'] = 'Toronto'
df = pd.concat([df_mtl, df_toronto], axis = 0)
fig = px.funnel(df, x = 'number', y = 'stage', color = 'office')
fig.show()

"""**Ternary chart**"""

df = px.data.election()
fig = px.scatter_ternary(df, a = 'Joly', b = 'Coderre', c = 'Bergeron', hover_name='district', color='winner', 
                         size='total', size_max=15, color_discrete_map={'Joly':'Blue','Bergeron':'green','Coderre':'red'})
fig.show()
