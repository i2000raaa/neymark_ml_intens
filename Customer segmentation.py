#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Classification

# Автомобильная компания планирует выйти на новый рынок с имеющимся арсеналом товаров. После проведенного исследоваия стало понятно, что поведение нового рынка идентично поведению старого. При работе со старым рынком отдел продаж сегментировал всех покупателей на 4 группы "А", "В", "С", "D". Эта стратегия хорошо сработала и на новом рынке. Так как было выявлено 2627 новых клиентов, то требуется помочь менеджеру спрогнозировать правильную группу для этих клиентов.

# In[296]:


import hashlib as hs
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import preprocessing
import time


# In[297]:


pd.set_option('display.max_columns', 40)
pd.set_option('display.max_colwidth', 1)


# In[298]:


#already separated on train and test
data = pd.read_csv("Train.csv")
data.shape


# In[299]:


data.head()


# ID - уникальный ID пользователя <br>
# Gender - пол <br>
# Ever_Married - был ли человек замужем/женат <br>
# Age - возраст
# Graduated - является ли клиент выпускником <br>
# Profession - профессия  <br>
# Work_experience - опыт работы <br>
# Spending_score - оценка расходов клиента <br>
# Family_size - количество членов семьи <br>
# Var_1 - анонимная категория клиента <br>
# Segmentation - (target) сегмент клиента <br>

# In[300]:


data.info()


# In[301]:


data.describe()


# ### Визуализация данных

# Построим матрицу корреляции для числовых признаков. И на основе ее выведем heatmap.

# In[302]:


import numpy as np
import pandas as pd

# будем отображать графики прямо в jupyter'e
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# стиль seaborn
# style.available выводит все доступные стили
from matplotlib import style
style.use('seaborn')

#графики в svg выглядят более четкими
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

# отключим предупреждения Anaconda
import warnings
warnings.simplefilter('ignore')

corr = data.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True, fmt=".2f",
        linewidths=.5)


# По heatmap видно, что нет определённой зависимости между числовыми признаками. Построим диаграмму рассеивания для признаков "Age" и "Work_Experience", "Age" и "Family_Size".

# In[303]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = data['Age'], y = data['Work_Experience'])
plt.xlabel("Age")
plt.ylabel("Work_Experience")

plt.show()


# In[304]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x = data['Age'], y = data['Family_Size'])
plt.xlabel("Age")
plt.ylabel("Family_Size")

plt.show()


# In[305]:


data.Gender.value_counts().plot(kind='bar', rot=45) 
plt.show()


# In[306]:


sns.factorplot(x='Gender', hue='Segmentation', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# Видно, что большинство мужчин и женщин попадает в категорию "D", меньшая часть как мужчин, так и женщин - категорию "B". 

# In[307]:


data.Ever_Married.value_counts().plot(kind='bar', rot=45) 
plt.show()


# In[308]:


sns.factorplot(x='Segmentation', hue='Ever_Married', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# Есть определенная зависимость между семейным положением и сегментацией клиента. Большинство клиентов, относящихся к сегменту "D" не замужем/женаты, а большинство клиентов, относящихся к сегментам "B" и "C" замужем/женаты. В сегменте "A" нет такой явной зависимости, хотя так же пробладают замужние/женатые клиенты.

# In[309]:


data.Graduated.value_counts().plot(kind='bar', rot=45) 
plt.show()


# In[310]:


sns.factorplot(x='Segmentation', hue='Graduated', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# Также видна определенная зависимость между статусом выпускника и сегментацией клиента. Большинство клиентов, относящихся к сегменту "D" не является выпускниками, а большинство клиентов, относящихся к сегментам "B" и "C" являются выпускниками. В сегменте "A" нет такой явной зависимости, хотя так же пробладают выпускники.

# In[311]:


data.Profession.value_counts().plot(kind='bar', rot=45) 
plt.show()


# По гистограмме признака "Profession" видно, что преобладают такие профессии, как "Artist", "Healthcare". Будем это учитывать при рассмотрении распределения профессий по сегментам.

# In[312]:


sns.factorplot(x='Segmentation', hue='Profession', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# Несмотря на то, что клиентов с профессией "Healthcare" и "Artist" больше всего, данные клиенты распределены неравномерно по сегментам. Видно, что клиентов с профессией "Healthcare" чаще всего относят к категории "D", а клиенты с профессией "Artist" чаще всего относят в другие сегменты.

# In[313]:


data.Spending_Score.value_counts().plot(kind='bar', rot=45) 
plt.show()


# По гистограмме признака "Spending_Score" видно, что преобладает низкий уровень расходов. Будем это учитывать при рассмотрении распределения уровня расходов по сегментам.

# In[314]:


sns.factorplot(x='Segmentation', hue='Spending_Score', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# Видно, что в сегментах "D" и "A" заметно преобладают люди с низкими расходами, в сегментах "B" и "C" преобладают люди со средним и высоким уровнем расходов.

# In[315]:


data.Var_1.value_counts().plot(kind='bar', rot=45) 
plt.show()


# In[316]:


sns.factorplot(x='Var_1', hue='Segmentation', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# In[317]:


data.Segmentation.value_counts().plot(kind='bar', rot=45) 
plt.show()


# In[318]:


sns.factorplot(x='Segmentation', hue='Var_1', data=data, kind='count', size = 6).set_xticklabels(rotation=60)


# Видно, что распределение 7ми категорий клиентов примерно одинаковы в различных сегментах. Только в сегменте "C" количество покупателей из "Cat_6" значительно выше остальных категорий.

# ### Поиск дубликатов

# In[319]:


data.duplicated().sum()


# Дубликатов нет.

# ### Поиск пропущенных значений

# In[320]:


data_main = data.copy(deep = True)
data_main.head(10)


# In[321]:


A = data.isnull()
A.head()
print('Missing values:', A.sum(), sep='\n')


# #### Вариант 1: Заполнить средним пропущенные числовые значения, категориальные - удалить

# In[322]:


data_train = data_main.copy(deep = True)

A = data_train.isnull()
A.head()
print('Missing values:', A.sum(), sep='\n')


# In[323]:


data_train.fillna(round(data_train.mean()), inplace = True)


# In[324]:


data_train.dropna(axis=0,inplace=True)
data_train.isnull().sum()


# #### Вариант 2 (запасной, не способствовал улучшению качества предсказаний): заполнить пропущенные значения логически

# Чтобы не потерять данные, удалив все пропущенные значения, попробуем заполнить пропуски логическт. Скорее всего, если у человека пропущен статус "Ever_Married", то он не замужем/женат (т.е. заполним "No"), "Graduated" аналогично "No", "Profession"  удалим, "Work_Experience" заполним "0", "Family_Size" заполним "0", "Var_1" удалим.

# In[325]:


data = data_main.copy(deep = True)
data.isnull().sum()


# In[326]:


data['Var_1'].mode()


# In[327]:


data['Ever_Married'].fillna('No', inplace=True)


# In[328]:


data['Graduated'].fillna('No', inplace=True)


# In[329]:


#data.dropna(subset=["Profession"], inplace=True)
data['Profession'].fillna('None', inplace=True)


# In[330]:


data['Work_Experience'].fillna(0, inplace=True)


# In[331]:


data['Family_Size'].fillna(data_train['Family_Size'].mode, inplace=True)


# In[332]:


data['Var_1'].fillna("Cat_6", inplace=True)


# In[333]:


A = data.isnull()
A.head()
print('Missing values by features:', A.sum(), sep='\n')


# Проверим результат:

# In[334]:


fig, (ax,ax2) = plt.subplots(figsize=(10,5), ncols=2)
fig.subplots_adjust(wspace=0.01)

sns.heatmap(data_main.corr(), annot=True, square=True, ax=ax, cbar=False, linewidths=.5)
ax.set_title('Изначальные данные')

sns.heatmap(data_train.corr(), annot=True, square=True, ax=ax2, cbar=False, linewidths=.5)
ax2.yaxis.tick_right()
ax2.tick_params(axis='y', rotation=0)
ax2.set_title('Обработанные пропуски')

plt.show()


# ### Исключение нерелевантных признаков

# In[159]:


data_train.drop(["ID"], axis=1, inplace=True)


# In[160]:


data_train.reset_index(drop=True, inplace=True)


# ### Обработка категориальных признаков

# In[161]:


from sklearn.preprocessing import  LabelEncoder
le = LabelEncoder()

data_train['Gender'] = le.fit_transform(data_train['Gender'])
Gender_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Ever_Married'] = le.fit_transform(data_train['Ever_Married'])
Ever_Married_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Graduated'] = le.fit_transform(data_train['Graduated'])
Graduated_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Spending_Score'] = le.fit_transform(data_train['Spending_Score'])
Spending_Score_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Var_1'] = le.fit_transform(data_train['Var_1'])
Var_1_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Profession'] = le.fit_transform(data_train['Profession'])
Profession_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Family_Size'] = le.fit_transform(data_train['Family_Size'])
Family_Size_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Work_Experience'] = le.fit_transform(data_train['Work_Experience'])
Work_Experience_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Age'] = le.fit_transform(data_train['Age'])
Age_mapping = {l: i for i, l in enumerate(le.classes_)}
data_train['Segmentation'] = le.fit_transform(data_train['Segmentation'])
Segmentation_mapping = {l: i for i, l in enumerate(le.classes_)}

print(Gender_mapping, Ever_Married_mapping, Graduated_mapping, Spending_Score_mapping, Profession_mapping, Family_Size_mapping, Work_Experience_mapping, Var_1_mapping, Age_mapping, Segmentation_mapping)


# In[162]:


print("Age:", Age_mapping,"\n",
      "Gender:", Gender_mapping,"\n",
      "Ever_Married:", Ever_Married_mapping,"\n", 
      "Graduated:", Graduated_mapping,"\n",  
      "Spending_Score:", Spending_Score_mapping,"\n",  
      "Profession:", Profession_mapping,"\n",  
      "Family_Size:", Family_Size_mapping,"\n",  
      "Work_Experience:", Work_Experience_mapping,"\n",  
      "Var_1:", Var_1_mapping,"\n")


# In[163]:


dg = data_train.copy(deep=True)

data_train = data_train.drop('Segmentation', axis=1)


# ### Масштабирование данных

# Алгоритмы Decision Tree Classifier и Random Forest ищут наилучшую точку разделения в каждой функции, что определяется процентным соотношением правильно классифицированных меток. Масштабирование несущественно повлияет на качество этих моделей.

# In[164]:


#from sklearn import preprocessing

#min_max_scaler = preprocessing.MinMaxScaler()

#data_train = min_max_scaler.fit_transform(data_train)
#data_train = pd.DataFrame(data_train, columns=dg.drop('Segmentation', axis=1).columns)

#data_train.head(10)


# In[165]:


data_train.info()


# In[166]:


#data_train.Gender = data_train.Gender.astype('float64')
#data_train.Ever_Married = data_train.Ever_Married.astype('float64')
#data_train.Graduated = data_train.Graduated.astype('float64')
#data_train.Profession = data_train.Profession.astype('float64')
#data_train.Spending_Score = data_train.Spending_Score.astype('float64')
#data_train.Var_1 = data_train.Var_1.astype('float64')
#dg.Segmentation = dg.Segmentation .astype('category')


# ### Поиск выбросов

# Найдем выбросы с помощью расстояния Махалонобиса.

# In[167]:


def MahalanobisDist(y, data, cov=None):
  
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()


# In[168]:


Mahal_dist = MahalanobisDist(data_train, data_train)


# In[169]:


def MD_detectOutliers(data, extreme=False, verbose=False):
    MD = MahalanobisDist(y=data, data=data)
  

    std = np.std(MD)
    k = 4. * std if extreme else 3. * std
    m = np.mean(MD)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(MD)):
        if (MD[i] >= up_t) or (MD[i] <= low_t):
            outliers.append(i)  # index of the outlier
    return np.array(outliers)


# In[170]:


outliers = np.array(data_train)
outliers_indices = MD_detectOutliers(data_train, verbose=True)

print(len(outliers_indices))


# In[171]:


data_train.drop(outliers_indices, inplace=True) # axis=0 will do for rows по умолчанию


# In[172]:


data_train.reset_index(drop=True, inplace=True)


# In[173]:


dg.drop(outliers_indices, inplace=True)


# In[174]:


dg.reset_index(drop=True, inplace=True)


# ### Повторная визуализация

# In[175]:


plt.figure(figsize=(10,5), dpi= 80)
sns.pairplot(data_train, diag_kind="kde", corner=True, height=3)
plt.show()


# Посмотрим на зависимости признаков.

# In[335]:


fig, (ax,ax2) = plt.subplots(figsize=(10,5), ncols=2)
fig.subplots_adjust(wspace=0.01)

sns.heatmap(data_main.corr(), annot=True, square=True, ax=ax, cbar=False, linewidths=.5)
ax.set_title('Изначальные данные')

sns.heatmap(data_train.corr(), annot=True, square=True, ax=ax2, cbar=False, linewidths=.5)
ax2.yaxis.tick_right()
ax2.tick_params(axis='y', rotation=0)
ax2.set_title('Обработанные данные')

plt.show()


# Явно зависимых признаков нет.

# In[177]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[244]:


(trainData, testData, train_label, test_label) = train_test_split(data_train,
                                            dg['Segmentation'],
                                            test_size=0.3, 
                                            random_state=5)


# ### Decision Tree

# Плюсом Decision Tree является прозрачность, ведь в каждом узле видно, какое решение принимает наша модель, откуда исходят ошибки и как значения признаков влияют на выход. 
# Не требует объемной подготовки данных. Стоимость использования дерева для вывода является логарифмической от числа точек данных, используемых для обучения дерева. 

# In[179]:


from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz


# In[180]:


dct = DecisionTreeClassifier()
dct.fit(trainData,train_label)
predictedDependentVariables = dct.predict(testData)
print(f"Accuracy: {(accuracy_score(test_label,predictedDependentVariables)*100).round(2)} %")


# In[181]:


from sklearn.model_selection import RepeatedStratifiedKFold
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]


# In[182]:


for train_index, test_index in splits:
    dtree_model = DecisionTreeClassifier(max_depth=5)
    dtree_model.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
    preds = dtree_model.predict(data_train.iloc[test_index])
    print("Accuracy:",
    round(metrics.accuracy_score(dg['Segmentation'].iloc[test_index], preds), 5))
    dot_data = export_graphviz(dtree_model, 
                           out_file=None, 
                           feature_names=data_train.columns, 
                           class_names=['1', '2', '3', '4'],
                           filled=True, 
                           rounded=True, 
                           special_characters=True)

    graph = graphviz.Source(dot_data)
    display(graph)


# ### KNN

# KNN - алгоритм довольно простой в реализации, но позволяющий решать сложные задачи классификации. Он использует все данные для обучения при классификации нового экземпляра данных, а также не выдвигает каких-либо предположений о базовых данных (их распределение, линейную разделимость, чего в большинстве данных и не имеется).

# In[183]:


from sklearn.neighbors import KNeighborsClassifier
import sklearn


# In[239]:


KNN_model = KNeighborsClassifier(n_neighbors=5)


# In[240]:


KNN_model.fit(trainData, train_label)


# In[241]:


KNN_preds = KNN_model.predict(testData)


# In[246]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print("Accuracy score: ", np.round(accuracy_score(test_label, KNN_preds), 5), "\n",
     "Recall score: ", np.round(recall_score(test_label, KNN_preds, average="weighted"), 5), "\n",
     "Precision score: ", np.round(precision_score(test_label, KNN_preds, average="weighted"), 5), "\n",
     "F1-score score: ", np.round(f1_score(test_label, KNN_preds, average="weighted"), 5))


# ### Random Forest

# Выбор пал на Random Forest Classifier, потому что он подходит для многоклассовой классификации, имеет высокую точность предсказания, не чувствителен к выбросам и масштабированию, зачастую качество его предсказаний выше чем у линейных алгоритмов и он не требует тщательной настройки параметров.

# In[247]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[248]:


rf_model = RandomForestClassifier(n_estimators=1000)
rf_model.fit(trainData,train_label)
rf_preds = rf_model.predict(testData)
print("Accuracy of Random Forest:", (accuracy_score(test_label,rf_preds)*100).round(2))


# In[249]:


print("Accuracy score: ", np.round(accuracy_score(test_label, rf_preds), 5), "\n",
     "Recall score: ", np.round(recall_score(test_label, rf_preds, average="weighted"), 5), "\n",
     "Precision score: ", np.round(precision_score(test_label, rf_preds, average="weighted"), 5), "\n",
     "F1-score score: ", np.round(f1_score(test_label, rf_preds, average="weighted"), 5))


# Так как Random Forest показал лучший Accuracy, будем дальше его использовать для настройки гиперпараметров.

# ### Настройка гиперпараметров

# In[250]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[251]:


rf = RandomForestClassifier()


# In[252]:


n_estimators = [i for i in range(1, 200, 5)]
accuracies = []
accuracies_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_est in n_estimators:
    clf = RandomForestClassifier(n_estimators=n_est)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        clf.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = clf.predict(data_train.iloc[test_index])
        preds_train = clf.predict(data_train.iloc[train_index])
        acc.append(metrics.accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(metrics.accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracies.append(np.mean(acc))
    accuracies_train.append(np.mean(acc_train))


# In[253]:


plt.plot(n_estimators, accuracies, label='test')
plt.plot(n_estimators, accuracies_train, label='train')
plt.legend()
plt.show()


# In[254]:


max_features = ["auto", "sqrt", "log2"]
accuracies = []
accuracies_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in max_features:
    clf = RandomForestClassifier(max_features=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        clf.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = clf.predict(data_train.iloc[test_index])
        preds_train = clf.predict(data_train.iloc[train_index])
        acc.append(metrics.accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(metrics.accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracies.append(np.mean(acc))
    accuracies_train.append(np.mean(acc_train))


# In[255]:


plt.plot(max_features, accuracies, label='test')
plt.plot(max_features, accuracies_train, label='train')
plt.legend()
plt.show()


# In[256]:


max_depth = list([i for i in range(1, 20)])
accuracies = []
accuracies_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in max_depth:
    clf = RandomForestClassifier(max_depth=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        clf.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = clf.predict(data_train.iloc[test_index])
        preds_train = clf.predict(data_train.iloc[train_index])
        acc.append(metrics.accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(metrics.accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracies.append(np.mean(acc))
    accuracies_train.append(np.mean(acc_train))


# In[257]:


plt.plot(max_depth, accuracies, label='test')
plt.plot(max_depth, accuracies_train, label='train')
plt.legend()
plt.show()


# In[258]:


min_samples_split = [2, 5, 10]
accuracies = []
accuracies_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in min_samples_split:
    clf = RandomForestClassifier(min_samples_split=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        clf.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = clf.predict(data_train.iloc[test_index])
        preds_train = clf.predict(data_train.iloc[train_index])
        acc.append(metrics.accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(metrics.accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracies.append(np.mean(acc))
    accuracies_train.append(np.mean(acc_train))


# In[259]:


plt.plot(min_samples_split, accuracies, label='test')
plt.plot(min_samples_split, accuracies_train, label='train')
plt.legend()
plt.show()


# In[260]:


#min_samples_leaf = [1, 2, 3, 4, 5]
min_samples_leaf = [i for i in range(1, 63, 5)]
accuracies = []
accuracies_train = []
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
splits = [(i, j) for (i,j) in skf.split(data_train, dg['Segmentation'])]
for n_feat in min_samples_leaf:
    clf = RandomForestClassifier(min_samples_leaf=n_feat)
    acc = []
    acc_train = []
    for train_index, test_index in splits:
        clf.fit(data_train.iloc[train_index], dg['Segmentation'].iloc[train_index])
        preds = clf.predict(data_train.iloc[test_index])
        preds_train = clf.predict(data_train.iloc[train_index])
        acc.append(metrics.accuracy_score(dg['Segmentation'].iloc[test_index], preds))
        acc_train.append(metrics.accuracy_score(dg['Segmentation'].iloc[train_index], preds_train))
    accuracies.append(np.mean(acc))
    accuracies_train.append(np.mean(acc_train))


# In[261]:


plt.plot(min_samples_leaf, accuracies, label='test')
plt.plot(min_samples_leaf, accuracies_train, label='train')
plt.legend()
plt.show()


# Подберем оптимальные параметры при помощи GridSearchCV.

# ### GridSearchCV

# In[262]:


pipeline = Pipeline([
    ('algo',RandomForestClassifier(n_jobs=-1,random_state=42))
])

param_rf = {
    'algo__n_estimators':[200],
    'algo__max_depth':[6],
    'algo__max_features':[0.5],
    'algo__min_samples_leaf':[63],
    'algo__class_weight':[{0:0.34,
                           1:0.48,
                           2:0.39,
                           3:0.2}]
}


# In[263]:


cv_rs = GridSearchCV(pipeline,param_rf,cv=10,n_jobs=-1,verbose=1)
cv_rs.fit(trainData, train_label)

print(cv_rs.best_params_)
print("Train data accuracy score: ", cv_rs.score(trainData,train_label))
print("Test data accuracy score: ", cv_rs.score(testData,test_label))


# In[264]:


gridcv_preds = cv_rs.predict(testData)


# In[265]:


from collections import Counter

Counter(gridcv_preds).keys()
Counter(gridcv_preds).values() 


# Классы сбалансированы.

# In[267]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print("Accuracy score: ", np.round(accuracy_score(test_label, gridcv_preds), 5), "\n",
     "Recall score: ", np.round(recall_score(test_label, gridcv_preds, average="weighted"), 5), "\n",
     "Precision score: ", np.round(precision_score(test_label, gridcv_preds, average="weighted"), 5), "\n",
     "F1-score score: ", np.round(f1_score(test_label, gridcv_preds, average="weighted"), 5))


# Вышло, что подобранные оптимальные параметры при помощи GridSearchCV совпадают с теми, что были видны на графиках. Качество модели улучшилось, но не сильно.

# ###  XGBoost

# In[268]:


get_ipython().run_line_magic('pylab', 'inline')
from xgboost import XGBClassifier


# In[269]:


xgb_model = XGBClassifier(objective='multi:softmax')


# In[270]:


xgb_model.fit(trainData, train_label, early_stopping_rounds=5, 
             eval_set=[(testData, test_label)], verbose=False)


# In[271]:


xgb_preds = xgb_model.predict(testData)


# In[273]:


print("Accuracy score: ", np.round(accuracy_score(test_label, xgb_preds), 5), "\n",
     "Recall score: ", np.round(recall_score(test_label, xgb_preds, average="weighted"), 5), "\n",
     "Precision score: ", np.round(precision_score(test_label, xgb_preds, average="weighted"), 5), "\n",
     "F1-score score: ", np.round(f1_score(test_label, xgb_preds, average="weighted"), 5))


# Видим, что метрики не сильно отличаются друг от друга, однако, качество модели оставляет желать лучшего.

# ### Тестовые данные

# In[274]:


trainData = data_train
train_label = dg['Segmentation']
test = pd.read_csv('Test.csv')


# In[275]:


test.info()


# In[276]:


test.head()


# ### Обработка тестовых данных

# In[277]:


test.Segmentation.value_counts().plot(kind='bar', rot=45) 
plt.show()


# In[278]:


A = test.isnull()
A.head()
print('Missing values by features:', A.sum(), sep='\n')


# In[279]:


data_test = test.copy(deep = True)


# In[280]:


data_test.fillna(round(data_test.mean()), inplace = True)


# In[281]:


data_test.dropna(axis=0,inplace=True)
data_test.isnull().sum()


# In[282]:


le = LabelEncoder()

data_test['Gender'] = le.fit_transform(data_test['Gender'])
Gender_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Ever_Married'] = le.fit_transform(data_test['Ever_Married'])
Ever_Married_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Graduated'] = le.fit_transform(data_test['Graduated'])
Graduated_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Spending_Score'] = le.fit_transform(data_test['Spending_Score'])
Spending_Score_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Var_1'] = le.fit_transform(data_test['Var_1'])
Var_1_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Profession'] = le.fit_transform(data_test['Profession'])
Profession_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Family_Size'] = le.fit_transform(data_test['Family_Size'])
Family_Size_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Work_Experience'] = le.fit_transform(data_test['Work_Experience'])
Work_Experience_mapping = {l: i for i, l in enumerate(le.classes_)}
data_test['Segmentation'] = le.fit_transform(data_test['Segmentation'])
Segmentation_mapping = {l: i for i, l in enumerate(le.classes_)}


# In[283]:


print("Ever_Married:", Ever_Married_mapping,"\n", 
      "Graduated:", Graduated_mapping,"\n",  
      "Spending_Score:", Spending_Score_mapping,"\n",  
      "Profession:", Profession_mapping,"\n",  
      "Family_Size:", Family_Size_mapping,"\n",  
      "Work_Experience:", Work_Experience_mapping,"\n",  
      "Var_1:", Var_1_mapping,"\n",  
      "Segmentation:", Segmentation_mapping)


# In[284]:


dg_2 = data_test.copy(deep=True)
data_test = data_test.drop('Segmentation', axis=1)


# In[ ]:


#from sklearn import preprocessing

#min_max_scaler = preprocessing.MinMaxScaler()

#data_test = min_max_scaler.fit_transform(data_test)
#data_test = pd.DataFrame(data_test, columns=dg_2.drop('Segmentation', axis=1).columns)

#data_test.head(10)


# In[ ]:


#data_test.Gender = data_test.Gender.astype('float64')
#data_test.Ever_Married = data_test.Ever_Married.astype('float64')
#data_test.Graduated = data_test.Graduated.astype('float64')
#data_test.Profession = data_test.Profession.astype('float64')
#data_test.Spending_Score = data_test.Spending_Score.astype('float64')
#data_test.Var_1 = data_test.Var_1.astype('float64')
#dg_2.Segmentation = dg_2.Segmentation .astype('category')


# In[285]:


data_test.drop(["ID"], axis=1, inplace=True)
data_test.reset_index(drop=True, inplace=True)
data_test


# In[286]:


test_label = dg_2.Segmentation


# ### RandomForestClassifier on Test

# In[289]:


param_grid = {'n_estimators': [i for i in range(1, 200, 5)], 'max_features' : ["auto", "sqrt", "log2"],
'max_depth': [i for i in range(1, 20)], 'min_samples_split' : [2, 5, 10],
'min_samples_leaf': [1, 63, 5]}


# In[290]:


start = time.time()
#cv_rs = RandomizedSearchCV(rf, param_distributions = param_grid, cv = 5, verbose=True, n_jobs=-1,  error_score=0.0)
cv_rs = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=100, cv=10, n_jobs=-1)
cv_rs.fit(trainData, train_label)
end=time.time()
print(end-start)


# In[291]:


print(cv_rs.best_params_)


# In[292]:


cv_rs.fit(trainData, train_label)


# In[293]:


preds = cv_rs.predict(data_test)


# In[294]:


print('F1-Score:', metrics.f1_score(test_label, preds, average="weighted"), 'Accuracy: ', metrics.accuracy_score(test_label, preds)) #score ensemble


# In[295]:


print("Accuracy score: ", np.round(accuracy_score(test_label, preds), 5), "\n",
     "Recall score: ", np.round(recall_score(test_label, preds, average="weighted"), 5), "\n",
     "Precision score: ", np.round(precision_score(test_label, preds, average="weighted"), 5), "\n",
     "F1-score score: ", np.round(f1_score(test_label, preds, average="weighted"), 5))


# ### GridSearchCV on Test

# In[69]:


model = GridSearchCV(pipeline,param_rf,cv=5,n_jobs=-1,verbose=1)
model.fit(trainData,train_label)

print(model.best_params_)
print("Train data accuracy score: ", model.score(trainData,train_label))
print("Test data accuracy score: ", model.score(data_test,test_label))


# In[71]:


y_test_pred = model.predict(data_test)


# In[73]:


print("Test data prediction accuracy score: ", model.score(data_test,test_label))


# In[524]:


from collections import Counter

Counter(y_test_pred).keys()
Counter(y_test_pred).values() 


# Видно, что классы сбалансированы. Выбранная модель Random Forest с использованием GridSearchCV показала лучший accuracy score на тестовых данных. Значения метрик могли бы быть выше при наличии большей информации о покупателях.
