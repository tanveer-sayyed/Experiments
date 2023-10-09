#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:38:07 2019

@author: tanveer
"""
import pandas as pd
import numpy as np
np.random.seed = 0
import random
random.seed = 0
import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split as tts, GridSearchCV, cross_val_score


"""-------------------------------------------------------------------------"""
valueCounts = {}
def CountAll():
    global all_columns, nanCounts, valueCounts, nanPercent
    all_columns = list(df)
    nanCounts = df.isnull().sum()
    nanPercent = nanCounts / len(df) * 100
    for x in all_columns:
        valueCounts[x] = df[x].value_counts()

"""-------------------------------------------------------------------------"""
def Drop(cols):
    df.drop(columns=cols, inplace= True)
    if type(cols) == str:
        _ = valueCounts.pop(cols)
    else: # cols is a list
        for col in cols:
            _ = valueCounts.pop(col)
    CountAll()

"""-------------------------------------------------------------------------"""
"""Random but proportional replacement(RBPR) of categoricals."""
def Fill_NaNs_Catigorical(col):
    """Calculating proportion."""
    proportion = np.array(valueCounts[col].values) / valueCounts[col].sum() * nanCounts[col]
    proportion = np.around(proportion).astype('int')

    """Adjusting proportion."""
    diff = int(nanCounts[col] - np.sum(proportion))
    if diff > 0:
        for x in range(diff):
            idx = random.randint(0, len(proportion) - 1)
            proportion[idx] =  proportion[idx] + 1
    else:
        diff = -diff
        while(diff != 0):
            idx = random.randint(0, len(proportion) - 1)
            if proportion[idx] > 0:
                proportion[idx] =  proportion[idx] - 1
                diff = diff - 1

    """Filling NaNs."""
    nan_indexes = df[df[col].isnull()].index.tolist()
    for x in range(len(proportion)):
        if proportion[x] > 0:
            random_subset = random.sample(population = nan_indexes, k = proportion[x])
            df.loc[random_subset, col] = valueCounts[col].keys()[x]
            nan_indexes = list(set(nan_indexes) - set(random_subset))


"""-------------------------------------------------------------------------"""
"""Random but proportional replacement(RBPR) of numeric"""
def Fill_NaNs_Numeric(col):

    mini = df[col].min()
    maxi = df[col].max()
    """Selecting ONLY non-NaNs."""
    temp = df[df[col].notnull()][col] # type --> pd.Series

    """Any continuous data is 'always' divided into 45 bins (Hard-Coded)."""
    bin_size = 45
    bins = np.linspace(mini, maxi, bin_size)

    """Filling the bins (with non-NaNs) and calculating mean of each bin."""
    non_NaNs_per_bin = []
    mean_of_bins = []

    non_NaNs_per_bin.append(len(temp[(temp <= bins[0])]))
    mean_of_bins.append(temp[(temp <= bins[0])].mean())
    for x in range(1, bin_size):
        non_NaNs_per_bin.append(len(temp[(temp <= bins[x]) & (temp > bins[x-1])]))
        mean_of_bins.append(temp[(temp <= bins[x]) & (temp > bins[x-1])].mean())

    mean_of_bins = pd.Series(mean_of_bins)
    # np.around() on  list 'proportion' may create trouble and we may get a zero-value imputed, hence,
    mean_of_bins.fillna(temp.mean(), inplace= True)
    non_NaNs_per_bin = np.array(non_NaNs_per_bin)

    """Followoing part is SAME as Fill_NaNs_Catigorical()"""

    """Calculating proportion."""
    proportion = np.array(non_NaNs_per_bin) / valueCounts[col].sum() * nanCounts[col]
    proportion = np.around(proportion).astype('int')

    """Adjusting proportion."""
    diff = int(nanCounts[col] - np.sum(proportion))
    if diff > 0:
        for x in range(diff):
            idx = random.randint(0, len(proportion) - 1)
            proportion[idx] =  proportion[idx] + 1
    else:
        diff = -diff
        while(diff != 0):
            idx = random.randint(0, len(proportion) - 1)
            if proportion[idx] > 0:
                proportion[idx] =  proportion[idx] - 1
                diff = diff - 1

    """Filling NaNs."""
    nan_indexes = df[df[col].isnull()].index.tolist()
    for x in range(len(proportion)):
            if proportion[x] > 0:
                random_subset = random.sample(population= nan_indexes, k= proportion[x])
                df.loc[random_subset, col] = mean_of_bins[x] # <--- Replacing with bin mean
                nan_indexes = list(set(nan_indexes) - set(random_subset))

"""-------------------------------------------------------------------------"""

df = pd.read_csv('startup_funding.csv')
CountAll()
Drop(['SNo','Remarks'])

plt.title('% NaN')
nanPercent.plot(kind= 'bar')
plt.xticks(rotation= 60)
plt.show()

cols = list(nanCounts[nanCounts > 0].index)
for col in cols[:-1]: # Removing 'AmountInUSD' as it is a numeric/continuous
    Fill_NaNs_Catigorical(col)

for x in all_columns[1:-1]:
    df[x] = df[x].str.lower()
del x

df['Date'] = df['Date'].str.replace(',','/')
df['Date'] = df['Date'].str.replace('.','/')
df['Date'] = df['Date'].str.replace('//','/')
df['Date'] = pd.to_datetime(df['Date']).dt.date
today = datetime.date.today()
df['Days'] = (today - df['Date']).dt.days
del today
Drop('Date')


df['IndustryVertical'] = df["IndustryVertical"].str.split(" ", n = 1, expand = True)[0]
df['InvestorsName'] = df["InvestorsName"].str.split(" ", n = 1, expand = True)[0]
df['SubVertical'] = df["SubVertical"].str.split(" ", n = 1, expand = True)[0]
df['AmountInUSD'] = df['AmountInUSD'].str.replace(',', '').astype('float')


req_cities = list(valueCounts['CityLocation'][valueCounts['CityLocation'] > 4].keys())
all_cities = list(df['CityLocation'].values)
for city in all_cities:
    if city not in req_cities:
        all_cities[all_cities.index(city)] = 'other cities'
df['CityLocation'] = all_cities
del req_cities, all_cities, city
CountAll() # --> see CityLocation in valueCounts


valueCounts['IndustryVertical'] = df['IndustryVertical'].value_counts()
req_industries = list(valueCounts['IndustryVertical'][valueCounts['IndustryVertical'] > 10].keys())
all_industries = list(df['IndustryVertical'].values)
for industry in all_industries:
    if industry not in req_industries:
        all_industries[all_industries.index(industry)] = 'other industries'
df['IndustryVertical'] = all_industries
del req_industries, all_industries, industry
CountAll() # --> see IndustryVertical in valueCounts


valueCounts['InvestorsName'] = df['InvestorsName'].value_counts()
req_names = list(valueCounts['InvestorsName'][valueCounts['InvestorsName'] > 12].keys())
all_names = list(df['InvestorsName'].values)
for name in all_names:
    if name not in req_names:
        all_names[all_names.index(name)] = 'other investors'
df['InvestorsName'] = all_names
del req_names, all_names, name
CountAll() # --> see InvestorsName in valueCounts


#req_names = list(valueCounts['StartupName'][valueCounts['StartupName'] > 2].keys())
#all_names = list(df['StartupName'].values)
#for name in all_names:
#    if name not in req_names:
#        all_names[all_names.index(name)] = 'other start-ups'
#df['StartupName'] = all_names
#del req_names, all_names, name
#CountAll() # --> see StartupName in valueCounts
Drop(['StartupName']) #--> Too many unique names.


valueCounts['SubVertical'] = df['SubVertical'].value_counts()
req_subV = list(valueCounts['SubVertical'][valueCounts['SubVertical'] > 10].keys())
all_subV = list(df['SubVertical'].values)
for subV in all_subV:
    if subV not in req_subV:
        all_subV[all_subV.index(subV)] = 'other sub-verticals'
df['SubVertical'] = all_subV
del req_subV, all_subV, subV
#CountAll() # --> see SubVertical in valueCounts


# df['AmountInUSD'].mean() = 12031073.099016393 --> Before filling nans
Fill_NaNs_Numeric(cols[-1]) # 'AmountInUSD'
# df['AmountInUSD'].mean() = 11999228.0100382 --> After filling nans
# %Change in mean = -0.265%
CountAll()
del col, cols


import matplotlib.pyplot as plt
#%matplotlib inline
plt.figure(figsize=(8,35))
for i in range(len(all_columns)):
    plt.subplot(7,1,i+1)
    plt.scatter(df[all_columns[i]], df['AmountInUSD'], color=plt.cm.Paired(i/10.))
    plt.xticks(rotation = 80)
    plt.xlabel(all_columns[i])
    plt.ylabel('AmountInUSD')
    plt.title('target with ' + str(all_columns[i]))
    plt.legend(loc=2)
plt.tight_layout()
plt.show()

def BarsH(col, no_of_bars):
    ser = pd.Series(data=valueCounts[col].values[0:no_of_bars],
                index= valueCounts[col].keys()[0:no_of_bars])
    #ser = pd.Series(data=valueCounts['AmountInUSD'].values,
    #                index= valueCounts['AmountInUSD'].keys())
    ser.sort_index(inplace= True)
    fig, ax = plt.subplots(figsize= (7,7))
    ax = ser.plot(kind= 'barh')
    ax.set_xlabel('Frequency')
    ax.set_ylabel(str(col))
    ax.set_yticklabels(ser.index, rotation=0)
    for i, v in enumerate(ser.values):
        ax.text(v + 3, i + .25, str(v), color='black', va='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

#BarsH('AmountInUSD', 10)

for x in df.columns:
    df[x] = df[x].map(df.groupby(x)['AmountInUSD'].mean())
CountAll()

"""Classifying data as <3848035.5 or >3848035.5"""
#bin_size = 45
#bins = np.linspace(mini, maxi, bin_size)

bins = [3848035.5]
df['target'] = pd.Series(np.digitize(df['AmountInUSD'], bins, right=True), index= df.index)
CountAll()
BarsH('target', 3)
BarsH('AmountInUSD', 10)
numerical = list(df.columns[df.dtypes != object])
categorical = list(df.columns[df.dtypes == object])
df[categorical] = df[categorical].apply(lambda x: pd.factorize(x)[0])


#df_mean_encoded = pd.concat([X,y],1)
#for x in list(df_mean_encoding.drop(['click'],1)):
#    df_mean_encoded[x] = df_mean_encoded[x].map(df_mean_encoding.groupby(x)['click'].mean())


train_ind = random.sample(population = list(df.index), k = 1660)
test_ind = list(set(df.index) - set(train_ind))

X_train = df.iloc[train_ind].drop(columns= ['AmountInUSD','target'])
X_test = df.iloc[test_ind].drop(columns= ['AmountInUSD','target'])

y_train = df.iloc[train_ind]['target']
y_test = df.iloc[test_ind]['target']


#from sklearn.feature_selection import chi2, f_classif, SelectKBest
#selector = SelectKBest(score_func=chi2, k=3)
#selector.fit(X_train, y_train)
#selector.get_support(indices=True)
#
#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(n_estimators=400, n_jobs=-1,
#                            min_samples_split=0.12, random_state=42)
#
#rfc.fit(X_train,y_train)
#fi = pd.Series(rfc.feature_importances_,index= X_train.columns)
#fi = fi.sort_values(ascending = True)
#fi.plot(kind = "barh")
X = df.drop(columns=['AmountInUSD', 'target'])
y = df['target']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=0)
dtc = DecisionTreeClassifier(max_depth=3)
classifier = dtc.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
predict_proba = classifier.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))

z = pd.Series(index= y_test.index)

for x in range(len(y_test)):

#    ind = np.argmax(predict_proba[x])
#    z.iloc[x] =  predict_proba[x][ind] * bins[ind]
    if y_test.iloc[x] == 0:
        z.iloc[x] = y_pred[x][0] * bins[0]
    else:
        z.iloc[x] = (1+y_pred[x][1]) * bins[0]
#
r2_score(df.iloc[test_ind]['AmountInUSD'], z)
#list(zip(df.iloc[test_ind]['AmountInUSD'], z))
#
#plt.scatter(np.arange(len(df.iloc[test_ind]['AmountInUSD'])), df.iloc[test_ind]['AmountInUSD'],color = 'r')
#plt.scatter(np.arange(len(z)), z,color = 'b')
#
#plt.figure(figsize=(8,8))
#plt.scatter(np.arange(len(predict_proba[:,0])), predict_proba[:,0],color = 'b', marker= '.')
#plt.scatter(np.arange(len(predict_proba[:,1])), predict_proba[:,1],color = 'r', marker= '.')
#plt.scatter(np.arange(len(predict_proba[:,2])), predict_proba[:,2],color = 'g', marker= '.')
#plt.show()


from sklearn.neighbors import KNeighborsClassifier
knnC = KNeighborsClassifier()
y_pred = knnC.fit(X_train, y_train).predict(X_test)
y_pred = knnC.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
