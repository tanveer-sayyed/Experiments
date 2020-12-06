#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:03:53 2019

@author: tanveer
"""
"""Task:
    Find farthest points from a given cluster.
    This approach will work with any number of clusters."""

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

n = 100
l1 = np.random.normal(size= n)
l2 = np.random.normal(size= n)
df = pd.DataFrame(np.array([l1, l2]).T, columns= ['x1', 'x2'])
df['label'] = pd.Series().astype(object)
# preparing the df
all_idx = list(df.index)
idx = random.sample(population= all_idx, k= 33)
df['label'].loc[idx] = 0
all_idx = set(all_idx) - set(idx)
idx = random.sample(population= all_idx, k= 33)
df['label'].loc[idx] = 1
all_idx = set(all_idx) - set(idx)
idx = random.sample(population= all_idx, k= 34)
df['label'].loc[idx] = 2

# Observe the range of the data
plt.plot(df['x1'])
plt.plot(df['x2'])
plt.legend()
plt.show()

# A look at our HORRIBLE unclusterable data
# This data has been sparsely distributed ON PURPOSE
sns.scatterplot(x='x1', y= 'x2', hue= 'label', data= df, palette="Set1")
plt.title('Data')
plt.legend(loc = 'upper left')
plt.show()


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(1)
knc.fit(df[['x1', 'x2']], df['label'])

# The MORE clustered the data, the BETTER this function works
def non_neighbour(x1, x2):
    label_proba = pd.Series(knc.predict_proba([[x2,x2]])[0])
    idx = label_proba.idxmax()
    label_proba[idx] = 1 # the point(x1, x2) belonged to this label
    label_proba = 1 - label_proba # REVERSING the probabilities so that the highest
                                  # proba now means the farthest label from that point
    points = 20
    result = pd.DataFrame()

    for i in range(len(label_proba)):
        result = pd.concat( [result, df[df['label'] == i].sample(n= np.int(points * label_proba.iloc[i] / sum(label_proba)))])

    sns.scatterplot(x='x1', y= 'x2', hue= 'label', data= result, palette="Set1")
    plt.scatter(x1, x2, color= 'black') # Our point is HERE
    plt.title(str(x1)+', '+str(x2)+' belonged to label: '+str(idx))
    plt.show()

non_neighbour(1,-1)
non_neighbour(-1,1)
non_neighbour(0,0)
non_neighbour(-1.8, -1.8)
non_neighbour(-1.5, 0.5)
non_neighbour(0, -1.5)
non_neighbour(-0.8, 0.8)
non_neighbour(-1, -2.3)
non_neighbour(-2, -2.3)
non_neighbour(0, 2.3)
