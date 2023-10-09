#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:09:29 2019

@author: tanveer
"""

from yellowbrick.features import RadViz, Rank1D, Rank2D, ParallelCoordinates
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.manifold import Manifold
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_boston
from matplotlib import pyplot as plt

iris = load_iris()
boston = load_boston()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df_ = pd.DataFrame(boston.data, columns= boston.feature_names)
df_['target'] = boston.target
df['target'] = iris.target
classes = list(iris.target_names)
features = list(iris.feature_names)
X = df.drop(columns= 'target')
y = df['target']
X_ = df_.drop(columns= 'target')
y_ = df_['target']
classes_ = 'target'
featrues_ = list(boston.feature_names)


"""detect separability between classes"""
visualizer = RadViz(classes= classes, features= features, alpha= 0.4)
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof()


"""a ranking algorithm that takes into account only a single feature at a time"""
viz = Rank1D(features=list(iris.feature_names), algorithm='shapiro')
viz.fit(X, y)
viz.transform(X)
viz.poof()


""" to detect clusters of instances that have similar classes, after normalisation
and to note features that have high variance or different distributions"""


viz = ParallelCoordinates(classes=classes, features=features,
                          sample=0.99, shuffle=False, normalize='standard')
# for HUGE datasets:
#viz = ParallelCoordinates(classes=classes, features=features,
#                          sample=0.2, shuffle=True, normalize='standard',
#                          fast=True)
viz.fit_transform(X, y)
viz.poof()


"""visualizer utilizes principal component analysis to decompose high dimensional
data into two or three dimensions so that each instance can be plotted in a scatter
plot"""
colors= plt.rainbow(np.linspace(0,1,y.unique()))
colors = y.replace(to_replace= [0,1,2], value= ['r', 'y', 'g'])
viz = PCADecomposition(scale=True)#, color=[0.22, 0.33, 0.44])
viz.fit_transform(X, y)
viz.poof()
"""in 3-D"""
viz = PCADecomposition(scale=True, color=colors, proj_dim=3)
viz.fit_transform(X, y)
viz.poof()

pca_decomposition_components =\
                                pd.DataFrame(
                                  data= viz.fit_transform(X, y),
                                  columns= [name+'_0',name+'_1'],
                                  dtype= float)

viz = PCADecomposition(scale=True,  proj_dim=3)
viz.fit_transform(X_, y_)
viz.poof()


"""The PCA projection can be enhanced to a biplot whose points are the projected
instances and whose vectors represent the structure of the data in high dimensional
space. By using the proj_features=True flag, vectors for each feature in the dataset
are drawn on the scatter plot in the direction of the maximum variance for that
feature. These structures can be used to analyze the importance of a feature to
the decomposition or to find features of related variance for further analysis."""
viz = PCADecomposition(scale=True, proj_features=True)
viz.fit_transform(X, y)
viz.poof()
"""in 3-D"""
viz = PCADecomposition(scale=True, proj_features=True, proj_dim=3)
viz.fit_transform(X, y)
viz.poof()

viz = PCADecomposition(scale=True, proj_features=True, proj_dim=3)
viz.fit_transform(X_, y_)
viz.poof()


"""For more : --> https://www.scikit-yb.org/en/latest/api/index.html """

visualizer = Manifold(manifold='tsne', target='discrete')
visualizer.fit_transform(X,y)
visualizer.poof()

""" allowing the creation of a scatter plot that shows latent structures in data.
Unlike decomposition methods such as PCA and SVD, manifolds generally use nearest-
neighbors approaches to embedding, allowing them to capture non-linear structures
that would be otherwise lost. The projections that are produced can then be analyzed
for noise or separability to determine if it is possible to create a decision
space in the data."""
"""This takes time. One tip is scale your data using the StandardScalar; another
is to sample your instances (e.g. using train_test_split to preserve class
stratification) or to filter features to decrease sparsity in the dataset."""
visualizer = Manifold(manifold='isomap', target='continuous')
visualizer.fit_transform(X_,y_)
visualizer.poof()

"""The feature engineering process involves selecting the minimum required
features to produce a valid model because the more features a model contains, the
more complex it is (and the more sparse the data), therefore the more sensitive
the model is to errors due to variance. """
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from yellowbrick.features.importances import FeatureImportances
fig = plt.figure()
ax = fig.add_subplot()
viz = FeatureImportances(GradientBoostingClassifier(), ax=ax)
viz.fit(X, y)
viz.poof()

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
fig = plt.figure()
ax = fig.add_subplot()
# Title case the feature for better display and create the visualizer
labels = list(map(lambda s: s.title(), features_))
viz = FeatureImportances(Lasso(), ax=ax, labels=labels, relative=False)
#viz = FeatureImportances(RandomForestRegressor(), ax=ax, labels=labels, relative=False)
viz.fit(X_, y_)
viz.poof()
