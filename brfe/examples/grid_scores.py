# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

from sklearn.datasets import make_friedman1
from sklearn.svm import SVR
from brfe.bisecting_rfe import BisectingRFE

X, y = make_friedman1(n_samples=100, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = BisectingRFE(estimator, cv=5)
selector = selector.fit(X, y)

print(selector.n_features_)
print(selector.ranking_)
print(selector.grid_scores_)