# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

from sklearn.datasets import make_friedman1
from sklearn.svm import SVR
from srfe.subsecting_rfe import SubsectingRFE

X, y = make_friedman1(n_samples=100, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = SubsectingRFE(estimator, step=3, method="subsect", cv=5)
selector = selector.fit(X, y)

print(selector.n_features_)
print(selector.ranking_)
print(selector.grid_scores_)