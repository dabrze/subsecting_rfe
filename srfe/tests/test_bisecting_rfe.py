# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

from sklearn.datasets import make_friedman1, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR

from srfe.subsecting_rfe import SubsectingRFE
from unittest import TestCase


class TestBRFE(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_friedman1(n_samples=50, n_features=10,
                                      random_state=0)
        cls.X_c, cls.y_c = make_classification(n_samples=200, n_features=10,
                                               n_informative=5, n_redundant=0,
                                               n_repeated=0, n_classes=4,
                                               shuffle=False, random_state=0)

        cls.X_l, cls.y_l = make_classification(n_samples=200, n_features=10,
                                               n_informative=1, n_redundant=0,
                                               n_repeated=9, n_classes=2,
                                               n_clusters_per_class=1,
                                               shuffle=False, random_state=0)

        cls.X_r, cls.y_r = make_classification(n_samples=1000, n_features=10,
                                               n_informative=10, n_redundant=0,
                                               n_repeated=0, n_classes=5,
                                               n_clusters_per_class=2,
                                               shuffle=False, random_state=0)

    def test_simple_brfe(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, method="subsect", step=2, cv=5)
        selector = selector.fit(self.X, self.y)

        self.assertEqual(selector.n_features_, 6)
        self.assertListEqual(list(selector.support_),
                             [True, True, True, True, True,
                              False, False, False, True, False])
        self.assertListEqual(list(selector.ranking_),
                             [4, 3, 5, 1, 2, 9, 8, 10, 6, 7])

    def test_derivative_brfe(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, cv=5, method="bisect")
        selector = selector.fit(self.X, self.y)

        self.assertEqual(selector.n_features_, 6)
        self.assertListEqual(list(selector.support_),
                             [True, True, True, True, True,
                              False, False, False, True, False])
        self.assertListEqual(list(selector.ranking_),
                             [4, 3, 5, 1, 2, 9, 8, 10, 6, 7])

    def test_grid_scores(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, cv=5, method="subsect")
        selector = selector.fit(self.X, self.y)

        self.assertDictEqual(selector.grid_scores_,
                             {8: [0.5185335061284553, 0.20757885759115446,
                                  0.47032927512437284, 0.3776846502752701,
                                  0.5451975991145226],
                              10: [0.41190487128807507, 0.13605155234265276,
                                   0.46973011408552845, 0.37393771166171363,
                                   0.5417655748256495],
                              5: [0.5381369168437966, 0.2586791264531122,
                                  0.4754050489169737, 0.4397808711286595,
                                  0.5328327875008043],
                              6: [0.5157600865654008, 0.27223326180097385,
                                  0.5074814094501234, 0.4147212020096894,
                                  0.5385411492748724],
                              7: [0.5109260270024892, 0.22786073839525267,
                                  0.492240987028568, 0.41598318948771607,
                                  0.530390339891014]})

    def test_classification(self):
        estimator = RandomForestClassifier(random_state=23, max_features=None)
        selector = SubsectingRFE(estimator, cv=5, method="bisect")
        selector = selector.fit(self.X_c, self.y_c)

        self.assertEqual(selector.n_features_, 5)
        self.assertListEqual(list(selector.support_),
                             [True, True, True, True, True,
                              False, False, False, False, False])
        self.assertListEqual(list(selector.ranking_),
                             [4, 5, 2, 1, 3, 7, 8, 9, 10, 6])

    def test_limits(self):
        estimator = RandomForestClassifier(random_state=23, max_features=None)
        selector = SubsectingRFE(estimator, cv=5, step=3)
        selector = selector.fit(self.X_l, self.y_l)

        self.assertEqual(selector.n_features_, 1)

        estimator = RandomForestClassifier(random_state=23, max_features=None)
        selector = SubsectingRFE(estimator, cv=5, method="bisect")
        selector = selector.fit(self.X_l, self.y_l)

        self.assertEqual(selector.n_features_, 1)

        estimator = RandomForestClassifier(random_state=23, max_features=None)
        selector = SubsectingRFE(estimator, cv=5, step=3)
        selector = selector.fit(self.X_r, self.y_r)

        self.assertEqual(selector.n_features_, 10)

        estimator = RandomForestClassifier(random_state=23, max_features=None)
        selector = SubsectingRFE(estimator, cv=5, method="bisect")
        selector = selector.fit(self.X_r, self.y_r)

        self.assertEqual(selector.n_features_, 10)
