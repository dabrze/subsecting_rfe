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

    def test_min_srfe(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, method="subsect", step=2, cv=5)
        selector = selector.fit(self.X, self.y)

        self.assertEqual(selector.n_features_, 9)
        self.assertListEqual(list(selector.support_),
                             [True, True, True, True, True,
                              False, True, True, True, True])
        self.assertListEqual(list(selector.ranking_),
                             [5, 3, 4, 1, 2, 10, 8, 7, 6, 9])

    def test_max_srfe(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, method="subsect", step=10, cv=5)
        selector = selector.fit(self.X, self.y)

        self.assertEqual(selector.n_features_, 5)
        self.assertListEqual(list(selector.support_),
                             [True, True, True, True, True,
                              False, False, False, False, False])
        self.assertListEqual(list(selector.ranking_),
                             [5, 3, 4, 1, 2, 10, 8, 7, 6, 9])

    def test_derivative_brfe(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, cv=5, method="bisect")
        selector = selector.fit(self.X, self.y)

        self.assertEqual(selector.n_features_, 2)
        self.assertListEqual(list(selector.support_),
                             [False, False, False, True, True,
                              False, False, False, False, False])
        self.assertListEqual(list(selector.ranking_),
                             [5, 3, 4, 1, 2, 10, 8, 7, 6, 9])

    def test_grid_scores(self):
        estimator = SVR(kernel="linear")
        selector = SubsectingRFE(estimator, cv=5, method="subsect")
        selector = selector.fit(self.X, self.y)

        self.assertDictEqual(selector.grid_scores_,
                             {1: [0.3292585855834661, 0.09948802621756825,
                                  0.23431100934725313, 0.05184167205410384,
                                  0.3447052037680079],
                              2: [0.3614134708042146, 0.31978853750926894,
                                  0.34085152150463516, 0.20069623061669464,
                                  0.4779986874099314],
                              3: [0.38937936748076785, 0.2426011887329178,
                                  0.29511883178915677, 0.2137608641085056,
                                  0.4840433064804135],
                              4: [0.44814077865830815, 0.2356586592425571,
                                  0.41580010132152834, 0.3814514476737748,
                                  0.5109883252425744],
                              5: [0.5381369168437966, 0.22815006450017938,
                                  0.4754050489169737, 0.4397808711286595,
                                  0.5328327875008043],
                              6: [0.44877715009043917, 0.17781307616497966,
                                  0.5074814094501234, 0.42718548677721024,
                                  0.5443321456365926],
                              7: [0.4286816277965356, 0.0810943918479532,
                                  0.49224098702856817, 0.38700115376139377,
                                  0.535944025056118],
                              8: [0.4163755774609529, 0.15836791821251173,
                                  0.470329275124373, 0.3776846502752699,
                                  0.5451975991145226],
                              9: [0.4162113861294575, 0.16678129366936612,
                                  0.46879939519105335, 0.3741498782955357,
                                  0.543981486779614],
                              10: [0.41190487128807507, 0.13605155234265276,
                                   0.46973011408552845, 0.37393771166171363,
                                   0.5417655748256495]})

    def test_classification(self):
        estimator = RandomForestClassifier(random_state=23, max_features=None)
        selector = SubsectingRFE(estimator, cv=5, method="bisect")
        selector = selector.fit(self.X_c, self.y_c)

        self.assertEqual(selector.n_features_, 5)
        self.assertListEqual(list(selector.support_),
                             [True, True, True, True, True,
                              False, False, False, False, False])
        self.assertListEqual(list(selector.ranking_),
                             [3, 4, 1, 2, 5, 9, 8, 7, 10, 6])

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
