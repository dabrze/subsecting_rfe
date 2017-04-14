# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

"""Bisecting recursive feature elimination for feature ranking"""

import numpy as np
import operator

from collections import OrderedDict

from sklearn.utils import check_X_y, safe_sqr
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _safe_split, _score
from sklearn.metrics.scorer import check_scoring
from sklearn.feature_selection.base import SelectorMixin


def _single_fit(brfe, features, X, y, train, test, scorer):
    """
    Return the score and feature ranking for a fit across one fold.
    """
    X_train, y_train = _safe_split(brfe.estimator, X, y, train)
    X_test, y_test = _safe_split(brfe.estimator, X, y, test, train)

    return brfe._fit_rank_test(features, X_train, y_train, X_test, y_test,
                               scorer)

class BisectingRFE(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature selection with bisecting recursive feature elimination.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` (or feature_importances_) attribute that holds the fitted
        parameters. Important features must correspond to high absolute 
        values in the `coef_` (feature_importances_) array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules. 
        Similarly, algorithms based on decision trees also rank feature 
        importance.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the 
        estimator is a classifier or if ``y`` is neither binary nor multiclass, 
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel while fitting across folds.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that `ranking_[i]` corresponds to the 
        ranking position of the i-th feature. Selected (i.e., estimated 
        best) features are assigned rank 1.

    grid_scores_ : dictionary in the form {feture_num: cv_results}
        The cross-validation scores such that ``grid_scores_[k]`` 
        corresponds to a list of CV scores of k features. Correct key values
        ``k`` depend on the bisected feature counts. 

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.svm import SVR
    >>> from bisecting_rfe import BisectingRFE
    >>>
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = BisectingRFE(estimator, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,
            False, False, False, False, False], dtype=bool)
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    def __init__(self, estimator, cv=None, scoring=None, verbose=0, n_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the BRFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        X, y = check_X_y(X, y, "csr")

        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        # Initial values to calculate "right" in the first iteration
        # and a proper mid in subsequent iterations
        left = 0
        right = n_features
        mid = n_features // 2
        scores = {}
        mean_scores = {}
        rankings = {}

        if self.n_jobs == 1:
            parallel, func = list, _single_fit
        else:
            parallel, func, = Parallel(n_jobs=self.n_jobs), \
                              delayed(_single_fit)
        mid_scores_and_ranks = parallel(func(self, np.arange(n_features),
                                             X, y, train, test, scorer)
                                        for train, test in cv.split(X, y))
        scores[right], rankings[right] = map(list, zip(*mid_scores_and_ranks))
        mean_scores[right] = np.mean(scores[right], axis=0)

        while right != mid and left != mid:
            features = self._get_top_k_features(rankings[right], mid)

            if self.n_jobs == 1:
                parallel, func = list, _single_fit
            else:
                parallel, func, = Parallel(n_jobs=self.n_jobs), \
                                  delayed(_single_fit)

            mid_scores_and_ranks = parallel(func(self, features, X, y, train,
                                                 test, scorer)
                                            for train, test in cv.split(X, y))
            scores[mid], rankings[mid] = map(list, zip(*mid_scores_and_ranks))
            mean_scores[mid] = np.mean(scores[mid], axis=0)

            if self.verbose > 1:
                print("%d feature score: %.3f, %d feature score: %.3f" %
                      (mid, mean_scores[mid], right, mean_scores[right]))

            # update boundaries and reference objects
            if mean_scores[mid] >= mean_scores[right]: #
                right = mid
            else:
                left = mid
            mid = (right + left) // 2

        # Set final attributes
        # todo right niekoniecznie?
        features = self._get_top_k_features(rankings[right], right)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        self.n_features_ = len(features)
        self.support_ = np.zeros(n_features, dtype=np.bool)
        self.support_[features] = True
        self.ranking_ = np.ones(n_features, dtype=np.int)
        self.ranking_[np.logical_not(self.support_)] += 1

        self.grid_scores_ = scores

        if self.verbose > 0:
            print("Final number of features: %d." % self.n_features_)

        return self

    def _fit_rank_test(self, features, X_train, y_train, X_test, y_test,
                       scorer):
        """
        Score and rank features for given training and testing dataset.
        """
        # Rank the remaining features
        estimator = clone(self.estimator)
        if self.verbose > 0:
            print("Fitting estimator with %d features." % len(features))

        estimator.fit(X_train[:, features], y_train)

        # Get coefs
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" or "feature_importances_" '
                               'attributes')

        # Get ranks
        if coefs.ndim > 1:
            ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
        else:
            ranks = np.argsort(safe_sqr(coefs))

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)
        ranks = features[ranks]
        score = _score(estimator, X_test[:, features], y_test, scorer)

        return (score, ranks)

    def _get_top_k_features(self, ranks, k):
        """
        Get the top ranked features from a list of rankings. Ranks from each 
        ranking are first summed, then the top-k features with the best summed
        rank are returned.
        """
        if len(ranks) == 1:
            #no cv or initial set of ranks
            return np.asarray(ranks[0][-k:])
        else:
            import collections
            summed_ranks = collections.defaultdict(lambda: 0)
            for ranking in ranks:
                for pos in range(len(ranking)):
                    summed_ranks[ranking[pos]] += pos

            result = []
            for feature, summed_rank in sorted(summed_ranks.items(),
                                               key=operator.itemgetter(1)):
                result.append(feature)

            return np.asarray(result[-k:])

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))
