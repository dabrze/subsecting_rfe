# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

"""Bisecting recursive feature elimination for feature ranking"""

import numpy as np
import operator
import collections

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


def _single_fit(brfe, features, X, y, train, test, scorer, fold):
    """
    Return the score and feature ranking for a fit across one fold.
    """
    X_train, y_train = _safe_split(brfe.estimator, X, y, train)
    X_test, y_test = _safe_split(brfe.estimator, X, y, test, train)

    return brfe._fit_rank_test(features[fold], X_train, y_train, X_test,
                               y_test, scorer)


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

    step : int or "bisect", optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If "bisect", then `the algorithm performs bisecting recursive feature
        elimination.

    method : str, optional, default: "subsect"
        Tells the algorithm whether how to look for best number of features. 
        Possible inputs for method are:
        
        - "subsect", to divide each surrounding of max with with step 
        additional points,
        - "bisect", to perform bisection.

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
    """
    def __init__(self, estimator, step=1, method="subsect", cv=None,
                 scoring=None, verbose=0, n_jobs=1):
        self.estimator = estimator
        self.method = method
        self.step = step
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
        n_features = int(X.shape[1])

        self.grid_scores_ = dict()
        self.mean_scores_ = dict()
        self.rankings_ = dict()
        self.rankings_[n_features] = []
        self.rankings_[n_features + 1] = []

        for fold in range(cv.n_splits):
            self.rankings_[n_features].append(list(range(n_features)))
            self.rankings_[n_features + 1].append(list(range(n_features)))

        if self.method == "bisect":
            # Initial values
            lower = 0
            upper = n_features
            self.mean_scores_[0] = float("-inf")
            self.mean_scores_[n_features + 1] = float("-inf")

            while upper - lower > 1:
                mid = (upper + lower) // 2
                d_upper = self._discrete_derivative(upper, upper, cv, X, y,
                                                    scorer)
                d_mid = self._discrete_derivative(mid, upper, cv, X, y, scorer)

                # update interval
                old_settings = np.seterr(invalid="ignore")
                if d_upper * d_mid < 0:
                    lower = mid
                elif d_upper * d_mid == 0:
                    if d_mid > 0:
                        lower = mid
                    else:
                        upper = mid
                else:
                    upper = mid
                np.seterr(**old_settings)

            # Determine final attributes
            n_features_to_select = lower \
                if self.mean_scores_[lower] > self.mean_scores_[upper] \
                else upper

            features = self._top_features(self.rankings_[upper],
                                          n_features_to_select)
        elif self.method == "subsect":
            if self.step < 2:
                raise ValueError("Step for method='subsect' must be >= 2")

            # Starting range
            lower = 1
            upper = n_features
            m_step = (upper-lower) // self.step

            if m_step == 0:
                m_step = 1

            self.mean_scores_[0] = float("-inf")
            features = self._top_features(self.rankings_[n_features],
                                          n_features)
            self.grid_scores_[upper], self.rankings_[upper], \
            self.mean_scores_[upper] \
                = self._get_cv_results(features, cv, X, y, scorer)

            while m_step > 0:
                mids = [m for m in range(upper - m_step, lower-1, -m_step)]
                previous_mid = upper

                for mid in mids:
                    features = self._top_features(self.rankings_[previous_mid],
                                                  mid)
                    self.grid_scores_[mid], self.rankings_[mid], \
                    self.mean_scores_[mid] \
                        = self._get_cv_results(features, cv, X, y, scorer)
                    previous_mid = mid

                # Find feature num with max score
                best = 0
                best_score = float("-inf")
                for feat_num in [upper] + mids:
                    if self.mean_scores_[feat_num] >= best_score:
                        best_score = self.mean_scores_[feat_num]
                        best = feat_num

                lower = best - m_step if best - m_step > 0 else best
                upper = best + m_step if best != upper else best
                m_step = min((upper-lower) // self.step, m_step - 1)

            n_features_to_select = best
            features = self.rankings_[n_features_to_select]
        else:
            raise ValueError("Invalid 'method' value: %s" % self.method)

        # Set final attributes
        features = self._get_final_features(features)
        self.n_features_ = len(features)
        self.support_ = np.zeros(n_features, dtype=np.bool)
        self.support_[features] = True
        self.ranking_ = features

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        if self.verbose > 0:
            print("Final number of features: %d." % self.n_features_)

        return self

    def _get_cv_results(self, features, cv, X, y, scorer):
        if self.n_jobs == 1:
            parallel, func = list, _single_fit
        else:
            parallel, func, = Parallel(n_jobs=self.n_jobs), \
                              delayed(_single_fit)

        mid_scores_and_ranks = parallel(func(self, features, X, y, train,
                                             test, scorer, fold)
                                        for fold, (train, test) in
                                        enumerate(cv.split(X, y)))
        cv_scores, cv_ranks = map(list, zip(*mid_scores_and_ranks))
        mean_cv_score = np.mean(cv_scores, axis=0)

        if self.verbose > 1:
            print("Mean cv-score:", mean_cv_score)

        return cv_scores, cv_ranks, mean_cv_score

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

        if scorer is not None:
            score = _score(estimator, X_test[:, features], y_test, scorer)
        else:
            score = None

        return (score, ranks)

    @staticmethod
    def _top_features(ranks, k):
        """
        Get the top ranked features from a list of rankings. Ranks from each 
        ranking are first summed, then the top-k features with the best summed
        rank are returned.
        """
        new_ranks = []

        for rank_list in ranks:
            new_ranks.append(np.asarray(rank_list[-k:]))

        return new_ranks

    def _discrete_derivative(self, mid, upper, cv, X, y, scorer):
        if mid+1 not in self.mean_scores_:
            features = self._top_features(self.rankings_[upper], mid+1)
            self.grid_scores_[mid+1], self.rankings_[mid+1], \
            self.mean_scores_[mid+1] = \
                self._get_cv_results(features, cv, X, y, scorer)

        if mid not in self.mean_scores_:
            features = self._top_features(self.rankings_[mid+1], mid)
            self.grid_scores_[mid], self.rankings_[mid],\
            self.mean_scores_[mid] = \
                self._get_cv_results(features, cv, X, y, scorer)

        return self.mean_scores_[mid+1] - self.mean_scores_[mid]

    def _get_final_features(self, features):
        n_to_select = len(features[0])

        summed_ranks = collections.defaultdict(lambda: 0)
        for ranking in features:
            for pos in range(len(ranking)):
                summed_ranks[ranking[pos]] += pos

        mean_cv_ranking = []
        for feature, summed_rank in sorted(summed_ranks.items(),
                                           key=operator.itemgetter(1)):
            mean_cv_ranking.append(feature)

        return mean_cv_ranking[-n_to_select:]

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
