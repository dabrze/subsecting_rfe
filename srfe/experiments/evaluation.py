# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import csv
import ast
import time
import math
import logging
import warnings

import numpy as np
import pandas as pd
import scipy as sp

from sklearn import metrics
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.classification import _prf_divide
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


class Evaluation:
    """
    Evaluation results for a given classifier on a given dataset.
    """
    def __init__(self, dataset_name, selector_name, X, y, classifier,
                 selector, scorer, processing_time, selected_feature_num,
                 y_true, y_pred, grid_scores, selected_features,
                 write_selected=False):
        np.set_printoptions(threshold=1000000)

        self.dataset_name = dataset_name
        self.dataset_stats = DatasetStatistics(X, y)
        self.selector_name = selector_name
        self.classifier = classifier.__repr__().replace(",", ";")\
            .replace("\n", " ").replace("\r", "")
        self.selector = selector.__repr__().replace(",", ";")\
            .replace("\n", " ").replace("\r", "")
        self.scorer = scorer
        self.feature_num = self.dataset_stats.attributes
        self.selected_feature_num = selected_feature_num
        self.y_true = y_true
        self.y_pred = y_pred
        self.selected_features = \
            np.array2string(selected_features, max_line_width=1000000)\
                if selected_features is not None else None
        self.write_selected = write_selected
        self.processing_time = processing_time
        self.accuracy = metrics.accuracy_score(y_true, y_pred)
        self.macro_recall = metrics.recall_score(y_true, y_pred,
                                                 average="macro")
        self.kappa = metrics.cohen_kappa_score(y_true, y_pred)
        self.gmean = g_mean(y_true, y_pred)
        self.num_of_classes = self.dataset_stats.num_of_classes
        if processing_time is not None:
            self.start_date_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                time.localtime(time.time()-processing_time))
        else:
            self.start_date_time = "Error"
        if grid_scores is None or isinstance(grid_scores, dict):
            self.grid_scores = str(grid_scores).replace('\n', ' ')\
                .replace('\r', '')
        else:
            self.grid_scores = str(list(grid_scores))

        np.set_printoptions(threshold=1000)

    def write_to_csv(self, file_name="ExperimentResults.csv",
                     save_to_folder=os.path.join(os.path.abspath(
                         os.path.dirname(__file__)), "results")):
        """
        Adds a new row to a csv file with evaluation results. If the given filenmae does not correspond to any existing
        csv, a new file is created.
        :param file_name: csv file name
        :type file_name: string
        :param save_to_folder: folder to save the file to
        :type save_to_folder: string, optional (default=source file folder/ExperimentResults)
        """

        if not os.path.exists(save_to_folder):
            logging.info("Creating folder: %s", save_to_folder)
            os.mkdir(save_to_folder)
        file_path = os.path.join(save_to_folder, file_name)

        if os.path.isfile(file_path):
            write_header = False
            mode = "a"
        else:
            write_header = True
            mode = "w"

        with open(file_path, mode) as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
            np.set_printoptions(threshold=np.inf)

            if write_header:
                writer.writerow(["Start date",
                                 "Dataset",
                                 "Examples",
                                 "Attributes",
                                 "Number of classes",
                                 "Min class examples",
                                 "Max class examples",
                                 "Classifier",
                                 "Feature selector",
                                 "Selector params",
                                 "Scorer",
                                 "Processing time",
                                 "Feature num",
                                 "Selected num",
                                 "Accuracy",
                                 "Macro recall",
                                 "Kappa",
                                 "G-mean",
                                 "Grid scores",
                                 "Selected features"
                                 ])

            writer.writerow([self.start_date_time,
                             self.dataset_name,
                             self.dataset_stats.examples,
                             self.dataset_stats.attributes,
                             self.dataset_stats.num_of_classes,
                             self.dataset_stats.min_examples,
                             self.dataset_stats.max_examples,
                             self.classifier,
                             self.selector_name,
                             self.selector,
                             self.scorer,
                             self.processing_time,
                             self.feature_num,
                             self.selected_feature_num,
                             self.accuracy,
                             self.macro_recall,
                             self.kappa,
                             self.gmean,
                             self.grid_scores,
                             self.selected_features if self.write_selected
                             else None
                             ])


class DatasetStatistics:
    """
    Dataset statistics.
    """
    def __init__(self, X, y):
        """
        Constructor.
        :param data_frame: dataset
        :type data_frame: Pandas data frame
        :param class_attribute: class attribute column name
        :type class_attribute: string
        """
        self.examples = X.shape[0]
        self.attributes = X.shape[1]
        class_count = list(pd.Series(y).value_counts())
        self.min_examples = class_count[-1]
        self.max_examples = class_count[0]
        self.num_of_classes = class_count.__len__()
        self.classes = class_count

    def __repr__(self):
        """
        Returns a string description of a dataset containing basic statistics (number of examples, attributes, classes).
        :return: string representation
        """
        return "\texamples: {0}\r\n".format(self.examples) + \
               "\tattributes: {0}\r\n".format(self.attributes) + \
               "\tnum of classes: {0}\r\n".format(self.num_of_classes) + \
               "\tmin class examples: {0}\r\n".format(self.min_examples) + \
               "\tmax class examples: {0}\r\n".format(self.max_examples) + \
               "\tclasses: {0}".format(" ".join([str(key) + ": " + str(value)
                                                 for key, value in self.classes.iteritems()])
                                       if self.classes.shape[0] <= 200 else str(self.classes.shape[0]))


def evaluate(dataset, selector_name, selector, classifier, scorer, X, y,
             seed, folds=10, n_jobs=-1, timeout=1*60*60,
             results_file="ExperimentResults.csv", write_selected=False):
    cv = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=False)

    try:
        evaluations = Parallel(n_jobs=n_jobs, timeout=timeout)(
            delayed(_single_fit)(dataset, selector_name, selector, classifier,
                                 scorer, X, y, train, test, write_selected,
                                 fold, results_file)
            for fold, (train, test) in enumerate(cv.split(X, y)))
    except Exception as ex:
        evaluation = Evaluation(dataset, selector_name, X, y, classifier,
                                selector, scorer, timeout, "error", [1], [0],
                                None, None)
        evaluations = [evaluation] * folds
        logging.warning("Exception: %s" % ex)
    except:
        evaluation = Evaluation(dataset, selector_name, X, y, classifier,
                                selector, scorer, timeout, "timeout", [1], [0],
                                None, None)
        evaluations = [evaluation] * folds
        logging.warning("%s probably interrupted after timeout %d seconds" %
                        (selector_name, timeout))

    for evaluation in evaluations:
        evaluation.write_to_csv(results_file)


def _step_num_from_results(dataset, classifier, selector, results_file, fold):
    clf_str = classifier.__repr__().replace(",", ";").replace("\n", " ").\
        replace("\r", "")
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             "results", results_file)
    df = pd.read_csv(file_path)
    selector_mapping = {
        "RFE-log-3": "3-SRFE",
        "RFE-log-5": "5-SRFE",
        "RFE-log-10": "10-SRFE",
        "RFE-log": "FRFE"
    }

    folds = df[(df["Feature selector"] == selector_mapping[selector]) &
               (df["Dataset"] == dataset) &
               (df["Classifier"].str.replace('\n', ' ').str.replace('\r', '')
                == clf_str)]["Grid scores"]
    grid_scores = ast.literal_eval(folds.iloc[fold])
    return len(grid_scores)

def _single_fit(dataset, selector_name, selector, classifier, scorer, X, y,
                train, test, write_selected, fold, results_file):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    if selector is None:
        clf = make_pipeline(MinMaxScaler(), clone(classifier))
    else:
        sel = clone(selector)
        sel.set_params(estimator=clone(classifier), scoring=scorer)
        if "step" in sel.get_params():
            if sel.get_params()["step"] == "custom":
                feature_num = X.shape[1]
                srfe_step_num = _step_num_from_results(dataset, classifier,
                                                       selector_name,
                                                       results_file, fold)
                step = feature_num // srfe_step_num + 1
                sel.set_params(step=step)

        clf = make_pipeline(MinMaxScaler(), sel)

    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        clf.fit(X_train, y_train)
    training_time = time.time() - start
    y_true, y_pred = y_test, clf.predict(X_test)

    if selector is None:
        selected_feature_num = None
        grid_scores = None
        selected_features = None
    else:
        selected_feature_num = clf.steps[1][1].n_features_
        grid_scores = clf.steps[1][1].grid_scores_
        selected_features = clf.steps[1][1].support_

    return Evaluation(dataset, selector_name, X, y, classifier, selector,
                      scorer, training_time, selected_feature_num, y_true,
                      y_pred, grid_scores, selected_features, write_selected)


def g_mean(y_true, y_pred, labels=None, correction=0.001):
    """
    Computes the geometric mean of class-wise recalls.
    :param y_true: True class labels.
    :type y_true: list
    :param y_pred: Predicted class labels.
    :type y_pred: array-like
    :param labels:  Labels present in the data can be excluded, for example to calculate a multiclass average ignoring a
    majority negative class, while labels not present in the data will result in 0 components in a macro average.
    :type labels: list, optiona
    :param correction: substitution/correction for zero values in class-wise recalls
    :type correction: float
    :return: G-mean value
    """
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels, assume_unique=True)])

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    # labels are now from 0 to len(labels) - 1 -> use bincount
    tp = y_true == y_pred
    tp_bins = y_true[tp]

    if len(tp_bins):
        tp_sum = np.bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        # Pathological case
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = np.bincount(y_true, weights=None, minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
    recall[recall == 0] = correction

    return sp.stats.mstats.gmean(recall)
