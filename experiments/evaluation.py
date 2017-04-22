# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import csv
import time
import logging
import numpy as np
import pandas as pd
import scipy as sp

from sklearn import metrics
from sklearn.base import clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.fixes import bincount
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.classification import _prf_divide
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


class Evaluation:
    """
    Evaluation results for a given classifier on a given dataset.
    """
    def __init__(self, dataset_name, X, y, fold, classifier, processing_time,
                 selected_feature_num, y_true, y_pred_all, y_pred_sel):
        self.dataset_name = dataset_name
        self.fold = fold
        self.dataset_stats = DatasetStatistics(X, y)
        self.classifier = classifier
        self.all_feature_num = self.dataset_stats.attributes
        self.selected_feature_num = selected_feature_num
        self.y_true = y_true
        self.y_pred_all = y_pred_all
        self.y_pred_sel = y_pred_sel
        self.processing_time = processing_time
        self.accuracy_all = metrics.accuracy_score(y_true, y_pred_all)
        self.macro_recall_all = metrics.recall_score(y_true, y_pred_all,
                                                     average="macro")
        self.kappa_all = metrics.cohen_kappa_score(y_true, y_pred_all)
        self.gmean_all = g_mean(y_true, y_pred_all)
        self.accuracy_sel = metrics.accuracy_score(y_true, y_pred_sel)
        self.macro_recall_sel = metrics.recall_score(y_true, y_pred_sel,
                                                     average="macro")
        self.kappa_sel = metrics.cohen_kappa_score(y_true, y_pred_sel)
        self.gmean_sel = g_mean(y_true, y_pred_sel)
        self.num_of_classes = self.dataset_stats.num_of_classes
        self.start_date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()-processing_time))

    def write_to_csv(self, file_name="ExperimentResults.csv",
                     save_to_folder=os.path.join(os.path.dirname(__file__),
                                                 "results")):
        """
        Adds a new row to a csv file with evaluation results. If the given filenmae does not correspond to any existing
        csv, a new file is created.
        :param file_name: csv file name
        :type file_name: string
        :param save_to_folder: folder to save the file to
        :type save_to_folder: string, optional (default=source file folder/ExperimentResults)
        """
        if not os.path.exists(save_to_folder):
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

            if write_header:
                writer.writerow(["Start date",
                                 "Dataset",
                                 "Examples",
                                 "Attributes",
                                 "Number of classes",
                                 "Min class examples",
                                 "Max class examples",
                                 "Classifier",
                                 "Fold",
                                 "Processing time",
                                 "Feature num",
                                 "Accuracy (All)",
                                 "Macro recall (All)",
                                 "Kappa (All)",
                                 "G-mean (All)",
                                 "Selected feature num",
                                 "Accuracy (Selected)",
                                 "Macro recall (Selected)",
                                 "Kappa (Selected)",
                                 "G-mean (Selected)"
                                 ])

            writer.writerow([self.start_date_time,
                             self.dataset_name,
                             self.dataset_stats.examples,
                             self.dataset_stats.attributes,
                             self.dataset_stats.num_of_classes,
                             self.dataset_stats.min_examples,
                             self.dataset_stats.max_examples,
                             self.classifier,
                             self.fold,
                             self.processing_time,
                             self.all_feature_num,
                             self.accuracy_all,
                             self.macro_recall_all,
                             self.kappa_all,
                             self.gmean_all,
                             self.selected_feature_num,
                             self.accuracy_sel,
                             self.macro_recall_sel,
                             self.kappa_sel,
                             self.gmean_sel
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

def evaluate(dataset, selector, classifier, scorer, X, y, seed, folds=10):
    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=False)

    fold = 0
    for train_index, test_index in skf.split(X, y):
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = clone(classifier)
        clf.fit(X_train, y_train)
        y_true, y_pred_all = y_test, clf.predict(X_test)

        clf = clone(classifier)
        start = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start
        selected_feature_num = 10 #TODO
        y_pred = y_pred_all #TODO

        evaluation = Evaluation(dataset, X, y, fold, classifier, training_time,
                                selected_feature_num, y_true, y_pred_all,
                                y_pred)
        evaluation.write_to_csv()


def plot_comparison(file_name="ExperimentResults.csv",
                    save_to_folder=os.path.join(os.path.dirname(__file__),
                                                 "results")):
    logging.info("Creating comparison plot")

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
        tp_sum = bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        # Pathological case
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = bincount(y_true, weights=None, minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
    recall[recall == 0] = correction

    return sp.stats.mstats.gmean(recall)
