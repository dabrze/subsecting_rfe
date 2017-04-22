# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import scipy
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.datasets import make_friedman1, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from bisecting_rfe import BisectingRFE
from evaluation import evaluate, plot_comparison

SEED = 23
DATA_PATH = "../data/"

selectors = {"BRFE": BisectingRFE(None, use_derivative=False, cv=5, verbose=0,
                                  n_jobs=-1),
             "d-BRFE": BisectingRFE(None, use_derivative=True, cv=5, verbose=0,
                                   n_jobs=-1),
             "RFE-1": RFECV(None, step=1, cv=5, verbose=0, n_jobs=-1),
             "RFE-5": RFECV(None, step=10, cv=5, verbose=0, n_jobs=-1)}
scorers = {"Accuracy": "accuracy",
           "Kappa": make_scorer(cohen_kappa_score)}
classifiers = {"Random Forest": RandomForestClassifier(n_estimators=50,
                                                       random_state=SEED),
               "SVM": SVC(kernel="linear", random_state=SEED),
               "Logistic Regression": LogisticRegression(random_state=SEED)}


for filename in os.listdir(DATA_PATH):
    logging.info(filename)
    mat = scipy.io.loadmat(os.path.join(DATA_PATH, filename))
    X = mat['X'].astype(float)
    y = mat['Y'][:, 0]

    for scorer in scorers:
        for classifier in classifiers:
            for selector in selectors:
                logging.info("Evaluating %s using %s scored with %s",
                             selector, classifier, scorer)
                evaluate(filename, selectors[selector], classifiers[
                    classifier], scorers[scorer], X, y, SEED)

plot_comparison()
