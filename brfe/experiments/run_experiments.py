# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import glob
import scipy.io
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from brfe.bisecting_rfe import BisectingRFE
from evaluation import evaluate

SEED = 23
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/*.mat")

selectors = {
    "3-SRFE": BisectingRFE(None, method="subsect", step=3, cv=5, n_jobs=1),
    "5-SRFE": BisectingRFE(None, method="subsect", step=5, cv=5, n_jobs=1),
    "10-SRFE": BisectingRFE(None, method="subsect", step=10, cv=5, n_jobs=1),
    "BRFE": BisectingRFE(None, method="bisect", cv=5, n_jobs=1),
    "RFE-log-3": RFECV(None, step="log-3", cv=5, n_jobs=1),
    "RFE-log-5": RFECV(None, step="log-5", cv=5, n_jobs=1),
    "RFE-log-10": RFECV(None, step="log-10", cv=5, n_jobs=1),
    "RFE-log": RFECV(None, step="log", cv=5, n_jobs=1),
             }
scorers = {"Accuracy": "accuracy"}
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=30, max_features=0.3,
                                            n_jobs=-1, random_state=SEED),
    "SVM": SVC(kernel="linear", random_state=SEED, max_iter=1000),
    "Logistic Regression": LogisticRegression(random_state=SEED, n_jobs=-1)}

if __name__ == '__main__':
    for file in glob.glob(DATA_PATH):
        filename = os.path.basename(file)

        logging.info(filename)
        mat = scipy.io.loadmat(file)
        X = mat['X'].astype(float)
        y = mat['Y'][:, 0]

        for scorer in scorers:
            for classifier in classifiers:
                evaluate(filename, "All", None, classifiers[classifier],
                         scorers[scorer], X, y, SEED,
                         results_file="Benchmarks.csv")

                for selector in selectors:
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, selectors[selector],
                             classifiers[classifier], scorers[scorer], X, y,
                             SEED, results_file="Benchmarks.csv")
