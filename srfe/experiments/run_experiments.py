# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import glob
import warnings
import scipy.io
import logging

from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from srfe.subsecting_rfe import SubsectingRFE
from evaluation import evaluate

SEED = 23
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/*.mat")
njobs= -1

logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'results/logs.txt'),
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

srfe_selectors = {
    "3-SRFE": SubsectingRFE(None, method="subsect", step=3, cv=5, n_jobs=njobs),
    "5-SRFE": SubsectingRFE(None, method="subsect", step=5, cv=5, n_jobs=njobs),
    "10-SRFE": SubsectingRFE(None, method="subsect", step=10, cv=5, n_jobs=njobs),
    "FRFE": SubsectingRFE(None, method="fibonacci", cv=5, n_jobs=njobs),
}
rfe_selectors = {
    "RFE-log-3": RFECV(None, step="custom", cv=5, n_jobs=1),
    "RFE-log-5": RFECV(None, step="custom", cv=5, n_jobs=1),
    "RFE-log-10": RFECV(None, step="custom", cv=5, n_jobs=1),
    "RFE-log": RFECV(None, step="custom", cv=5, n_jobs=1),
}
scorers = {"Accuracy": "accuracy"}
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=30, max_features=0.3,
                                           n_jobs=njobs, random_state=SEED),
    "SVM": SVC(kernel="linear", random_state=SEED, max_iter=1000, probability=True),
    "Logistic Regression": LogisticRegression(random_state=SEED, n_jobs=njobs),
    "GBM": LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=njobs)
}
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.info('START')
    for file in glob.glob(DATA_PATH):
        filename = os.path.basename(file)
        logging.info(filename)
        mat = scipy.io.loadmat(file)
        X = mat['X'].astype(float)
        y = mat['Y'][:, 0]
        for scorer in scorers:
            for classifier in classifiers:
                # Subsecting and Fibonacci RFE
                logging.info("Evaluating all features using %s scored with %s",
                             classifier, scorer)
                evaluate(filename, "All", None, classifiers[classifier],
                         scorers[scorer], X, y, SEED,
                         results_file="Benchmarks.csv", n_jobs=njobs)

                # Subsecting and Fibonacci RFE
                for selector in srfe_selectors:
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, srfe_selectors[selector],
                        classifiers[classifier], scorers[scorer], X, y,
                        SEED, results_file="Benchmarks.csv", n_jobs=njobs)

                # Standard RFE equivalents
                for selector in rfe_selectors:
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, rfe_selectors[selector],
                             classifiers[classifier], scorers[scorer], X, y,
                             SEED, results_file="Benchmarks.csv", n_jobs=njobs)
    logging.info('FINISHED')
