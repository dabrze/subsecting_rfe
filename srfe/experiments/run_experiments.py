# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import glob
import warnings
from sklearn.exceptions import ConvergenceWarning
import scipy.io
import logging

from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from srfe.subsecting_rfe import SubsectingRFE
from evaluation import evaluate

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
SEED = 13
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/*.mat") # leukemia30min colon20min lung_small20min

srfe_selectors = {
    "3-SRFE": SubsectingRFE(None, method="subsect", step=3, cv=5, n_jobs=1),
    "5-SRFE": SubsectingRFE(None, method="subsect", step=5, cv=5, n_jobs=1),
    "10-SRFE": SubsectingRFE(None, method="subsect", step=10, cv=5, n_jobs=1),
    "FRFE": SubsectingRFE(None, method="fibonacci", cv=5, n_jobs=1),
}
scorers = {"Accuracy": "accuracy"}
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=30, max_features=0.3,
                                            n_jobs=-1, random_state=SEED),
    "SVM": SVC(kernel="linear", random_state=SEED, max_iter=1000, probability=True),
    "Logistic Regression": LogisticRegression(random_state=SEED, n_jobs=-1),
    "GBM": LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)
}

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    for file in glob.glob(DATA_PATH):
        filename = os.path.basename(file)
        logging.info(filename)
        mat = scipy.io.loadmat(file)
        X = mat['X'].astype(float)
        y = mat['Y'][:, 0]

        for scorer in scorers:
            for classifier in classifiers:
                # Subsecting and Fibonacci RFE
                for selector in srfe_selectors:
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, srfe_selectors[selector],
                        classifiers[classifier], scorers[scorer], X, y,
                        SEED, results_file="Benchmarks.csv")
    logging.info('Finished')
