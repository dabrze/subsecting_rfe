# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import glob
import warnings
import scipy.io
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from srfe.subsecting_rfe import SubsectingRFE
from evaluation import evaluate

SEED = 23
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/small_datasets/smallest/*.mat")

srfe_selectors = {
    "3-SRFE": SubsectingRFE(None, method="subsect", step=3, cv=5, n_jobs=1),
    "5-SRFE": SubsectingRFE(None, method="subsect", step=5, cv=5, n_jobs=1),
    "10-SRFE": SubsectingRFE(None, method="subsect", step=10, cv=5, n_jobs=1),
    "FRFE": SubsectingRFE(None, method="fibonacci", cv=5, n_jobs=1),
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
                                            n_jobs=-1, random_state=SEED),
    "SVM": SVC(kernel="linear", random_state=SEED, max_iter=1000, probability=True), 
    "Logistic Regression": LogisticRegression(random_state=SEED, n_jobs=-1),
    "GBM": LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)
}
warnings.filterwarnings("ignore")
from time import time
if __name__ == '__main__':
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
                    time0 = time()
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, srfe_selectors[selector],
                             classifiers[classifier], scorers[scorer], X, y,
                             SEED, results_file="Benchmarks.csv")

                    time1 = time()
                    minutes = int((time1 - time0) / 60)
                    secs = int( (time1 - time0) - minutes * 60)
                    print("Processing classifier: {} selector: {} took {} minutes and {} seconds".format(classifier, selector, minutes, secs) )
