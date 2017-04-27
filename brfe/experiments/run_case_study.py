# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import glob
import pandas as pd
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
X_DATA_PATH = os.path.join(os.path.dirname(__file__),
                           "../data/pdb_blobs_X.csv")
Y_DATA_PATH = os.path.join(os.path.dirname(__file__),
                           "../data/pdb_blobs_y.csv")
X_SMALL_DATA_PATH = os.path.join(os.path.dirname(__file__),
                                 "../data/pdb_blobs_X_small.csv")
Y_SMALL_DATA_PATH = os.path.join(os.path.dirname(__file__),
                                 "../data/pdb_blobs_y_small.csv")

selectors = {
    "BRFE": BisectingRFE(None, use_derivative=False, cv=5, n_jobs=1),
    "BRFE+": BisectingRFE(None, use_derivative=False, cv=5, n_jobs=1,
                          promote_more_features=True),
    "d-BRFE": BisectingRFE(None, use_derivative=True, cv=5, n_jobs=1),
    "d-BRFE+": BisectingRFE(None, use_derivative=True, cv=5, n_jobs=1,
                            promote_more_features=True),
    "RFE-1": RFECV(None, step=1, cv=5, verbose=0, n_jobs=1),
    "RFE-log": RFECV(None, step="log", cv=5, verbose=0, n_jobs=1)
             }
scorers = {"Accuracy": "accuracy"}
classifiers = {"Random Forest": RandomForestClassifier(n_estimators=30,
                                                       max_features=0.3,
                                                       n_jobs=-1,
                                                       random_state=SEED)}

if __name__ == '__main__':
    for file_pair in [(X_SMALL_DATA_PATH, Y_SMALL_DATA_PATH),
                      (X_DATA_PATH, Y_DATA_PATH)]:
        filename = os.path.basename(file_pair[0])
        logging.info(filename)
        X = pd.read_csv(file_pair[0], compression="gzip", index_col=0,
                        header=0).as_matrix()
        logging.info(os.path.basename(file_pair[1]))
        y = pd.read_csv(file_pair[1], index_col=0, header=0).iloc[:, 0].as_matrix()

        for scorer in scorers:
            for classifier in classifiers:
                evaluate(filename, "All", None, classifiers[classifier],
                         scorers[scorer], X, y, SEED, timeout=None)

                for selector in selectors:
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, selectors[selector],
                             classifiers[classifier], scorers[scorer], X, y,
                             SEED, timeout=None)
