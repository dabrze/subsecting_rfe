# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

from lightgbm import LGBMClassifier
from srfe.subsecting_rfe import SubsectingRFE
from evaluation import evaluate

SEED = 23
X_DATA_PATH = os.path.join(os.path.dirname(__file__),
                           "../data/pdb_blobs_X.csv.gz")
Y_DATA_PATH = os.path.join(os.path.dirname(__file__),
                           "../data/pdb_blobs_y.csv.gz")

srfe_selectors = {
    "3-SRFE": SubsectingRFE(None, method="subsect", step=3, cv=5, n_jobs=1),
    "5-SRFE": SubsectingRFE(None, method="subsect", step=5, cv=5, n_jobs=1),
    "10-SRFE": SubsectingRFE(None, method="subsect", step=10, cv=5, n_jobs=1),
    "FRFE": SubsectingRFE(None, method="fibonacci", cv=5, n_jobs=1),
}
scorers = {"Accuracy": "accuracy"}
classifiers = {
    "GBM": LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)
}

if __name__ == '__main__':
    for file_pair in [(X_DATA_PATH, Y_DATA_PATH)]:
        filename = os.path.basename(file_pair[0])
        logging.info(filename)
        X = pd.read_csv(file_pair[0], index_col=0, header=0,
                        compression='gzip').as_matrix()
        logging.info(os.path.basename(file_pair[1]))
        y = pd.read_csv(file_pair[1], index_col=0, header=0,
                        compression='gzip').iloc[:, 0].as_matrix()

        for scorer in scorers:
            for classifier in classifiers:
                logging.info("Evaluating all features using %s scored with %s",
                             classifier, scorer)
                evaluate(filename, "All", None, classifiers[classifier],
                         scorers[scorer], X, y, SEED, timeout=None, n_jobs=-1,
                         results_file="CaseStudy.csv")

                # Subsecting and Fibonacci RFE
                for selector in srfe_selectors:
                    logging.info("Evaluating %s using %s scored with %s",
                                 selector, classifier, scorer)
                    evaluate(filename, selector, srfe_selectors[selector],
                             classifiers[classifier], scorers[scorer], X, y,
                             SEED, timeout=None, n_jobs=-1,
                             results_file="CaseStudy.csv", write_selected=True)
