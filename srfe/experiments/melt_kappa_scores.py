import os
import pandas as pd
import numpy as np
import ast
import re
import math
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "../experiments/results")
DATA_FILE = "CaseStudy.csv"
RESULT_FILE = "KappaScores.csv"
DEBUG = True

def computeKappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = checkEachLineCount(mat)  # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if DEBUG:
        print
        n, "raters."
        print
        N, "subjects."
        print
        k, "categories."

    # Computing p[]
    p = [0.0] * k
    for j in range(k):
        p[j] = 0.0
        for i in range(N):
            p[j] += mat[i][j]
        p[j] /= N * n
    if DEBUG: print
    "p =", p

    # Computing P[]
    P = [0.0] * N
    for i in range(N):
        P[i] = 0.0
        for j in range(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    if DEBUG: print
    "P =", P

    # Computing Pbar
    Pbar = sum(P) / N
    if DEBUG: print
    "Pbar =", Pbar

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    if DEBUG: print
    "PbarE =", PbarE

    kappa = (Pbar - PbarE) / (1 - PbarE)
    if DEBUG: print
    "kappa =", kappa

    return kappa

def checkEachLineCount(mat):
    """ Assert that each line has a constant number of ratings
        @param mat The matrix checked
        @return The number of ratings
        @throws AssertionError If lines contain different number of ratings """
    n = sum(mat[0])

    assert all(
        sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    return n


if __name__ == '__main__':
    np.set_printoptions(threshold=1000000)
    pdb = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                   "../data/pdb_blobs_X.csv.gz"),
                      index_col=0, header=0, compression='gzip')
    df = pd.read_csv(os.path.join(DATA_PATH, DATA_FILE))

    datasets = df.loc[:, "Dataset"].unique()
    selectors = df.loc[:, "Feature selector"].unique()
    classifiers = df.loc[:, "Classifier"].unique()
    selectors = selectors[selectors != "All"]
    result_df = pd.DataFrame()

    for dataset in datasets:
        logging.info("Processing %s" % dataset)

        for classifier in classifiers:
            for selector in selectors:
                melted = None

                folds = df[(df["Feature selector"] == selector) &
                           (df["Dataset"] == dataset) &
                           (df["Classifier"] == classifier)]
                folds_num = folds.shape[0]
                fold_features = []

                for index, fold in folds.iterrows():
                    features_str = re.sub(r"\s+", ", ", fold["Selected "
                                                            "features"])
                    feature_list = ast.literal_eval(features_str)
                    fold_features.append(feature_list)

                selected = np.sum(fold_features, axis=0)
                non_selected = -selected + folds_num
                mat = np.vstack((non_selected, selected)).T
                f_nums= np.where(np.all(fold_features, axis=0))
                common_features = np.array(pdb.columns.values)[f_nums]
                melted = pd.DataFrame({"Dataset": [dataset],
                                       "Classifier": [classifier],
                                       "Feature selector": [selector],
                                       "Common Features": [np.array2string(common_features, max_line_width=1000000)],
                                       "Fleiss Kappa": [computeKappa(mat)]})

                if melted is not None:
                    result_df = result_df.append(melted)

    result_df.to_csv(os.path.join(DATA_PATH, RESULT_FILE))


