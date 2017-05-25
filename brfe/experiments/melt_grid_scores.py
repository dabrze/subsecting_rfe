# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import pandas as pd
import ast
import seaborn as sns
import math
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

sns.set_style("whitegrid")
DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "../experiments/results")
DATA_FILE = "Benchmarks.csv"
RESULT_FILE = "GridScores.csv"

if __name__ == '__main__':
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

                if selector in ["BRFE", "3-SRFE", "5-SRFE", "10-SRFE"]:
                    for index, fold in folds.iterrows():
                        grid_dict = ast.literal_eval(fold["Grid scores"])
                        melted = pd.DataFrame(grid_dict)
                        melted = melted.mean().rename("Accuracy")

                        melted = melted.to_frame()
                        melted.loc[:, "Feature num"] = list(melted.index)
                        melted.loc[:, "Dataset"] = dataset
                        melted.loc[:, "Classifier"] = classifier
                        melted.loc[:, "Fold"] = index % folds.shape[0]
                        melted.loc[:, "Feature selector"] = selector

                        if melted is not None:
                            result_df = result_df.append(melted)
                else:
                    feature_num = folds["Feature num"].iloc[0]

                    if selector == "RFE-log":
                        log_steps = math.log(feature_num, 2) // 1
                        step = int(feature_num // log_steps)
                    else:
                        log_steps = math.log(feature_num, 2) // 1
                        log_base = int(selector.split("-")[2])
                        log_steps = log_steps = (math.log(feature_num,
                                                          (log_base+1)/2)
                                                 * log_base // 1)
                        step = int(feature_num // log_steps)

                    for index, fold in folds.iterrows():
                        score_list = ast.literal_eval(fold["Grid scores"])
                        feature_list = list(range(feature_num, 0, -step))
                        if feature_list[-1] > 1:
                            feature_list.append(1)
                        feature_list = list(reversed(feature_list))

                        melted = pd.DataFrame({"Accuracy": score_list,
                                               "Feature num": feature_list})
                        melted.loc[:, "Dataset"] = dataset
                        melted.loc[:, "Classifier"] = classifier
                        melted.loc[:, "Fold"] = index % folds.shape[0]
                        melted.loc[:, "Feature selector"] = selector

                        if melted is not None:
                            result_df = result_df.append(melted)

    result_df.to_csv(os.path.join(DATA_PATH, RESULT_FILE))
