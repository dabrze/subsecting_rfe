# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>
# License: MIT

import os
import pandas as pd
import ast
import math
from evaluation import _step_num_from_results
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

DATA_PATH = os.path.join(os.path.dirname(__file__),
                         "../experiments/results")

if __name__ == '__main__':
    for results_file, grid_file in [
        (os.path.join(DATA_PATH, "Benchmarks.csv"),
         os.path.join(DATA_PATH, "GridScores.csv")),
        (os.path.join(DATA_PATH, "CaseStudy.csv"),
         os.path.join(DATA_PATH, "CaseGridScores.csv"))]:

        df = pd.read_csv(results_file)

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

                        for index, fold in folds.iterrows():
                            score_list = ast.literal_eval(fold["Grid scores"])
                            if selector == "RFE-1":
                                step = 1
                            else:
                                step_num = _step_num_from_results(dataset, classifier,
                                                              selector, results_file,
                                                              index % folds.shape[0])
                                step = feature_num // step_num + 1
                            feature_list = list(range(feature_num, 0, -int(step)))
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

        result_df.to_csv(grid_file)
