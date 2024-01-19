"""Run kNN experiments

Script to append kNN predictions to results from an existing run of the experimental pipeline
(the latter have to exist, else this script won't work). Overwrites the results files by appending
 two columns to each of them ("knn_train_mcc" and "knn_test_mcc").

This script is not part of the main experiments but was written when drafting a rebuttal for the
journal version of the paper, where a reviewer requested kNN as prediction model. By default, only
decision trees and random forests are used (defined by predictions.MODELS). While re-running the
full experimental pipeline (including search for feature sets and alternatives) is very costly,
only evaluating prediction performance with a new model for existing feature sets is rather cheap.

Usage: python -m run_experiments_knn --help
"""


import argparse
import ast
import itertools
import multiprocessing
import pathlib
from typing import Any, Dict, Optional, Sequence, Type

import pandas as pd
import sklearn.neighbors
import tqdm

import afs
import data_handling
import prediction


# Different components of the experimental design. kNN as prediction model is hard-coded below.
# Unlike in "run_experiments.py", settings for the search for alternatives are not defined here
# since we load existing experimental results to append kNN predictions to.
N_FOLDS = 5  # cross-validation for search and predictions
FEATURE_SELECTOR_TYPES = [afs.FCBFSelector, afs.GreedyWrapperSelector, afs.MISelector,
                          afs.ModelImportanceSelector, afs.MRMRSelector]


# Define experimental settings as cross-product of datasets (from "data_dir"), cross-validation
# folds, and feature-selection methods. Unlike in "run_experiments.py", we don't exclude existing
# experimental results, because we want to append to all of them. Provide a dictionary for calling
# "evaluate_feature_selector()".
def define_experimental_settings(data_dir: pathlib.Path,
                                 results_dir: pathlib.Path) -> Sequence[Dict[str, Any]]:
    experimental_settings = []
    dataset_names = data_handling.list_datasets(directory=data_dir)
    for dataset_name, split_idx, feature_selector_type in itertools.product(
            dataset_names, range(N_FOLDS), FEATURE_SELECTOR_TYPES):
        experimental_settings.append(
            {'dataset_name': dataset_name, 'data_dir': data_dir, 'results_dir': results_dir,
             'split_idx': split_idx, 'feature_selector_type': feature_selector_type})
    return experimental_settings


# Evaluate kNN on results for one feature-selection method on one split of a dataset. The dataset
# with the "dataset_name" is read in from the "data_dir" and the "split_idx"-th split is extracted.
# The corresponding results file, also considering the "feature_selector_type", is loaded and a kNN
# classifier is evaluated for each feature set stored in the results. Feature-selection and search
# for alternatives themselves are not run, unlike in "run_experiments.py"!
# Return the previous results table appended with two columns (train and test prediction
# performance for kNN). Additionally, save this table to "results_dir".
def evaluate_feature_selector(
        dataset_name: str, data_dir: pathlib.Path, results_dir: pathlib.Path, split_idx: int,
        feature_selector_type: Type[afs.AlternativeFeatureSelector]) -> pd.DataFrame:
    results = data_handling.load_results(
        directory=results_dir, dataset_name=dataset_name, split_idx=split_idx,
        fs_name=feature_selector_type.__name__)
    results['selected_idxs'] = results['selected_idxs'].apply(ast.literal_eval)
    X, y = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    train_idx, test_idx = list(prediction.split_for_pipeline(X=X, y=y, n_splits=N_FOLDS))[split_idx]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    model = sklearn.neighbors.KNeighborsClassifier()
    prediction_performances = [prediction.train_and_evaluate(
        model=model, X_train=X_train.iloc[:, selected_idxs], y_train=y_train,
        X_test=X_test.iloc[:, selected_idxs], y_test=y_test)
        for selected_idxs in results['selected_idxs']]
    prediction_performances = pd.DataFrame(prediction_performances)
    prediction_performances.rename(columns={x: 'knn_' + x
                                            for x in prediction_performances.columns},
                                   inplace=True)  # put model name before metric name
    results = pd.concat([results, prediction_performances], axis='columns')
    data_handling.save_results(results=results, directory=results_dir, dataset_name=dataset_name,
                               split_idx=split_idx, fs_name=feature_selector_type.__name__)
    return results


# Main-routine, same as in "run_experiments.py".
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Dataset directory does not exist.')
    if not results_dir.is_dir():
        raise FileNotFoundError('Results directory does not exist.')
    experimental_settings = define_experimental_settings(data_dir=data_dir, results_dir=results_dir)
    progress_bar = tqdm.tqdm(total=len(experimental_settings))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(
        evaluate_feature_selector, kwds=setting, callback=lambda x: progress_bar.update())
        for setting in experimental_settings]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    results = data_handling.load_results(directory=results_dir)  # merge individual results files
    data_handling.save_results(results, directory=results_dir)


# Parse some command-line arguments and run the main routine. Similar to "run_experiments.py".
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs kNN predictions and appends them to existing experimental results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/datasets/', dest='data_dir',
                        help='Directory with input data, i.e., prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
