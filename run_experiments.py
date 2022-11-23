"""Run experiments

Script to run the full experimental pipeline. Should be run after dataset preparation, as this
script requires the prediction datasets as inputs. Saves its results for evaluation. If some
results already exist, only runs the missing experimental settings.

Usage: python -m run_experiments --help
"""


import argparse
import itertools
import multiprocessing
import pathlib
from typing import Any, Dict, Optional, Sequence, Type

import pandas as pd
import tqdm

import afs
import data_handling
import prediction


# Different components of the experimental design, excluding the names of the search methods for
# alternatives (hard-coded below) and prediction models (queried from the module "prediction").
N_FOLDS = 10  # cross-validation for search and predictions
FEATURE_SELECTOR_TYPES = [afs.FCBFSelector, afs.GreedyWrapperSelector, afs.MISelector,
                          afs.ModelImportanceSelector, afs.MRMRSelector]
K_VALUES = [5, 10]  # sensible values of "tau" will be determined automatically
NUM_ALTERNATIVES_SEQUENTIAL = 10  # sequential search (also yields all intermediate solutions)
NUM_ALTERNATIVES_SIMULTANEOUS_VALUES = [1, 2, 3, 4, 5]  # simultaneous search
OBJECTIVE_AGG_SIMULTANEOUS_VALUES = ['min', 'sum']


# Define experimental settings as cross-product of datasets (from "data_dir"), cross-validation
# folds, and feature-selection methods. Only return those settings for which there is no results
# file in "results_dir". Provide a dictionary for calling "evaluate_feature_selector()".
def define_experimental_settings(data_dir: pathlib.Path,
                                 results_dir: pathlib.Path) -> Sequence[Dict[str, Any]]:
    experimental_settings = []
    dataset_names = data_handling.list_datasets(directory=data_dir)
    for dataset_name, split_idx, feature_selector_type in itertools.product(
            dataset_names, range(N_FOLDS), FEATURE_SELECTOR_TYPES):
        results_file = data_handling.get_results_file_path(
            directory=results_dir, dataset_name=dataset_name, split_idx=split_idx,
            fs_name=feature_selector_type.__name__)
        if not results_file.exists():
            experimental_settings.append(
                {'dataset_name': dataset_name, 'data_dir': data_dir, 'results_dir': results_dir,
                 'split_idx': split_idx, 'feature_selector_type': feature_selector_type})
    return experimental_settings


# Evaluate one search for alternatives for one feature selection method (on one split of a dataset).
# In particular, call the "afs_search_func_name" on the "feature_selector", considering the
# parameters "k", "tau_abs", "num_alternatives", and "objective_agg".
# Return a table with various evaluation metrics, including parametrization of the search,
# objective value, and prediction performance with the feature sets found.
def evaluate_one_search(feature_selector: afs.AlternativeFeatureSelector, afs_search_func_name: str,
                        k: int, tau_abs: int, num_alternatives: int,
                        objective_agg: str = 'sum') -> pd.DataFrame:
    X_train, X_test, y_train, y_test = feature_selector.get_data()
    afs_search_func = getattr(feature_selector, afs_search_func_name)
    result = afs_search_func(k=k, tau_abs=tau_abs, num_alternatives=num_alternatives,
                             objective_agg=objective_agg)
    for model_dict in prediction.MODELS:  # train each model with all feature sets found
        model = model_dict['func'](**model_dict['args'])
        prediction_performances = [prediction.train_and_evaluate(
            model=model, X_train=X_train.iloc[:, selected_idxs], y_train=y_train,
            X_test=X_test.iloc[:, selected_idxs], y_test=y_test)
            for selected_idxs in result['selected_idxs']]
        prediction_performances = pd.DataFrame(prediction_performances)
        prediction_performances.rename(columns={x: model_dict['name'] + '_' + x
                                                for x in prediction_performances.columns},
                                       inplace=True)  # put model name before metric name
        result = pd.concat([result, prediction_performances], axis='columns')
    result['k'] = k
    result['tau_abs'] = tau_abs
    result['num_alternatives'] = num_alternatives
    result['objective_agg'] = objective_agg
    result['search_name'] = afs_search_func_name
    return result


# Evaluate one feature-selection method on one split of a dataset. The dataset with the
# "dataset_name" is read in from the "data_dir" and the "split_idx"-th split is extracted.
# "feature_selector_type" is a class with methods for feature selection and search for alternatives.
# We iterate over all settings for searching alternatives.
# Return a table with various evaluation metrics, including parametrization of the search,
# objective value, and prediction performance with the feature sets found. Additionally, save this
# table to "results_dir".
def evaluate_feature_selector(
        dataset_name: str, data_dir: pathlib.Path, results_dir: pathlib.Path, split_idx: int,
        feature_selector_type: Type[afs.AlternativeFeatureSelector]) -> pd.DataFrame:
    results = []
    X, y = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    feature_selector = feature_selector_type()
    train_idx, test_idx = list(prediction.split_for_pipeline(X=X, y=y, n_splits=N_FOLDS))[split_idx]
    feature_selector.set_data(X_train=X.iloc[train_idx], X_test=X.iloc[test_idx],
                              y_train=y.iloc[train_idx], y_test=y.iloc[test_idx])
    for k in K_VALUES:
        for tau_abs in range(1, k + 1):  # all overlap sizes (except complete overlap)
            results.append(evaluate_one_search(
                feature_selector=feature_selector, afs_search_func_name='search_sequentially',
                k=k, tau_abs=tau_abs, num_alternatives=NUM_ALTERNATIVES_SEQUENTIAL))
            for num_alternatives, objective_agg in itertools.product(
                    NUM_ALTERNATIVES_SIMULTANEOUS_VALUES, OBJECTIVE_AGG_SIMULTANEOUS_VALUES):
                results.append(evaluate_one_search(
                    feature_selector=feature_selector, afs_search_func_name='search_simultaneously',
                    k=k, tau_abs=tau_abs, num_alternatives=num_alternatives,
                    objective_agg=objective_agg))
    results = pd.concat(results, ignore_index=True)
    results['fs_name'] = feature_selector_type.__name__
    results['dataset_name'] = dataset_name
    results['n'] = X.shape[1]
    results['split_idx'] = split_idx
    data_handling.save_results(results=results, directory=results_dir, dataset_name=dataset_name,
                               split_idx=split_idx, fs_name=feature_selector_type.__name__)
    return results


# Main-routine: run complete experimental pipeline. This pipeline roughly considers a cross-product
# of datasets, feature-selection methods, settings for finding alternatives, and prediction models.
# To that end, read datasets from "data_dir", save results to "results_dir". "n_processes" controls
# parallelization (over datasets, cross-validation folds, and feature-selection methods).
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Dataset directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Only missing experiments will be run.')
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


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs complete experimental pipeline except settings that already have results.',
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
