"""Run experiments

Script to run the full experimental pipeline. Should be run after dataset preparation, as this
script requires the prediction datasets as inputs. Saves its results for evaluation.

Usage: python -m run_experiments --help
"""


import argparse
import itertools
import multiprocessing
import pathlib
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import tqdm

import afs
import data_handling
import prediction


# Different components of the experimental design, excluding search methods for alternatives
# (defined below) and prediction models (defined in module "prediction").
CV_FOLDS = 10
FEATURE_SELECTOR_TYPES = [afs.MISelector, afs.FCBFSelector, afs.ModelImportanceSelector,
                          afs.GreedyWrapperSelector]
K_VALUES = [5, 10]
TAU_VALUES = [x / 10 for x in range(1, 11)]
NUM_ALTERNATIVES_VALUES = list(range(1, 11))


# Define experimental design as cross-product of various experimental components, in particular,
# datasets, feature-selection methods, and search-methods for alternatives. (Missing: prediction
# models, which will be iterated over later.) The resulting configurations can be run parallelized.
def define_experimental_settings(data_dir: pathlib.Path) -> List[Dict[str, Any]]:
    results = []
    dataset_names_list = data_handling.list_datasets(directory=data_dir)
    for dataset_name, feature_selector_types, k, tau in itertools.product(
            dataset_names_list, FEATURE_SELECTOR_TYPES, K_VALUES, TAU_VALUES):
        base_setting = {'dataset_name': dataset_name, 'data_dir': data_dir}
        # For sequential search, returning more alternatives does not affect prior results, so using
        # max number of alternatives suffices; for simultaneous search, this is not the case.
        results.append({**base_setting, 'alternatives_func': 'search_sequentially',
                        'alternatives_args': {'k': k, 'tau': tau,
                                              'num_alternatives': max(NUM_ALTERNATIVES_VALUES)},
                       'feature_selector_type': feature_selector_types})
        for num_alternatives in NUM_ALTERNATIVES_VALUES:
            results.append({**base_setting, 'alternatives_func': 'search_simultaneously',
                            'alternatives_args': {'k': k, 'tau': tau,
                                                  'num_alternatives': num_alternatives},
                            'feature_selector_type': feature_selector_types})
    return results


# Evaluate one approach to find alternatives for one dataset and feature-selection method. The
# datasets with the "dataset_name" is read in from the "data_dir". "alternatives_func" does the
# search for alternative feature sets. All the arguments of the search (including the
# feature-selection method, which determines the objective of the search) are hidden in
# "alternatives_args".
# Return a table with various evaluation metrics, including parametrization of the search,
# objective value, and prediction performance with the feature sets found.
def run_experimental_setting(
        dataset_name: str, data_dir: pathlib.Path, alternatives_func: str, alternatives_args: Dict[str, Any],
        feature_selector_type: Type[afs.AlternativeFeatureSelector]) -> pd.DataFrame:
    results = []
    X, y = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    feature_selector = feature_selector_type()
    alternatives_func = getattr(feature_selector, alternatives_func)
    for fold_id, (train_idx, test_idx) in enumerate(prediction.split_for_pipeline(X=X, y=y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        feature_selector.set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        result = alternatives_func(**alternatives_args)
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
        result['fold_id'] = fold_id
        results.append(result)
    results = pd.concat(results, ignore_index=True)
    results['dataset_name'] = dataset_name
    results['alternatives_func'] = alternatives_func.__name__
    results['fs_func'] = feature_selector_type.__name__
    results['n'] = X.shape[1]
    results['k'] = alternatives_args['k']
    results['tau'] = alternatives_args['tau']
    results['num_alternatives'] = alternatives_args['num_alternatives']
    return results


# Main-routine: run complete experimental pipeline. This pipeline considers the cross-product of
# datasets, feature-selection methods, settings for finding alternatives, and prediction models.
# To that end, read datasets from "data_dir", save results to "results_dir". "n_processes" controls
# parallelization.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Dataset directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    experimental_settings_list = define_experimental_settings(data_dir=data_dir)
    progress_bar = tqdm.tqdm(total=len(experimental_settings_list))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(run_experimental_setting, kwds=experimental_setting,
                                        callback=lambda x: progress_bar.update())
               for experimental_setting in experimental_settings_list]
    process_pool.close()
    process_pool.join()
    results = pd.concat([x.get() for x in results])
    data_handling.save_results(results, directory=results_dir)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the complete experimental pipeline. Might take a while.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/datasets/', dest='data_dir',
                        help='Directory with input data, i.e., datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
