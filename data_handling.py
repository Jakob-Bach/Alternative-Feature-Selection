"""Data handling

Functions for data I/O.
"""


import pathlib
from typing import Optional, Sequence, Tuple

import pandas as pd


# Feature-part and target-part of a dataset are saved separately.
def load_dataset(dataset_name: str, directory: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(directory / (dataset_name + '_X.csv'))
    y = pd.read_csv(directory / (dataset_name + '_y.csv')).squeeze(axis='columns')
    assert isinstance(y, pd.Series)  # a DataFrame would cause errors somewhere in the pipeline
    return X, y


def save_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str, directory: pathlib.Path) -> None:
    X.to_csv(directory / (dataset_name + '_X.csv'), index=False)
    y.to_csv(directory / (dataset_name + '_y.csv'), index=False)


# List dataset names based on target-values_files.
def list_datasets(directory: pathlib.Path) -> Sequence[str]:
    return [file.name.split('_y.')[0] for file in list(directory.glob('*_y.*'))]


# Load either complete results or only a particular combi of dataset, fold, and feature selection.
def load_results(directory: pathlib.Path, dataset_name: Optional[str] = None,
                 split_idx: Optional[int] = None, fs_name: Optional[str] = None) -> pd.DataFrame:
    if (dataset_name is not None) and (split_idx is not None) and (fs_name is not None):
        return pd.read_csv(directory / (f'{dataset_name}_{split_idx}_{fs_name}_results.csv'))
    results_file = directory / 'results.csv'
    if results_file.exists():
        return pd.read_csv(results_file)
    return pd.concat([pd.read_csv(x) for x in directory.glob('*_results.*')], ignore_index=True)


def save_results(results: pd.DataFrame, directory: pathlib.Path, dataset_name: Optional[str] = None,
                 split_idx: Optional[int] = None,  fs_name: Optional[str] = None) -> None:
    if (dataset_name is not None) and (split_idx is not None) and (fs_name is not None):
        results.to_csv(directory / (f'{dataset_name}_{split_idx}_{fs_name}_results.csv'), index=False)
    results.to_csv(directory / 'results.csv', index=False)
