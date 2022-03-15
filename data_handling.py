"""Data handling

Functions for data I/O.
"""


import pathlib
from typing import Sequence, Tuple

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


# At the moment, we only have a single file containing all results.
def load_results(directory: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(directory / 'results.csv')


def save_results(results: pd.DataFrame, directory: pathlib.Path) -> None:
    results.to_csv(directory / 'results.csv', index=False)
