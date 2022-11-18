"""Prepare datasets

Script to download and save datasets from PMLB, so they can be used in the experimental pipeline.

Usage: python -m prepare_datasets --help
"""


import argparse
import pathlib

import pandas as pd
import pmlb
import tqdm

import data_handling


# Manually defined by looking for similar dataset names and dataset properties on the website
# https://epistasislab.github.io/pmlb/index.html
DUPLICATE_DATASETS = ['agaricus_lepiota', 'breast_cancer_wisconsin', 'buggyCrx', 'colic', 'crx',
                      'german', 'Hill_Valley_without_noise', 'kr_vs_kp', 'vote']


# Main-routine: download, pre-process, and save (to "data_dir") datasets from PMLB.
def prepare_datasets(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Dataset directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Dataset directory is not empty. Files might be overwritten, but not deleted.')

    # Get an overview of datasets and filter it:
    dataset_overview = pmlb.dataset_lists.df_summary
    dataset_overview = dataset_overview[
        (dataset_overview['task'] == 'classification') &
        (dataset_overview['n_classes'] == 2) &
        (dataset_overview['n_instances'] >= 100) &
        (dataset_overview['n_features'] >= 15) &
        (dataset_overview['n_features'] <= 200)
    ]  # filtering steps described in paper
    assert pd.Series(DUPLICATE_DATASETS).isin(dataset_overview['dataset']).all()  # check for typos
    dataset_overview = dataset_overview[~dataset_overview['dataset'].isin(DUPLICATE_DATASETS)]
    assert len(dataset_overview) == 30  # if this changes, we would need to adapt paper as well
    data_handling.save_dataset_overview(dataset_overview=dataset_overview, directory=data_dir)

    # Save individual datasets:
    print('Downloading and saving datasets ...')
    for dataset_name in tqdm.tqdm(dataset_overview['dataset']):
        dataset = pmlb.fetch_data(dataset_name=dataset_name, dropna=False)
        assert dataset.notna().all().all()  # datasets we chose don't contain missing values
        data_handling.save_dataset(X=dataset.drop(columns='target'), y=dataset['target'],
                                   dataset_name=dataset_name, directory=data_dir)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves datasets from PMLB, prepares them for the experiment pipeline and ' +
        'stores them in the specified directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/datasets/',
                        dest='data_dir', help='Directory to store prediction datasets.')
    print('Dataset preparation started.')
    prepare_datasets(**vars(parser.parse_args()))
    print('Datasets prepared and saved.')
