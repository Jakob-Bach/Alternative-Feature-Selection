"""Run experiments

Script to run the full experimental pipeline. Should be run after dataset preparation, as this
script requires the prediction datasets as inputs. Saves its results for evaluation.

Usage: python -m run_experiments --help
"""


import argparse
import multiprocessing
import pathlib
from typing import Optional

import pandas as pd
import tqdm

import alternatives
import data_handling
import feature_selection
import prediction


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
