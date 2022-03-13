"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
"""


import argparse
import pathlib

import data_handling


# Main-routine: run complete evaluation pipeline. To that end, read results from the "results_dir"
# and save plots to the "plot_dir". Prints some statistics to the console.
def evaluate(results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('Results directory does not exist.')
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the paper\'s plots and print statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')
