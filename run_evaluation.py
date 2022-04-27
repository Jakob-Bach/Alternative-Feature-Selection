"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
"""


import argparse
import ast
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

import data_handling


# Main-routine: run complete evaluation pipeline. To that end, read results from the "results_dir"
# and save plots to the "plot_dir". Prints some statistics to the console.
def evaluate(results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('Results directory does not exist.')
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if any(plot_dir.glob('*.pdf')):
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    results = data_handling.load_results(directory=results_dir)
    # Make feature sets proper lists:
    results['selected_idxs'] = results['selected_idxs'].apply(ast.literal_eval)
    # Rename selectors:
    results['fs_name'] = results['fs_name'].str.removesuffix('Selector')

    print('\n------ Experimental Design ------')

    print('\n---- Approaches ----')

    print('\n-- Feature Selection --')

    grouping = ['dataset_name', 'split_idx', 'fs_name', 'search_name', 'k', 'tau_abs',
                'num_alternatives']  # independent search runs
    results['real_k'] = results['selected_idxs'].apply(len)
    results['is_faulty'] = results['train_objective'].notna() & (results['k'] != results['real_k'])
    is_search_faulty = results.groupby(grouping)['is_faulty'].any()
    print('\n{:.2%}'.format(results['is_faulty'].sum() / len(results)),
          'feature sets violate the prescribed k.')
    print('{:.2%}'.format(is_search_faulty.sum() / len(is_search_faulty)),
          'of the searches for alternatives are affected.')
    results = results.groupby(grouping).filter(lambda x: ~x['is_faulty'].any())
    results.drop(columns=['real_k', 'is_faulty'], inplace=True)

    print('\n-- Alternatives --')

    print('\nHow often do certain optimization statuses occur?')
    print(results['optimization_status'].value_counts(normalize=True).apply(
        lambda x: '{:.2%}'.format(x)))

    print('\n------ Evaluation ------')

    print('\n---- Datasets ----')

    print('\nHow does median feature-set quality differ between datasets?')
    print(results.groupby('dataset_name')[['train_objective', 'Decision tree_test_mcc']].median(
        ).describe().round(2))

    print('\nHow does median feature-set quality differ between "n" (dataset dimensionality)?')
    print(results.groupby('n')[['train_objective', 'Decision tree_test_mcc']].median().round(2))

    quality_metrics = [x for x in results.columns if 'train' in x or 'test' in x]
    print('\nHow does feature set-quality (Spearman-)correlate with "n"?')
    print(results[quality_metrics].corrwith(results['n'], method='spearman'))

    print('\nHow does feature set-quality (Spearman-)correlate with "k"/"n"?')
    print(results[quality_metrics].corrwith(results['k'] / results['n'], method='spearman'))

    print('\n---- Prediction Models ----')

    prediction_metrics = [x for x in results.columns if '_mcc' in x]
    print('\nHow is prediction performance distributed for different models?')
    print(results[prediction_metrics].describe().round(2).transpose())

    print('\nWhat\'s the median overfitting (train-test difference)?')
    results['objective_dif'] = results['train_objective'] - results['test_objective']
    results['tree_mcc_dif'] = results['Decision tree_train_mcc'] - results['Decision tree_test_mcc']
    print(results.groupby('fs_name')[['objective_dif', 'tree_mcc_dif']].median().round(2))
    results.drop(columns=['objective_dif', 'tree_mcc_dif'], inplace=True)

    plot_results = results[quality_metrics].corr(method='spearman').round(2)
    name_mapping = {'train_objective': '$Q_{train}$', 'test_objective': '$Q_{test}$',
                    'Decision tree_train_mcc': '$MCC_{train}^{Tree}$',
                    'Decision tree_test_mcc': '$MCC_{test}^{Tree}$',
                    'Random forest_train_mcc': '$MCC_{train}^{Forest}$',
                    'Random forest_test_mcc': '$MCC_{test}^{Forest}$'}
    plot_results.rename(columns=name_mapping, index=name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 14
    sns.heatmap(plot_results, vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'evaluation-metrics-correlation.pdf')

    print('\n---- Feature-Selection Methods ----')

    print('\nHow does the optimization status differ between feature-selection methods?')
    print(pd.crosstab(results['optimization_status'], results['fs_name'],
                      normalize='columns').applymap(lambda x: '{:.2%}'.format(x)))

    print('\nHow does prediction performance differ between feature-selection methods?')
    for metric in prediction_metrics:
        print('\nMetric:', metric)
        print(results.groupby('fs_name')[metric].describe().round(2))

    print('\nHow do the results differ between k?')
    print(results.groupby('k')[['train_objective', 'Decision tree_test_mcc']].median().round(2))
    print(results.groupby(['fs_name', 'k'])[['train_objective', 'Decision tree_test_mcc']].median(
        ).round(2))


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
