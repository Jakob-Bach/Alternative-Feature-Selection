"""Run evaluation

Script to compute summary statistics and create plots for the CIKM version of the paper. Should be
run after the experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation_cikm --help
"""


import argparse
import ast
import itertools
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns

import data_handling


plt.rcParams['font.family'] = 'Linux Biolinum'  # fits to serif font "Libertine" from ACM template
plt.rcParams['figure.max_open_warning'] = 0


# Main-routine: run complete evaluation pipeline. To that end, read results from the "results_dir"
# and some dataset information from "data_dir". Save plots to the "plot_dir". Print some statistics
# to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('The results directory does not exist.')
    if not plot_dir.is_dir():
        print('The plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if any(plot_dir.glob('*.pdf')):
        print('The plot directory is not empty. Files might be overwritten but not deleted.')

    results = data_handling.load_results(directory=results_dir)

    # Make feature sets proper lists:
    results['selected_idxs'] = results['selected_idxs'].apply(ast.literal_eval)
    # Sanity check: correct number of feature selected
    assert ((results['train_objective'].isna() & (results['selected_idxs'].apply(len) == 0)) |
            (results['selected_idxs'].apply(len) == results['k'])).all()
    # Rename some values:
    results['fs_name'] = results['fs_name'].str.removesuffix('Selector').replace(
        {'GreedyWrapper': 'Greedy Wrapper', 'ModelImportance': 'Model Gain', 'MRMR': 'mRMR'})
    fs_name_plot_order = ['MI', 'FCBF', 'mRMR', 'Model Gain', 'Greedy Wrapper']
    results['search_name'] = results['search_name'].replace({'search_sequentially': 'seq.',
                                                             'search_simultaneously': 'sim.'})
    results.loc[results['search_name'] == 'sim.', 'search_name'] = (
        'sim. (' + results.loc[results['search_name'] == 'sim.', 'objective_agg'] + ')')
    results.drop(columns='objective_agg', inplace=True)
    search_name_plot_order = ['sim. (min)', 'sim. (sum)', 'seq.']
    results['optimization_status'].replace({0: 'Optimal', 1: 'Feasible', 2: 'Infeasible',
                                            6: 'Not solved'}, inplace=True)
    # Define columns for main experimental dimensions (corresponding to independent search runs):
    group_cols = ['dataset_name', 'split_idx', 'fs_name', 'search_name', 'k', 'tau_abs',
                  'num_alternatives']
    # Define columns for evaluation metrics:
    metric_name_mapping = {'train_objective': '$Q_{\\mathrm{train}}$',
                           'test_objective': '$Q_{\\mathrm{test}}$',
                           'decision_tree_test_mcc': '$MCC_{\\mathrm{test}}^{\\mathrm{tree}}$'}
    # Number the alternatives (however, they only have a natural order in sequential search)
    results['n_alternative'] = results.groupby(group_cols).cumcount()

    print('\n-------- Experimental Design --------')

    print('\n------ Methods ------')

    print('\n---- Alternatives (Constraints) ----')

    print('\nHow is the optimization status distributed (for all experimental settings)?')
    print(results['optimization_status'].value_counts(normalize=True).apply('{:.2%}'.format))

    print('\n------ Datasets ------')

    print('\nHow many instances and features do the datasets have?')
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    print(dataset_overview[['n_instances', 'n_features']].describe().round().astype(int))

    print('\n-------- Evaluation --------')

    print('\n------ Search Methods for Alternatives ------')

    comparison_results = results[(results['search_name'].str.startswith('sim')) &
                                 (results['k'] == 5)]
    for num_alternatives in results.loc[results['search_name'].str.startswith('sim'),
                                        'num_alternatives'].unique():
        # Extract first "num_alternatives + 1" feature sets (sequential search is only run for one
        # value of "num_alternatives", but you can get "smaller" results by subsetting)
        seq_results = results[(results['search_name'] == 'seq.') & (results['k'] == 5) &
                              (results['n_alternative'] <= num_alternatives)].copy()
        seq_results['num_alternatives'] = num_alternatives
        comparison_results = pd.concat([comparison_results, seq_results])

    print('\n-- Variance in feature-set quality --')

    print('\nWhat is the median standard deviation of feature-set quality within one search run',
          'for different feature-selection methods, search methods, and numbers of alternatives',
          '(for k=5 and 1-5 alternatives)?')
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        print(comparison_results.groupby(group_cols)[metric].std().reset_index().groupby(
            ['fs_name', 'search_name', 'num_alternatives'])[metric].median().reset_index(
                ).pivot(index=['fs_name', 'num_alternatives'], columns='search_name').round(3))

    # Figures 1a, 1b: Standard deviation of feature-set quality in search runs by search method
    for metric in ['train_objective', 'test_objective']:
        plot_results = comparison_results[comparison_results['fs_name'] == 'MI']
        plot_results = plot_results.groupby(group_cols)[metric].std().reset_index()
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 14
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette='RdPu', fliersize=1, hue_order=search_name_plot_order)
        plt.xlabel('Number of alternatives')
        plt.ylabel(f'$\\sigma$ of {metric_name_mapping[metric]}')
        plt.yticks(np.arange(start=0, stop=0.35, step=0.1))
        plt.ylim(-0.05, 0.35)
        plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(0, -0.1),
                   columnspacing=1, framealpha=0, ncols=2)
        plt.figtext(x=0.17, y=0.13, s='Search', rotation='vertical')
        plt.tight_layout()
        plt.savefig(plot_dir / f'afs-impact-search-stddev-{metric.replace("_", "-")}.pdf')

    print('\nHow do the evaluation metrics (Spearman-)correlate (for all experimental settings)?')
    print(results[['train_objective', 'test_objective', 'decision_tree_test_mcc']].corr(
        method='spearman').round(2))

    print('\n-- Average value of feature-set quality --')

    print('\nWhat is the median average value of feature-set quality within one search run for',
          'different feature-selection methods, search methods, and numbers of alternatives',
          '(for k=5 and 1-5 alternatives)?')
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        print(comparison_results.groupby(group_cols)[metric].mean().reset_index().groupby(
            ['fs_name', 'search_name', 'num_alternatives'])[metric].median().reset_index(
                ).pivot(index=['fs_name', 'num_alternatives'], columns='search_name').round(3))

    # Figure 1c: Average feature-set quality in search runs by search method
    metric = 'train_objective'
    plot_results = comparison_results[comparison_results['fs_name'] == 'MI']
    plot_results = plot_results.groupby(group_cols)[metric].mean().reset_index()
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14
    sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                palette='RdPu', fliersize=1, hue_order=search_name_plot_order)
    plt.xlabel('Number of alternatives')
    plt.ylabel(f'Mean of {metric_name_mapping[metric]}')
    plt.ylim((-0.05, 1.05))
    plt.yticks(np.arange(start=0, stop=1.05, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(0, -0.1),
               columnspacing=1, framealpha=0, ncols=2)
    plt.figtext(x=0.17, y=0.13, s='Search', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / f'afs-impact-search-mean-{metric.replace("_", "-")}.pdf')

    print('\nHow is the feature-set-quality difference per experimental setting between',
          'simultaneous search (sum-aggregation) and sequential search distributed for different',
          'feature-selection methods and numbers of alternatives (for k=5 and 1-5 alternatives)?')
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        plot_results = comparison_results.groupby(group_cols)[metric].mean().reset_index(
            ).pivot(index=[x for x in group_cols if x != 'search_name'], columns='search_name',
                    values=metric).reset_index()
        plot_results['sim - seq'] = plot_results['sim. (sum)'] - plot_results['seq.']
        print(f'Metric: {metric}')
        print(plot_results.groupby(['fs_name', 'num_alternatives'])['sim - seq'].agg(
            ['min', 'median', 'mean', 'max']).round(2))

    print('\n-- Optimization status --')

    # While sequential search has one optimization status per feature set (as the latter are found
    # separately), simultaneous search has same status for multiple feature sets; to not bias
    # analysis towards higher humber of alternatives, we only extract one status per sim. search
    assert ((comparison_results[comparison_results['search_name'].str.startswith(
        'sim')].groupby(group_cols)['optimization_status'].nunique() == 1).all())
    plot_results = pd.concat([
        comparison_results.loc[
            comparison_results['search_name'] == 'seq.',
            ['fs_name', 'search_name', 'num_alternatives', 'optimization_status']
        ],
        comparison_results[comparison_results['search_name'].str.startswith('sim')].groupby(
            group_cols).first().reset_index()[
                ['fs_name', 'search_name', 'num_alternatives', 'optimization_status']
        ]
    ])
    plot_results = plot_results[plot_results['fs_name'] != 'Greedy Wrapper']

    # Table 1
    print('\n## Table: Optimization status by search method and feature-selection method (for k=5',
          'and 1-5 alternatives) ##\n')
    print_results = (plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_status'].value_counts(normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index=['fs_name', 'search_name'], values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    status_order = ['Infeasible', 'Not solved', 'Feasible', 'Optimal']
    print_results = print_results[print_results.columns[:2].tolist() + status_order]  # re-order
    print(print_results.style.format('{:.2f}\\%'.format, subset=status_order).hide(
        axis='index').to_latex(hrules=True))

    print('\nHow is the optimization status distributed for different numbers of alternatives',
          '(for simultaneous search with sum-aggregation, k=5, and excluding Greedy Wrapper)?')
    print(pd.crosstab(plot_results.loc[plot_results['search_name'] == 'sim. (sum)',
                                       'optimization_status'],
                      plot_results.loc[plot_results['search_name'] == 'sim. (sum)',
                                       'num_alternatives'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\n-- Optimization time --')

    # While sequential search has one optimization time per feature set (as the latter are found
    # separately), simultaneous search duplicates same runtime record for multiple feature sets
    # found by one search; for a fair comparison, we only extract one runtime per simultaneous
    # search and sum the runtimes of sequential search for search runs
    assert ((comparison_results[comparison_results['search_name'].str.startswith(
        'sim')].groupby(group_cols)['optimization_time'].nunique() == 1).all())
    plot_results = pd.concat([
        comparison_results[comparison_results['search_name'] == 'seq.'].groupby(
            group_cols + ['n'])['optimization_time'].sum().reset_index()[
                ['n', 'fs_name', 'search_name', 'num_alternatives', 'optimization_time']
        ],
        comparison_results[comparison_results['search_name'].str.startswith('sim')].groupby(
            group_cols + ['n']).first().reset_index()[
                ['n', 'fs_name', 'search_name', 'num_alternatives', 'optimization_time']
        ]
    ])

    # Table 2
    print('\n## Table: Mean optimization time by feature-selection method and search method',
          '(for k=5 and 1-5 alternatives) ##\n')
    print_results = plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_time'].mean().reset_index()
    print_results = print_results.pivot(index='fs_name', columns='search_name')
    print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'numbers of alternatives (for sequential search with k=5 and 1-5 alternatives)?')
    print(plot_results[plot_results['search_name'] == 'seq.'].groupby(
        ['fs_name', 'num_alternatives'])['optimization_time'].mean().reset_index().pivot(
            index='num_alternatives', columns='fs_name').round(3))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'numbers of alternatives (for simultaneous search with sum-aggregation and k=5)?')
    print(plot_results[plot_results['search_name'] == 'sim. (sum)'].groupby(
        ['fs_name', 'num_alternatives'])['optimization_time'].mean().reset_index().pivot(
            index='num_alternatives', columns='fs_name').round(3))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'dataset dimensionalities "n" (for sequential search with k=5 and 1-5 alternatives)?')
    print(plot_results[plot_results['search_name'] == 'seq.'].groupby(['fs_name', 'n'])[
        'optimization_time'].mean().reset_index().pivot(index='n', columns='fs_name').round(3))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'dataset dimensionalities "n" (for simultaneous search with sum-aggregation and k=5)?')
    print(plot_results[plot_results['search_name'] == 'sim. (sum)'].groupby(['fs_name', 'n'])[
        'optimization_time'].mean().reset_index().pivot(index='n', columns='fs_name').round(3))

    print('\n------ User Parameters "a" and "tau" ------')

    print('\n-- Feature-set quality --')

    # Here, we use k=10 instead of k=5 to show more distinct values of "tau" (10 instead of 5)
    normalization_funcs = {'max': lambda x: x / x.max(),
                           'min-max': lambda x: (x - x.min()) / (x.max() - x.min())}
    seq_results = results[(results['search_name'] == 'seq.') & (results['k'] == 10)].copy()
    metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

    # Shift [-1, 1] metrics to [0, 1] first, since (1) max-normalizing with a negative max changes
    # order, e.g., [-0.5, -0.6, ..., -1] becomes [1, 1.2, ..., 2] (lower numbers get higher and
    # maximum can exceed 1) and (2) filling NAs with 0 (which we do for some of the plots to account
    # for infeasibility) makes most sense if 0 is the theoretical minimum of the metric
    condition = seq_results['fs_name'].isin(('mRMR', 'Greedy Wrapper'))
    seq_results.loc[condition, ['train_objective', 'test_objective']] = (
        seq_results.loc[condition, ['train_objective', 'test_objective']] + 1) / 2
    seq_results['decision_tree_test_mcc'] = (seq_results['decision_tree_test_mcc'] + 1) / 2

    for (func_name, normalization_func), fillna in itertools.product(
            normalization_funcs.items(), (False, True)):
        norm_results = seq_results[group_cols + metrics + ['n_alternative']].copy()
        normalization_name = func_name
        if fillna:  # after shifting performed above, all metrics have 0 as theoretical minimum
            norm_results[metrics] = norm_results[metrics].fillna(0)
            normalization_name += '-fillna'
        norm_results[metrics] = norm_results.groupby(group_cols)[metrics].apply(
            normalization_func)  # applies function to each column independently

        # Figures 2a, 2b, 2c: Feature-set quality by number of alternatives and dissimilarity
        # threshold "tau"
        plot_metrics = []
        if fillna and (func_name == 'max'):
            plot_metrics = ['train_objective', 'test_objective']
        if fillna and (func_name == 'min-max'):
            plot_metrics = ['decision_tree_test_mcc']
        for metric in plot_metrics:
            plot_results = norm_results[norm_results['fs_name'] == 'MI'].groupby(
                ['n_alternative', 'tau_abs'])[metric].mean().reset_index()
            plot_results['tau'] = plot_results['tau_abs'] / 10
            plt.figure(figsize=(5, 3))
            plt.rcParams['font.size'] = 18
            sns.lineplot(x='n_alternative', y=metric, hue='tau', data=plot_results, palette='RdPu',
                         hue_norm=(-0.2, 1), legend=False)
            # Use color scale instead of standard line plot legend; start color scaling at -0.2, so
            # that the color for the actual lowest value (tau=0) is more readable (darker):
            cbar = plt.colorbar(ax=plt.gca(), mappable=plt.cm.ScalarMappable(
                cmap="RdPu", norm=plt.Normalize(-0.2, 1)), values=plot_results['tau'].unique())
            cbar.ax.invert_yaxis()  # put low values at top (like most lines are ordered)
            cbar.ax.set_title('$\\tau}$', y=0, pad=-20, loc='left')
            cbar.ax.set_yticks(np.arange(start=0.2, stop=1.1, step=0.2))
            plt.xlabel('Number of alternative')
            plt.ylabel(f'Normalized {metric_name_mapping[metric]}')
            plt.xticks(range(0, 11, 1))
            plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig(plot_dir / (f'afs-impact-num-alternatives-tau-{metric.replace("_", "-")}-' +
                                    f'{normalization_name}.pdf'))

        if fillna and (func_name == 'max'):
            metric = 'train_objective'

            # Figure 3b: Feature-set quality by number of alternatives and feature-selection method
            plot_results = norm_results.groupby(['n_alternative', 'fs_name'])[metric].mean(
                ).reset_index()
            plt.figure(figsize=(5, 5))
            plt.rcParams['font.size'] = 18
            sns.lineplot(x='n_alternative', y=metric, hue='fs_name', style='fs_name',
                         data=plot_results, palette='Set2', hue_order=fs_name_plot_order,
                         style_order=fs_name_plot_order)
            plt.xlabel('Number of alternative')
            plt.xticks(range(11))
            plt.ylabel(f'Normalized {metric_name_mapping[metric]}')
            plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
            plt.ylim(-0.05, 1.05)
            plt.legend(title=' ', edgecolor='white', loc='upper left', columnspacing=1, ncols=2,
                       bbox_to_anchor=(-0.15, -0.1), framealpha=0, handletextpad=0.2)
            plt.figtext(x=0.06, y=0.12, s='Selection', rotation='vertical')
            plt.tight_layout()
            plt.savefig(plot_dir / ('afs-impact-num-alternatives-fs-method-' +
                                    f'{metric.replace("_", "-")}-{normalization_name}.pdf'))

            # Figure 3c: Feature-set quality by dissimilarity threshold "tau" and feature-selection
            # method
            plot_results = norm_results.groupby(['tau_abs', 'fs_name'])[metric].mean(
                ).reset_index()
            plot_results['tau'] = plot_results['tau_abs'] / 10
            plt.figure(figsize=(5, 5))
            plt.rcParams['font.size'] = 18
            sns.lineplot(x='tau', y=metric, hue='fs_name', style='fs_name',
                         data=plot_results, palette='Set2', hue_order=fs_name_plot_order,
                         style_order=fs_name_plot_order)
            plt.xlabel('$\\tau$')
            plt.xticks(np.arange(start=0.2, stop=1.1, step=0.2))
            plt.xticks(np.arange(start=0.2, stop=1.1, step=0.1), minor=True)
            plt.ylabel(f'Normalized {metric_name_mapping[metric]}')
            plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
            plt.ylim(-0.05, 1.05)
            plt.legend(title=' ', edgecolor='white', loc='upper left', columnspacing=1, ncols=2,
                       bbox_to_anchor=(-0.15, -0.1), framealpha=0, handletextpad=0.2)
            plt.figtext(x=0.06, y=0.12, s='Selection', rotation='vertical')
            plt.tight_layout()
            plt.savefig(plot_dir / (f'afs-impact-tau-fs-method-{metric.replace("_", "-")}-' +
                                    f'{normalization_name}.pdf'))

    print('\n-- Optimization status --')

    plot_results = results[(results['fs_name'] == 'MI') & (results['k'] == 10) &
                           (results['search_name'] == 'seq.')]

    # Figure 3a: Optimization status by number of alternatives and dissimilarity threshold "tau"
    assert plot_results['optimization_status'].isin(['Infeasible', 'Optimal']).all()
    plot_results = plot_results.groupby(['tau_abs', 'n_alternative'])[
        'optimization_status'].agg(lambda x: (x == 'Optimal').sum() / len(x)).reset_index()
    plot_results['tau'] = plot_results['tau_abs'] / 10
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14
    sns.lineplot(x='n_alternative', y='optimization_status', hue='tau', data=plot_results,
                 palette='RdPu', hue_norm=(-0.2, 1), legend=False)
    cbar = plt.colorbar(ax=plt.gca(), mappable=plt.cm.ScalarMappable(
        cmap="RdPu", norm=plt.Normalize(-0.2, 1)), values=plot_results['tau'].unique())
    cbar.ax.invert_yaxis()  # put low values at top (like most lines are ordered)
    cbar.ax.set_title('$\\tau}$', y=0, pad=-20, loc='left')
    cbar.ax.set_yticks(np.arange(start=0.2, stop=1.1, step=0.2))
    plt.xlabel('Number of alternative')
    plt.ylabel('Valid feature sets')
    plt.xticks(range(0, 11, 1))
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-num-alternatives-tau-optimization-status.pdf')

    print('\n-- Influence of feature-selection method --')


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates the paper\'s plots and print statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/datasets/', dest='data_dir',
                        help='Directory with prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.\n')
    evaluate(**vars(parser.parse_args()))
    print('\nPlots created and saved.')
