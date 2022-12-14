"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
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


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.max_open_warning'] = 0


# Main-routine: run complete evaluation pipeline. To that end, read results from the "results_dir"
# and some dataset information from "data_dir". Save plots to the "plot_dir". Print some statistics
# to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
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
    quality_metrics = [x for x in results.columns if 'train' in x or 'test' in x]
    metric_name_mapping = {'train_objective': '$Q_{\\mathrm{train}}$',
                           'test_objective': '$Q_{\\mathrm{test}}$',
                           'decision_tree_train_mcc': '$MCC_{\\mathrm{train}}^{\\mathrm{tree}}$',
                           'decision_tree_test_mcc': '$MCC_{\\mathrm{test}}^{\\mathrm{tree}}$',
                           'random_forest_train_mcc': '$MCC_{\\mathrm{train}}^{\\mathrm{forest}}$',
                           'random_forest_test_mcc': '$MCC_{\\mathrm{test}}^{\\mathrm{forest}}$'}
    # Number the alternatives (however, they only have a natural order in sequential search)
    results['n_alternative'] = results.groupby(group_cols).cumcount()

    print('\n-------- Experimental Design --------')

    print('\n------ Methods ------')

    print('\n---- Alternatives (Constraints) ----')

    print('\nHow is the optimization status distributed (for all experimental settings)?')
    print(results['optimization_status'].value_counts(normalize=True).apply('{:.2%}'.format))

    print('\n------ Datasets ------')

    # Table 1 (arXiv version)
    print('\n## Table: Dataset overview ##\n')
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    dataset_overview = dataset_overview[['dataset', 'n_instances', 'n_features']]
    dataset_overview.rename(columns={'dataset': 'Dataset', 'n_instances': 'm',
                                     'n_features': 'n'}, inplace=True)
    dataset_overview['Dataset'] = dataset_overview['Dataset'].str.replace('GAMETES', 'G')
    dataset_overview.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(dataset_overview.style.format(escape='latex').hide(axis='index').to_latex(hrules=True))

    print('\n-------- Evaluation --------')

    print('\n------ Datasets ------')

    print('\nHow is the median feature-set quality per dataset distributed (for all experimental',
          'settings)?')
    print(results.groupby('dataset_name')[['train_objective', 'decision_tree_test_mcc']].median(
        ).describe().round(2))

    print('\nHow does the feature set-quality (Spearman-)correlate with dataset dimensionality "n"',
          '(for all experimental settings)?')
    print(results[quality_metrics].corrwith(results['n'], method='spearman').round(2))

    print('\nHow does the feature set-quality (Spearman-)correlate with relative feature-set size',
          '"k/n" (for all experimental settings)?')
    print(results[quality_metrics].corrwith(results['k'] / results['n'], method='spearman').round(2))

    # Figure 1 (arXiv version): Feature-set quality by feature-set size "k" and dataset size "n"
    plot_results = results[(results['search_name'] == 'seq.') & (results['tau_abs'] == 1) &
                           (results['n_alternative'] == 0) & (results['fs_name'] == 'MI')].copy()
    plot_results['k/n'] = plot_results['k'] / plot_results['n']
    plot_metrics = ['train_objective', 'decision_tree_test_mcc']
    plot_results = plot_results.groupby(['dataset_name', 'k', 'k/n'])[plot_metrics].mean(
        ).reset_index()  # average over cross-validation folds (per dataset and k)
    for metric in plot_metrics:
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.scatterplot(x='k/n', y=metric, hue='k', style='k', data=plot_results, palette='Set2')
        plt.xlabel('Relative feature-set size $k/n$')
        plt.ylabel(metric_name_mapping[metric])
        plt.ylim((-0.1, 1.1))
        plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
        leg = plt.legend(title='Feature-set size $k$', edgecolor='white', loc='upper left', ncols=2,
                         bbox_to_anchor=(0.3, -0.1),  columnspacing=1, handletextpad=0, framealpha=0)
        leg.get_title().set_position((-124, -22))
        plt.tight_layout()
        plt.savefig(plot_dir / f'afs-impact-dataset-k-{metric.replace("_", "-")}.pdf')

    print('\n------ Feature-Set Quality Metrics ------')

    print('\n-- Prediction models and overfitting --')

    print('\nHow is the prediction performance distributed for different prediction models (for',
          'all experimental settings)?')
    print(results[[x for x in results.columns if '_mcc' in x]].describe().round(2).transpose())

    # Figure 2a (arXiv version): Difference in feature-set quality between training set and test set
    # by evaluation metric and feature-selection method
    metric_pairs = {
        'Q': ('train_objective', 'test_objective'),
        '$MCC^{\\mathrm{tree}}$': ('decision_tree_train_mcc', 'decision_tree_test_mcc'),
        '$MCC^{\\mathrm{forest}}$': ('random_forest_train_mcc', 'random_forest_test_mcc')
    }
    plot_results = results[['fs_name'] + quality_metrics].copy()
    for metric, metric_pair in metric_pairs.items():
        plot_results[metric] = plot_results[metric_pair[0]] - plot_results[metric_pair[1]]
    plot_results = plot_results.melt(id_vars='fs_name', value_vars=list(metric_pairs.keys()),
                                     var_name='Metric', value_name='Train-test difference')
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Metric', y='Train-test difference', hue='fs_name', data=plot_results,
                palette='Set2', fliersize=1, hue_order=fs_name_plot_order)
    plt.ylim(-0.55, 1.35)
    plt.yticks(np.arange(start=-0.4, stop=1.3, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.2, -0.1),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.06, y=0.12, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-evaluation-metrics-overfitting.pdf')

    print('\n-- Correlation between evaluation metrics --')

    # Figure 2b (arXiv version): Correlation between evaluation metrics for feature-set quality
    plot_results = results[quality_metrics].corr(method='spearman').round(2)
    plot_results.rename(columns=metric_name_mapping, index=metric_name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 16
    sns.heatmap(plot_results, vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-evaluation-metrics-correlation.pdf')

    print('\n------ Feature-Selection Methods ------')

    plot_results = results[(results['search_name'] == 'seq.') & (results['tau_abs'] == 1) &
                           (results['n_alternative'] == 0)]

    print('\n-- Prediction performance --')

    print('\nHow is the decision-tree test-set prediction performance distributed for different',
          'feature-selection methods (for the original feature sets of sequential search)?')
    print(plot_results.groupby('fs_name')['decision_tree_test_mcc'].describe().round(2))

    # Figure 3a (arXiv version): Test-set prediction performance by feature-set size "k" and
    # feature-selection method
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='k', y='decision_tree_test_mcc', hue='fs_name', data=plot_results, palette='Set2',
                fliersize=1, hue_order=fs_name_plot_order)
    plt.xlabel('Feature-set size $k$')
    plt.ylabel(metric_name_mapping['decision_tree_test_mcc'])
    plt.yticks(np.arange(start=-0.4, stop=1.1, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.25, -0.1),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.07, y=0.12, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-fs-method-k-decision-tree-test-mcc.pdf')

    print('\nHow is the iteration count of greedy-wrapper feature selection distributed (for the',
          'original feature sets of sequential search)?')
    print(plot_results['wrapper_iters'].describe().round(2))

    print('\nHow is the iteration count of greedy-wrapper feature selection distributed (for all',
          'experimental settings)?')
    print(results['wrapper_iters'].describe().round(2))

    print('\nHow is the optimization status distributed for different feature-selection methods',
          '(for all experimental settings)?')
    print(pd.crosstab(results['optimization_status'], results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\nHow is the optimization status distributed for different feature-selection methods',
          '(for the original feature sets of sequential search)?')
    print(pd.crosstab(plot_results['optimization_status'], plot_results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\n-- Influence of feature-set size "k" --')

    # Figure 3b (arXiv version): Difference in feature-set quality between feature-set sizes "k" by
    # evaluation metric and feature-selection method
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']
    plot_results = plot_results[['dataset_name', 'split_idx', 'fs_name', 'k'] + plot_metrics].copy()
    plot_results = plot_results.pivot(index=['dataset_name', 'split_idx', 'fs_name'], columns='k',
                                      values=plot_metrics).reset_index()
    for metric in plot_metrics:
        plot_results[(metric, 'diff')] = plot_results[(metric, 10)] - plot_results[(metric, 5)]
        plot_results.drop(columns=[(metric, 10), (metric, 5)], inplace=True)
    plot_results = plot_results.droplevel(level='k', axis='columns')
    plot_results = plot_results.melt(id_vars='fs_name', value_vars=plot_metrics, var_name='Metric',
                                     value_name='Difference')
    plot_results['Metric'].replace(metric_name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Metric', y='Difference', hue='fs_name', data=plot_results,  palette='Set2',
                fliersize=1, hue_order=fs_name_plot_order)
    plt.ylabel('Difference $k$=10 vs. $k$=5', y=0.45)  # moved a bit downwards to fit on plot
    plt.ylim(-0.65, 0.65)
    plt.yticks(np.arange(start=-0.6, stop=0.7, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.2, -0.1),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.06, y=0.12, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-fs-method-k-metric-diff.pdf')

    print('\nWhat is the median feature-set-quality difference per experimental setting between',
          'k=10 and k=5 for different feature-selection methods (for the original feature sets of',
          'sequential search)?')
    print(plot_results.groupby(['Metric', 'fs_name']).median().round(2))

    print('\n------ Searching Alternatives ------')

    print('\n---- Search Method ----')

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
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

    print('\n-- Variance in feature-set quality --')

    print('\nWhat is the median standard deviation of feature-set quality within one search run',
          'for different feature-selection methods, search methods, and numbers of alternatives',
          '(for k=5 and 0-5 alternatives)?')
    for metric in plot_metrics:
        print(comparison_results.groupby(group_cols)[metric].std().reset_index().groupby(
            ['fs_name', 'search_name', 'num_alternatives'])[metric].median().reset_index(
                ).pivot(index=['fs_name', 'num_alternatives'], columns='search_name').round(3))

        # Figures 4a, 4c, 4e (arXiv version): Standard deviation of feature-set quality in search
        # runs by search method
        plot_results = comparison_results[comparison_results['fs_name'] == 'MI']
        plot_results = plot_results.groupby(group_cols)[metric].std().reset_index()
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette='Set2', fliersize=1, hue_order=search_name_plot_order)
        plt.xlabel('Number of alternatives $a$')
        plt.ylabel(f'$\\sigma$ of {metric_name_mapping[metric]}')
        plt.yticks(np.arange(start=0, stop=0.35, step=0.1))
        plt.ylim(-0.05, 0.35)
        plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.2, -0.1),
                   columnspacing=1, framealpha=0, ncols=2)
        plt.figtext(x=0.08, y=0.13, s='Search', rotation='vertical')
        plt.tight_layout()
        plt.savefig(plot_dir / f'afs-impact-search-stddev-{metric.replace("_", "-")}.pdf')

    print('\n-- Average value of feature-set quality --')

    print('\nWhat is the median average value of feature-set quality within one search run for',
          'different feature-selection methods, search methods, and numbers of alternatives',
          '(for k=5 and 0-5 alternatives)?')
    for metric, ylim, min_tick in zip(
            plot_metrics, [(-0.05, 1.05), (-0.05, 0.65), (-0.3, 1.05)], [0, 0, -0.2]):
        print(comparison_results.groupby(group_cols)[metric].mean().reset_index().groupby(
            ['fs_name', 'search_name', 'num_alternatives'])[metric].median().reset_index(
                ).pivot(index=['fs_name', 'num_alternatives'], columns='search_name').round(3))

        # Figures 4b, 4d, 4f (arXiv version): Average feature-set quality in search runs by search
        # method
        plot_results = comparison_results[comparison_results['fs_name'] == 'MI']
        plot_results = plot_results.groupby(group_cols)[metric].mean().reset_index()
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette='Set2', fliersize=1, hue_order=search_name_plot_order)
        plt.xlabel('Number of alternatives $a$')
        plt.ylabel(f'Mean of {metric_name_mapping[metric]}')
        plt.ylim(ylim)
        plt.yticks(np.arange(start=min_tick, stop=ylim[1], step=0.2))
        plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.2, -0.1),
                   columnspacing=1, framealpha=0, ncols=2)
        plt.figtext(x=0.08, y=0.13, s='Search', rotation='vertical')
        plt.tight_layout()
        plt.savefig(plot_dir / f'afs-impact-search-mean-{metric.replace("_", "-")}.pdf')

    # Figure 5a (arXiv version): Test-set prediction performance by search method and
    # feature-selection method
    plot_results = comparison_results[group_cols + ['decision_tree_test_mcc']].copy()
    plot_results = plot_results.groupby(group_cols)['decision_tree_test_mcc'].mean().reset_index()
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='search_name', y='decision_tree_test_mcc', hue='fs_name', data=plot_results,
                palette='Set2', fliersize=1, hue_order=fs_name_plot_order)
    plt.xlabel('Search')
    plt.xticks(rotation=10, horizontalalignment='right')
    plt.ylabel(metric_name_mapping['decision_tree_test_mcc'])
    plt.yticks(np.arange(start=-0.4, stop=1.1, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.2, -0.15),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.07, y=0.14, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-search-fs-method-decision-tree-test-mcc.pdf')

    print('\nWhat is the median feature-set-quality difference per experimental setting between',
          'simltaneous search (summed-quality objective) and sequential search for different',
          'feature-selection methods and numbers of alternatives (for k=5 and 0-5 alternatives)?')
    plot_results = comparison_results.groupby(group_cols)['train_objective'].mean().reset_index(
        ).pivot(index=[x for x in group_cols if x != 'search_name'], columns='search_name',
                values='train_objective').reset_index()
    plot_results['sim - seq'] = plot_results['sim. (sum)'] - plot_results['seq.']
    print(plot_results.groupby(['fs_name', 'num_alternatives'])['sim - seq'].median().round(3))

    # Figure 5b: Difference in feature-set quality between simultaneous and sequential search by
    # evaluation metric and feature-selection method
    plot_results = comparison_results.groupby(group_cols)[plot_metrics].mean().reset_index(
        ).pivot(index=[x for x in group_cols if x != 'search_name'], columns='search_name',
                values=plot_metrics).reset_index()
    for metric in plot_metrics:
        plot_results[(metric, 'diff')] = (plot_results[(metric, 'sim. (sum)')] -
                                          plot_results[(metric, 'seq.')])
    plot_results = plot_results.loc[:, (slice(None), ['', 'diff'])]  # keep "diff" & non-search cols
    plot_results = plot_results.droplevel(level='search_name', axis='columns')
    plot_results = plot_results.melt(id_vars='fs_name', value_vars=plot_metrics, var_name='Metric',
                                     value_name='Difference')
    plot_results['Metric'].replace(metric_name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Metric', y='Difference', hue='fs_name', data=plot_results,  palette='Set2',
                fliersize=1, hue_order=fs_name_plot_order)
    plt.ylabel('Difference sim. vs. seq.')
    plt.ylim(-0.35, 0.35)
    plt.yticks(np.arange(start=-0.3, stop=0.4, step=0.1))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.2, -0.1),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.06, y=0.12, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-search-fs-method-metric-diff.pdf')

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

    print('\nHow is the optimization status distributed for different feature-selection methods',
          '(excluding Greedy Wrapper) and search methods (for k=5 and 0-5 alternatives)?')
    print(plot_results.groupby(['fs_name', 'search_name'])['optimization_status'].value_counts(
        normalize=True).round(4).apply('{:.2%}'.format))

    # Table 3 (arXiv version)
    print('\n## Table: Optimization status by search method and feature-selection method (for k=5',
          'and 0-5 alternatives) ##\n')
    print_results = (plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_status'].value_counts(normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index=['fs_name', 'search_name'], values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    status_order = ['Infeasible', 'Not solved', 'Feasible', 'Optimal']
    print_results = print_results[print_results.columns[:2].tolist() + status_order]  # re-order
    print(print_results.style.format('{:.2f}\\%'.format, subset=status_order).hide(
        axis='index').to_latex(hrules=True))

    print('\nHow is the optimization status distributed for different numbers of alternatives',
          '(for simultaneous search with summed-quality objective, k=5, and excluding Greedy',
          'Wrapper)?')
    print(pd.crosstab(plot_results.loc[plot_results['search_name'] == 'sim. (sum)',
                                       'optimization_status'],
                      plot_results.loc[plot_results['search_name'] == 'sim. (sum)',
                                       'num_alternatives'],
                      normalize='columns').applymap('{:.2%}'.format))

    # Table 4 (arXiv version)
    print('\n## Table: Optimization status by number of alternatives (for simultaneous search with',
          'summed-quality objective, k=5, and excluding Greedy Wrapper) ##\n')
    print_results = plot_results[plot_results['search_name'] == 'sim. (sum)']
    print_results = (print_results.groupby('num_alternatives')['optimization_status'].value_counts(
        normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index='num_alternatives', values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    print_results = print_results[[print_results.columns[0]] + status_order]  # re-order
    print(print_results.style.format('{:.2f}\\%'.format, subset=status_order).hide(
        axis='index').to_latex(hrules=True))

    print('\n-- Optimization time --')

    # While sequential search has one optimization time per feature set (as the latter are found
    # separately), simultaneous search duplicates same runtime record for multiple feature sets
    # found by one search; for a fair comparison, we only extract one runtime per simultaneous
    # search and sum the runtimes of sequential search for search runs
    assert ((comparison_results[comparison_results['search_name'].str.startswith(
        'sim')].groupby(group_cols)['optimization_time'].nunique() == 1).all())
    plot_results = pd.concat([
        comparison_results[comparison_results['search_name'] == 'seq.'].groupby(
            group_cols)['optimization_time'].sum().reset_index()[
                ['fs_name', 'search_name', 'num_alternatives', 'optimization_time']
        ],
        comparison_results[comparison_results['search_name'].str.startswith('sim')].groupby(
            group_cols).first().reset_index()[
                ['fs_name', 'search_name', 'num_alternatives', 'optimization_time']
        ]
    ])

    print('\nHow is the optimization time distributed for different feature-selection methods',
          '(for sequential search with k=5 and 0-5 alternatives)?')
    print(plot_results[plot_results['search_name'] == 'seq.'].groupby('fs_name')[
        'optimization_time'].describe().round(3))

    print('\nHow is the optimization time distributed for different feature-selection methods',
          '(for simultaneous search with summed-quality objective and k=5)?')
    print(plot_results[plot_results['search_name'] == 'sim. (sum)'].groupby(
        'fs_name')['optimization_time'].describe().round(2))

    # Table 5 (arXiv version)
    print('\n## Table: Median optimization time by feature-selection method and search method',
          '(for k=5 and 0-5 alternatives) ##\n')
    print_results = plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_time'].median().reset_index()
    print_results = print_results.pivot(index='fs_name', columns='search_name')
    print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\nWhat is the median optimization time for different feature-selection methods and',
          'numbers of alternativess (for sequential search with k=5 and 0-5 alternatives)?')
    print(plot_results[plot_results['search_name'] == 'seq.'].groupby(
        ['fs_name', 'num_alternatives'])['optimization_time'].median().reset_index().pivot(
            index='num_alternatives', columns='fs_name').round(3))

    print('\nWhat is the median optimization time for different feature-selection methods and',
          'numbers of alternativess (for simultaneous search with summed-quality objective and',
          'k=5)?')
    print_results = plot_results[plot_results['search_name'] == 'sim. (sum)'].groupby(
        ['fs_name', 'num_alternatives'])['optimization_time'].median().reset_index()
    print_results = print_results.pivot(index='num_alternatives', columns='fs_name')
    print(print_results.round(3))

    # Table 6 (arXiv version)
    print('\n## Table: Median optimization time by number of alternatives and feature-selection',
          'method (for simultaneous search with summed-quality objective and k=5) ##\n')
    print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\n---- Number of Alternatives ----')

    print('\n-- Feature-set quality / Influence of feature-selection method --')

    seq_results = results[(results['search_name'] == 'seq.') & (results['k'] == 5)].copy()
    normalization_funcs = {'max': lambda x: x / x.max(),
                           'min-max': lambda x: (x - x.min()) / (x.max() - x.min())}
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

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
        norm_results = seq_results[group_cols + plot_metrics + ['n_alternative']].copy()
        normalization_name = func_name
        if fillna:  # after shifting performed above, all metrics have 0 as theoretical minimum
            norm_results[plot_metrics] = norm_results[plot_metrics].fillna(0)
            normalization_name += '-fillna'
        norm_results[plot_metrics] = norm_results.groupby(group_cols)[plot_metrics].apply(
            normalization_func)  # applies function to each column independently

        print(f'\nWhat is the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) for different iterations (alternatives) and feature-selection',
              'methods (for sequential search with k=5)?')
        for metric in plot_metrics:
            print(norm_results.groupby(['n_alternative', 'fs_name'])[metric].median().reset_index(
                ).pivot(index='n_alternative', columns='fs_name').round(2))

        # Figures 6a-6d (arXiv version): Feature-set quality by number of alternatives and
        # evaluation metric
        plot_results = norm_results[(norm_results['fs_name'] == 'MI')].melt(
            id_vars='n_alternative', value_vars=plot_metrics, var_name='Metric',
            value_name='quality')
        plot_results['Metric'].replace(metric_name_mapping, inplace=True)
        plt.figure(figsize=(8, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='n_alternative', y='quality', hue='Metric', data=plot_results,
                    palette='Set2', fliersize=1)
        plt.xlabel('Number of alternative')
        plt.ylabel('Normalized Quality', y=0.37)   # moved a bit downwards to fit on plot
        plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
        leg = plt.legend(title='Metric', edgecolor='white', loc='upper center', columnspacing=1,
                         ncols=3, bbox_to_anchor=(0.5, -0.1), handletextpad=0.2, framealpha=0)
        leg.get_title().set_position((-170, -24))
        plt.tight_layout()
        plt.savefig(plot_dir / f'afs-impact-num-alternatives-quality-{normalization_name}.pdf')

        if (func_name == 'max') and (not fillna):
            for metric in ['train_objective', 'decision_tree_test_mcc']:
                # Figures 7a, 7b (arXiv version): Feature-set quality by number of alternatives and
                # feature-selection method
                plot_results = norm_results.groupby(['n_alternative', 'fs_name'])[metric].median(
                    ).reset_index()
                plt.figure(figsize=(5, 5))
                plt.rcParams['font.size'] = 18
                sns.lineplot(x='n_alternative', y=metric, hue='fs_name', style='fs_name',
                             data=plot_results, palette='Set2', hue_order=fs_name_plot_order,
                             style_order=fs_name_plot_order)
                plt.xlabel('Number of alternative')
                plt.xticks(range(11))
                plt.ylabel(f'Normalized {metric_name_mapping[metric]}')
                plt.ylim(0.78, 1.02)
                plt.yticks(np.arange(start=0.8, stop=1.05, step=0.05))
                plt.legend(title=' ', edgecolor='white', loc='upper left', columnspacing=1, ncols=2,
                           bbox_to_anchor=(-0.2, -0.1), framealpha=0, handletextpad=0.2)
                plt.figtext(x=0.06, y=0.12, s='Selection', rotation='vertical')
                plt.tight_layout()
                plt.savefig(plot_dir / ('afs-impact-num-alternatives-fs-method-' +
                                        f'{metric.replace("_", "-")}-max.pdf'))

    print('\n-- Optimization status --')

    for k in results['k'].unique():
        print('\nHow is the optimization status distributed for different iterations',
              f'(alternatives) (for sequential search with k={k} and MI as feature-selection',
              'method)?')
        print(pd.crosstab(
            results.loc[(results['fs_name'] == 'MI') & (results['search_name'] == 'seq.') &
                        (results['k'] == k), 'n_alternative'],
            results.loc[(results['fs_name'] == 'MI') & (results['search_name'] == 'seq.') &
                        (results['k'] == k), 'optimization_status'],
            normalize='index').applymap('{:.2%}'.format))

    print('\n---- Dissimilarity Threshold ----')

    print('\n-- Feature-set quality / Influence of feature-selection method --')

    # Here, we use k=10 instead of k=5 to show more distinct values of "tau" (10 instead of 5)
    seq_results = results[(results['search_name'] == 'seq.') & (results['k'] == 10)].copy()
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

    # See above (analyzing num alternatives) for reasons for shifting values of metrics
    condition = seq_results['fs_name'].isin(('mRMR', 'Greedy Wrapper'))
    seq_results.loc[condition, ['train_objective', 'test_objective']] = (
        seq_results.loc[condition, ['train_objective', 'test_objective']] + 1) / 2
    seq_results['decision_tree_test_mcc'] = (seq_results['decision_tree_test_mcc'] + 1) / 2

    for (func_name, normalization_func), fillna in itertools.product(
            normalization_funcs.items(), (False, True)):
        norm_results = seq_results[group_cols + plot_metrics + ['n_alternative']].copy()
        normalization_name = func_name
        if fillna:  # after shifting performed above, all metrics have 0 as theoretical minimum
            norm_results[plot_metrics] = norm_results[plot_metrics].fillna(0)
            normalization_name += '-fillna'
        norm_results[plot_metrics] = norm_results.groupby(group_cols)[plot_metrics].apply(
            normalization_func)  # applies function to each column independently

        print(f'\nWhat is the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) for different dissimilarity thresholds "tau" and',
              'feature-selection methods (for sequential search with k=10)?')
        for metric in plot_metrics:
            print(norm_results.groupby(['tau_abs', 'fs_name'])[metric].median().reset_index(
                ).pivot(index='tau_abs', columns='fs_name').round(2))

        for metric in (['train_objective', 'test_objective'] if func_name == 'max'
                       else ['decision_tree_test_mcc']):
            # Figures 8a-8f (arXiv version): Feature-set quality by number of alternatives and
            # dissimilarity threshold "tau"
            plot_results = norm_results[norm_results['fs_name'] == 'MI'].groupby(
                ['n_alternative', 'tau_abs'])[metric].mean().reset_index()
            plot_results['tau'] = plot_results['tau_abs'] / 10
            plt.figure(figsize=(4, 3))
            plt.rcParams['font.size'] = 15
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

        if (func_name == 'max') and (not fillna):
            for metric in ['train_objective', 'decision_tree_test_mcc']:
                # Figures 10a, 10b (arXiv version): Feature-set quality by dissimilarity threshold
                # "tau" and feature selection-method
                plot_results = norm_results.groupby(['tau_abs', 'fs_name'])[metric].median(
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
                                        'max.pdf'))

    print('\n-- Optimization status --')

    for k in results['k'].unique():
        plot_results = results[(results['fs_name'] == 'MI') & (results['k'] == k) &
                               (results['search_name'] == 'seq.')]

        print('\nHow is the optimization status distributed for different dissimilarity thresholds',
              f'"tau" (for sequential search with k={k} and MI as feature-selection method)?')
        print(pd.crosstab(plot_results['tau_abs'], plot_results['optimization_status'],
                          normalize='index').applymap('{:.2%}'.format))

        # Figures 9a, 9b (arXiv version): Optimization status by number of alternatives and
        # dissimilarity threshold "tau"
        assert plot_results['optimization_status'].isin(['Infeasible', 'Optimal']).all()
        plot_results = plot_results.groupby(['tau_abs', 'n_alternative'])[
            'optimization_status'].agg(lambda x: (x == 'Optimal').sum() / len(x)).reset_index()
        plot_results['tau'] = plot_results['tau_abs'] / k
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
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
        plt.savefig(plot_dir / f'afs-impact-num-alternatives-tau-optimization-status-k-{k}.pdf')

    print('\n------ Composition of Feature Sets (not used in paper) ------')

    print('\nHow many new features occur in a feature set from one alternative to the next',
          '(for sequential search with k=10 and MI as feature-selection method)?')
    feature_diff_results = seq_results[['selected_idxs', 'train_objective', 'n_alternative'] +
                                       group_cols].copy()
    feature_diff_results['prev_selected_idxs'] = feature_diff_results.groupby(
        group_cols)['selected_idxs'].shift().fillna('').apply(list)
    feature_diff_results['features_added'] = feature_diff_results.apply(
        lambda x: len(set(x['selected_idxs']) - set(x['prev_selected_idxs'])), axis='columns')
    feature_diff_results['features_deleted'] = feature_diff_results.apply(
        lambda x: len(set(x['prev_selected_idxs']) - set(x['selected_idxs'])), axis='columns')
    print(feature_diff_results[
        (feature_diff_results['train_objective'].notna()) &  # current feature set not empty
        (feature_diff_results['n_alternative'] > 0) &  # previous feature set exists
        (feature_diff_results['fs_name'] == 'MI')].groupby(
            'tau_abs')['features_added'].describe().round(2))


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
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('\nPlots created and saved.')
