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
        {'GreedyWrapper': 'Greedy Wrapper', 'ModelImportance': 'Model Gain'})
    fs_name_plot_order = ['MI', 'FCBF', 'Greedy Wrapper', 'Model Gain']
    results['search_name'] = results['search_name'].str.replace('(search_|ly)', '', regex=True)
    results['optimization_status'].replace({0: 'Optimal', 1: 'Feasible', 2: 'Infeasible',
                                            6: 'Not solved'}, inplace=True)
    # Define columns for main experimental dimensions (corresponding to independent search runs):
    group_cols = ['dataset_name', 'split_idx', 'fs_name', 'search_name', 'k', 'tau_abs',
                  'num_alternatives']
    # Define columns for evaluation metrics:
    quality_metrics = [x for x in results.columns if 'train' in x or 'test' in x]
    prediction_metrics = [x for x in results.columns if '_mcc' in x]
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

    print('\nHow often do certain optimization statuses occur?')
    print(results['optimization_status'].value_counts(normalize=True).apply('{:.2%}'.format))

    print('\n------ Datasets ------')

    # Table 1 (arXiv version)
    print('\nTable: Dataset overview\n')
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    dataset_overview = dataset_overview[['dataset', 'n_instances', 'n_features']]
    dataset_overview.rename(columns={'dataset': 'Dataset', 'n_instances': 'm',
                                     'n_features': 'n'}, inplace=True)
    dataset_overview['Dataset'] = dataset_overview['Dataset'].str.replace('GAMETES', 'G')
    dataset_overview.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(dataset_overview.style.format(escape='latex').hide(axis='index').to_latex(hrules=True))

    print('\n-------- Evaluation --------')

    print('\n------ Datasets ------')

    print('\nHow does median feature-set quality (all experim. settings) differ between datasets?')
    print(results.groupby('dataset_name')[['train_objective', 'decision_tree_test_mcc']].median(
        ).describe().round(2))

    print('\nHow does feature set-quality (all experim. settings)  (Spearman-)correlate with "n"?')
    print(results[quality_metrics].corrwith(results['n'], method='spearman').round(2))

    print('\nHow does feature set-quality (all experim. settings) (Spearman-)correlate with "k/n"?')
    print(results[quality_metrics].corrwith(results['k'] / results['n'], method='spearman').round(2))

    # Figure 1 (arXiv version): Impact of feature-set size "k" and dataset size "n"
    plot_results = results[(results['search_name'] == 'sequential') & (results['tau_abs'] == 1) &
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
        plt.savefig(plot_dir / f'impact-dataset-k-{metric.replace("_", "-")}.pdf')

    print('\n------ Feature-Set Quality Metrics ------')

    print('\n-- Prediction models and overfitting --')

    print('\nHow is prediction performance distributed for different models?')
    print(results[prediction_metrics].describe().round(2).transpose())

    # Figure 2a (arXiv version): Distribution of train-test difference
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
    leg = plt.legend(title='Feature selector', edgecolor='white', loc='upper left', ncols=2,
                     bbox_to_anchor=(-0.25, -0.2), alignment='left', columnspacing=1, framealpha=0)
    plt.tight_layout()
    plt.savefig(plot_dir / 'evaluation-metrics-overfitting.pdf')

    print('\n-- Correlation between evaluation metrics --')

    # Figure 2b (arXiv version): Correlation between evaluation metrics
    plot_results = results[quality_metrics].corr(method='spearman').round(2)
    plot_results.rename(columns=metric_name_mapping, index=metric_name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 16
    sns.heatmap(plot_results, vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'evaluation-metrics-correlation.pdf')

    print('\nHow do the evaluation metrics (Spearman-)correlate for different feature selectors',
          'over all experimental settings?')
    print(results.rename(columns={'decision_tree_test_mcc': 'tree_test_mcc'}).groupby('fs_name')[
        ['train_objective', 'test_objective', 'tree_test_mcc']].corr(method='spearman').round(2))

    print('\n------ Feature-Selection Methods ------')

    plot_results = results[(results['search_name'] == 'sequential') & (results['tau_abs'] == 1) &
                           (results['n_alternative'] == 0)]

    print('\n-- Prediction performance --')

    print('\nHow does prediction performance differ between feature-selection methods for the',
          'original feature sets of sequential search?')
    for metric in prediction_metrics:
        print('\nMetric:', metric)
        print(plot_results.groupby('fs_name')[metric].describe().round(2))

    # Figure 3a (arXiv version): Impact of feature-set size "k" and feature selector on test MCC
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='k', y='decision_tree_test_mcc', hue='fs_name', data=plot_results, palette='Set2',
                fliersize=1, hue_order=fs_name_plot_order)
    plt.xlabel('Feature-set size $k$')
    plt.ylabel(metric_name_mapping['decision_tree_test_mcc'])
    plt.yticks(np.arange(start=-0.4, stop=1.1, step=0.2))
    leg = plt.legend(title='Feature selector', edgecolor='white', loc='upper left', ncols=2,
                     bbox_to_anchor=(-0.3, -0.2), alignment='left', columnspacing=1, framealpha=0)
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-fs-method-k-decision-tree-test-mcc.pdf')

    print('\nHow many iterations did wrapper search perform?')
    print(results['wrapper_iters'].describe().round(2))

    print('\nHow many iterations did sequential wrapper search perform?')
    print(results[results['search_name'] == 'sequential'].groupby('n_alternative')[
        'wrapper_iters'].describe().round(2))

    print('\nHow many iterations did sequential wrapper search perform?')
    print(results[results['search_name'] == 'simultaneous'].groupby('num_alternatives')[
        'wrapper_iters'].describe().round(2))

    print('\nHow does the optimization status differ between feature-selection methods?')
    print(pd.crosstab(results['optimization_status'], results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    plot_results = results[(results['search_name'] == 'sequential') & (results['tau_abs'] == 1) &
                           (results['n_alternative'] == 0)]

    print('\nHow does the optimization status differ between feature-selection methods for the',
          'original feature sets of sequential search?')
    print(pd.crosstab(plot_results['optimization_status'], plot_results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\n-- Influence of feature-set size "k" --')

    # Figure 3b (arXiv version): Distribution of difference of metric between feature-set size "k"
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
    leg = plt.legend(title='Feature selector', edgecolor='white', loc='upper left', ncols=2,
                     bbox_to_anchor=(-0.25, -0.2), alignment='left', columnspacing=1, framealpha=0)
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-fs-method-k-metric-diff.pdf')

    print('\nHow do the evaluation metrics differ between k=10 and k=5 on median for the original',
          'feature sets of sequential search?')
    print(plot_results.groupby(['Metric', 'fs_name']).median().round(2))

    print('\n------ Searching Alternatives ------')

    print('\n---- Search Method ----')

    comparison_results = results[(results['search_name'] == 'simultaneous') & (results['k'] == 5)]
    for num_alternatives in results.loc[results['search_name'] == 'simultaneous',
                                        'num_alternatives'].unique():
        # Extract first "num_alternatives + 1" feature sets (sequential search is only run for one
        # value of "num_alternatives", but you can get "smaller" results by subsetting)
        seq_results = results[(results['search_name'] == 'sequential') & (results['k'] == 5) &
                              (results['n_alternative'] <= num_alternatives)].copy()
        seq_results['num_alternatives'] = num_alternatives
        comparison_results = pd.concat([comparison_results, seq_results])

    print('\n-- Variance in feature-set quality --')

    # Figures 4a, 4c, 4e (arXiv version): Impact of search method on stddev of metrics
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        plot_results = comparison_results[comparison_results['fs_name'] == 'MI']
        plot_results = plot_results.groupby(group_cols)[metric].std().reset_index()
        plot_results['search_name'] = plot_results['search_name'].str.replace(
            '(uential|ultaneous)', '.', regex=True)
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette='Set2', fliersize=1)
        plt.xlabel('Number of alternatives $a$')
        plt.ylabel(f'$\\sigma$ of {metric_name_mapping[metric]}')
        plt.yticks(np.arange(start=0, stop=0.35, step=0.1))
        plt.ylim(-0.05, 0.35)
        leg = plt.legend(title='Search', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.1), columnspacing=1, framealpha=0, ncols=2)
        leg.get_title().set_position((-117, -21))
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-search-stddev-{metric.replace("_", "-")}.pdf')

    print('\nHow does the standard deviation of the training-set objective within results from one',
          'search (with k=5) depend on the feature-selection method, search method, and number of',
          'alternatives (on median)?')
    print(comparison_results.groupby(group_cols)['train_objective'].std().reset_index().groupby(
        ['fs_name', 'search_name', 'num_alternatives'])['train_objective'].median().reset_index(
            ).pivot(index=['fs_name', 'num_alternatives'], columns='search_name').round(3))

    print('\n-- Average value of feature-set quality --')

    # Figures 4b, 4d, 4f (arXiv version): Impact of search method on mean objective
    for metric, ylim, min_tick in zip(
            ['train_objective', 'test_objective', 'decision_tree_test_mcc'],
            [(-0.05, 1.05), (-0.05, 0.65), (-0.3, 1.05)], [0, 0, -0.2]):
        plot_results = comparison_results[comparison_results['fs_name'] == 'MI']
        plot_results = plot_results.groupby(group_cols)[metric].mean().reset_index()
        plot_results['search_name'] = plot_results['search_name'].str.replace(
            '(uential|ultaneous)', '.', regex=True)
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette='Set2', fliersize=1)
        plt.xlabel('Number of alternatives $a$')
        plt.ylabel(f'Mean of {metric_name_mapping[metric]}')
        plt.ylim(ylim)
        plt.yticks(np.arange(start=min_tick, stop=ylim[1], step=0.2))
        leg = plt.legend(title='Search', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.1), columnspacing=1, framealpha=0, ncols=2)
        leg.get_title().set_position((-117, -21))
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-search-mean-{metric.replace("_", "-")}.pdf')

    print('\nHow does the mean of the training-set objective within results from one search',
          '(with k=5) depend on the feature-selection method, search method, and number of',
          'alternatives (on median)?')
    print(comparison_results.groupby(group_cols)['train_objective'].mean().reset_index().groupby(
        ['fs_name', 'search_name', 'num_alternatives'])['train_objective'].median().reset_index(
            ).pivot(index=['fs_name', 'num_alternatives'], columns='search_name').round(3))

    # Figure 5a (arXiv version): Impact of search method and feature selector on test MCC
    plot_results = comparison_results[group_cols + ['decision_tree_test_mcc']].copy()
    plot_results = plot_results.groupby(group_cols)['decision_tree_test_mcc'].mean().reset_index()
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='search_name', y='decision_tree_test_mcc', hue='fs_name', data=plot_results,
                palette='Set2', fliersize=1, hue_order=fs_name_plot_order)
    plt.xlabel('Search')
    plt.ylabel(metric_name_mapping['decision_tree_test_mcc'])
    plt.yticks(np.arange(start=-0.4, stop=1.1, step=0.2))
    leg = plt.legend(title='Feature selector', edgecolor='white', loc='upper left', ncols=2,
                     bbox_to_anchor=(-0.3, -0.2), alignment='left', columnspacing=1, framealpha=0)
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-search-fs-method-decision-tree-test-mcc.pdf')

    print('\nWhat is the median of the average feature-set quality difference "simultaneous search',
          'minus sequential search" (with k=5; each comparison for the same search setting)?')
    plot_results = comparison_results.groupby(group_cols)['train_objective'].mean().reset_index(
        ).pivot(index=[x for x in group_cols if x != 'search_name'], columns='search_name',
                values='train_objective').reset_index()
    plot_results['sim - seq'] = plot_results['simultaneous'] - plot_results['sequential']
    print(plot_results.groupby(['fs_name', 'num_alternatives'])['sim - seq'].median().round(3))

    # Figure 5b: Distribution of difference of metrics between search methods
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']
    plot_results = comparison_results.groupby(group_cols)[plot_metrics].mean().reset_index(
        ).pivot(index=[x for x in group_cols if x != 'search_name'], columns='search_name',
                values=plot_metrics).reset_index()
    for metric in plot_metrics:
        plot_results[(metric, 'diff')] = (plot_results[(metric, 'simultaneous')] -
                                          plot_results[(metric, 'sequential')])
        plot_results.drop(columns=[(metric, 'sequential'), (metric, 'simultaneous')], inplace=True)
    plot_results = plot_results.droplevel(level='search_name', axis='columns')
    plot_results = plot_results.melt(id_vars='fs_name', value_vars=plot_metrics, var_name='Metric',
                                     value_name='Difference')
    plot_results['Metric'].replace(metric_name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Metric', y='Difference', hue='fs_name', data=plot_results,  palette='Set2',
                fliersize=1, hue_order=fs_name_plot_order)
    plt.ylabel('Difference sim. vs. seq.', y=0.45)  # moved a bit downwards to fit on plot
    plt.ylim(-0.45, 0.35)
    plt.yticks(np.arange(start=-0.4, stop=0.4, step=0.1))
    leg = plt.legend(title='Feature selector', edgecolor='white', loc='upper left', ncols=2,
                     bbox_to_anchor=(-0.25, -0.2), alignment='left', columnspacing=1, framealpha=0)
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-search-fs-method-metric-diff.pdf')

    print('\n-- Optimization status --')

    print('\nHow often does each optimization status occurr for each combination of search method',
          '(with k=5 and 1-5 alternatives) and feature selector (excluding greedy, where status',
          'only describes last solver run)?')
    # While sequential search has one optimization status per feature set (as they are found
    # separately), simultaneous search has same status for multiple feature sets; to not bias
    # analysis towards higher humber of alternatives, we only extract one status per sim. search
    assert ((comparison_results.loc[comparison_results['search_name'] == 'simultaneous'].groupby(
        group_cols)['optimization_status'].nunique() == 1).all())
    plot_results = pd.concat([
        comparison_results.loc[
            comparison_results['search_name'] == 'sequential',
            ['fs_name', 'search_name', 'num_alternatives', 'optimization_status']
        ],
        comparison_results[comparison_results['search_name'] == 'simultaneous'].groupby(
            group_cols).first().reset_index()[
                ['fs_name', 'search_name', 'num_alternatives', 'optimization_status']
        ]
    ])
    plot_results = plot_results[plot_results['fs_name'] != 'Greedy Wrapper']
    print(plot_results.groupby(['fs_name', 'search_name'])['optimization_status'].value_counts(
        normalize=True).round(4).apply('{:.2%}'.format))

    # Table 3 (arXiv version)
    print('\nTable: Impact of search and feature selector on optimization status\n')
    print_results = (plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_status'].value_counts(normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index=['fs_name', 'search_name'], values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    status_order = ['Infeasible', 'Not solved', 'Feasible', 'Optimal']
    print_results = print_results[print_results.columns[:2].tolist() + status_order]
    print(print_results.style.format('{:.2f}\\%'.format, subset=status_order).hide(
        axis='index').to_latex(hrules=True))

    print('\nHow does the optimization status depend on the number of alternatives in simultaneous',
          'search (with k=5 and excluding greedy FS)?')
    print(pd.crosstab(plot_results.loc[plot_results['search_name'] == 'simultaneous',
                                       'optimization_status'],
                      plot_results.loc[plot_results['search_name'] == 'simultaneous',
                                       'num_alternatives'],
                      normalize='columns').applymap('{:.2%}'.format))

    # Table 4 (arXiv version)
    print('\nTable: Impact of number of alternatives on sim-search optimization status\n')
    print_results = plot_results[plot_results['search_name'] == 'simultaneous']
    print_results = (print_results.groupby('num_alternatives')['optimization_status'].value_counts(
        normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index='num_alternatives', values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    print_results = print_results[[print_results.columns[0]] + status_order]
    print(print_results.style.format('{:.2f}\\%'.format, subset=status_order).hide(
        axis='index').to_latex(hrules=True))

    print('\n-- Optimization time --')

    print('\nHow does the optimization time depend on the feature selectors in sequential search',
          '(with k=5)?')
    print(results[(results['search_name'] == 'sequential') & (results['k'] == 5)].groupby(
        'fs_name')['optimization_time'].describe().round(3))

    print('\nHow does the optimization time depend on the feature selectors in simultaneous search',
          '(with k=5)?')
    print(results[(results['search_name'] == 'simultaneous') & (results['k'] == 5)].groupby(
        'fs_name')['optimization_time'].describe().round(3))

    # Table 5 (arXiv version)
    print('\nTable: Impact of search and feature selector on optimization time\n')
    print_results = results[results['k'] == 5].groupby(['fs_name', 'search_name'])[
        'optimization_time'].mean().reset_index()
    print_results = print_results.pivot(index='fs_name', columns='search_name')
    print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\nHow does the mean optimization time develop over the number of alternatives for',
          'different feature selectors in sequential search (with k=5)?')
    print(results[(results['search_name'] == 'sequential') & (results['k'] == 5)].groupby(
        ['fs_name', 'n_alternative'])['optimization_time'].mean().reset_index().pivot(
            index='n_alternative', columns='fs_name').round(3))

    print('\nHow does the mean optimization time develop over the number of alternatives for',
          'different feature selectors in simultaneous search (with k=5)?')
    print_results = results[(results['search_name'] == 'simultaneous') & (results['k'] == 5)]
    print_results = print_results.groupby(['fs_name', 'num_alternatives'])[
        'optimization_time'].mean().reset_index()
    print_results = print_results.pivot(index='num_alternatives', columns='fs_name')
    print(print_results.round(3))

    # Table 6 (arXiv version)
    print('\nTable: Impact of search and number of alternatives on sim-search optimization time\n')
    print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\n---- Number of Alternatives ----')

    print('\n-- Feature-set quality --')

    seq_results = results[(results['search_name'] == 'sequential') & (results['k'] == 5)]
    normalization_funcs = {'max': lambda x: x / x.max(),
                           'min-max': lambda x: (x - x.min()) / (x.max() - x.min())}
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

    for (func_name, normalization_func), fillna in itertools.product(
            normalization_funcs.items(), (False, True)):
        norm_results = seq_results[group_cols + plot_metrics + ['n_alternative']].copy()
        normalization_name = func_name
        if fillna:  # not all metrics have 0 as theoretical minimum, but it suffices for analysis
            norm_results[plot_metrics] = norm_results[plot_metrics].fillna(0)
            normalization_name += '-fillna'
        norm_results[plot_metrics] = norm_results.groupby(group_cols)[plot_metrics].apply(
            normalization_func)  # applies function to each column independently

        print(f'\nHow does the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) develop over the iterations of sequential search (k=5)?')
        for metric in plot_metrics:
            print('\nMetric:', metric)
            print(norm_results.groupby(['n_alternative', 'fs_name'])[metric].median().reset_index(
                ).pivot(index='n_alternative', columns='fs_name').round(2))

        # Figures 6a-6d (arXiv version): Impact of number of alternatives on objective value
        plot_results = norm_results[(norm_results['fs_name'] == 'MI')].melt(
            id_vars=['n_alternative'], value_vars=['train_objective', 'test_objective'],
            var_name='split', value_name='objective')
        plot_results['split'] = plot_results['split'].str.replace('_objective', '')
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='n_alternative', y='objective', hue='split', data=plot_results,
                    palette='Set2', fliersize=1)
        plt.xlabel('Number of alternative')
        plt.ylabel('Normalized $Q$')
        plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
        leg = plt.legend(title='Split', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.1), columnspacing=1, framealpha=0, ncols=2)
        leg.get_title().set_position((-107, -21))
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-num-alternatives-objective-{normalization_name}.pdf')

        if func_name == 'min-max':  # with pure max normalization, large negative values can occur
            # Figures 6e, 6f (arXiv version): Impact of number of alternatives on test MCC
            metric = 'decision_tree_test_mcc'
            plot_results = norm_results[norm_results['fs_name'] == 'MI']
            plt.figure(figsize=(4, 3))
            plt.rcParams['font.size'] = 15
            sns.boxplot(x='n_alternative', y=metric, data=plot_results,
                        color=sns.color_palette('Set2').as_hex()[1], fliersize=1)
            plt.xlabel('Number of alternative')
            plt.ylabel(f'Normalized {metric_name_mapping[metric]}')
            plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
            plt.tight_layout()
            plt.savefig(plot_dir / (f'impact-num-alternatives-{metric.replace("_", "-")}-' +
                                    f'{normalization_name}.pdf'))

    print('\n-- Optimization status --')

    for k in results['k'].unique():
        print('\nHow does the optimization status differ between the number of alternatives in',
              f'sequential search (with k={k}) for MI feature selection?')
        print(pd.crosstab(
            results.loc[(results['fs_name'] == 'MI') & (results['search_name'] == 'sequential') &
                        (results['k'] == k), 'n_alternative'],
            results.loc[(results['fs_name'] == 'MI') & (results['search_name'] == 'sequential') &
                        (results['k'] == k), 'optimization_status'],
            normalize='index').applymap('{:.2%}'.format))

    print('\n---- Dissimilarity Threshold ----')

    print('\n-- Feature-set quality --')

    # Here, we use k=10 instead of k=5 to have more distinct values of "tau" (10 instead of 5)
    seq_results = results[(results['search_name'] == 'sequential') & (results['k'] == 10)]

    for (func_name, normalization_func), fillna in itertools.product(
            normalization_funcs.items(), (False, True)):
        if func_name == 'max':
            plot_metrics = ['train_objective', 'test_objective']
        else:
            plot_metrics = ['decision_tree_test_mcc']
        norm_results = seq_results[group_cols + plot_metrics + ['n_alternative']].copy()
        normalization_name = func_name
        if fillna:  # not all metrics have 0 as theoretical minimum, but it suffices for analysis
            norm_results[plot_metrics] = norm_results[plot_metrics].fillna(0)
            normalization_name += '-fillna'
        norm_results[plot_metrics] = norm_results.groupby(group_cols)[plot_metrics].apply(
            normalization_func)  # applies function to each column independently

        print(f'\nHow does the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) develop over the iterations and "tau" of sequential search',
              '(k=10) for MI feature selection?')
        for metric in plot_metrics:
            print(norm_results[norm_results['fs_name'] == 'MI'].groupby(
                ['n_alternative', 'tau_abs'])[metric].median().reset_index(
                ).pivot(index='n_alternative', columns='tau_abs').round(2))

        print(f'\nHow does the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) develop over "tau" for different feature selectors in',
              'sequential search (k=10)?')
        for metric in plot_metrics:
            print(norm_results.groupby(['tau_abs', 'fs_name'])[metric].median().reset_index(
                ).pivot(index='tau_abs', columns='fs_name').round(2))

        for metric in plot_metrics:
            # Figures 7a-7f (arXiv version): Impact of number of alternatives and "tau" on
            # feature-set quality
            plot_results = norm_results[norm_results['fs_name'] == 'MI'].groupby(
                ['n_alternative', 'tau_abs'])[metric].mean().reset_index()
            plot_results['tau'] = plot_results['tau_abs'] / 10
            plt.figure(figsize=(4, 3))
            plt.rcParams['font.size'] = 15
            sns.lineplot(x='n_alternative', y=metric, hue='tau', data=plot_results, palette='RdPu',
                         hue_norm=(-0.2, 1), legend=False)
            # Use color scale instead of standard line plot legend; start color scaling at -0.2 such
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
            plt.savefig(plot_dir / (f'impact-num-alternatives-tau-{metric.replace("_", "-")}-' +
                                    f'{normalization_name}.pdf'))

    print('\n-- Optimization status --')

    for k in results['k'].unique():
        plot_results = results[(results['fs_name'] == 'MI') & (results['k'] == k) &
                               (results['search_name'] == 'sequential')]

        print('\nHow does the optimization status differ between the dissimilarity thresholds in',
              f'sequential search (with k={k}) for MI feature selection?')
        print(pd.crosstab(plot_results['tau_abs'], plot_results['optimization_status'],
                          normalize='index').applymap('{:.2%}'.format))

        # Figures 8a, 8b (arXiv version): Impact of number of alternatives and "tau" on
        # optimization status
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
        plt.savefig(plot_dir / f'impact-num-alternatives-tau-optimization-status-k-{k}.pdf')

    print('\nHow many new features occur in a feature set from one alternative to next in',
          'sequential search (k=10) for MI as feature selector?')
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
    print('Plots created and saved.')
