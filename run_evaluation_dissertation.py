"""Run dissertation evaluation

Script to compute summary statistics and create plots for the dissertation. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation_dissertation --help
"""


import argparse
import ast
import pathlib
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

import data_handling


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.max_open_warning'] = 0
DEFAULT_COL_PALETTE = 'YlGnBu'


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
    results['search_name'] = results['search_name'].replace(
        {'search_sequentially': 'seq.', 'search_simultaneously': 'sim.',
         'search_greedy_balancing': 'bal.', 'search_greedy_replacement': 'rep.'})
    results.loc[results['search_name'] == 'sim.', 'search_name'] = (
        'sim. (' + results.loc[results['search_name'] == 'sim.', 'objective_agg'] + ')')
    results.drop(columns='objective_agg', inplace=True)
    search_name_hue_order_solver = ['sim. (min)', 'sim. (sum)', 'seq.']
    search_name_hue_order_all = ['sim. (sum)', 'sim. (min)', 'bal.', 'seq.', 'rep.']
    results['optimization_status'].replace({0: 'Optimal', 1: 'Feasible', 2: 'Infeasible',
                                            6: 'Not solved'}, inplace=True)
    status_order = ['Infeasible', 'Not solved', 'Feasible', 'Optimal']
    # Define columns for main experimental dimensions (corresponding to independent search runs):
    group_cols = ['dataset_name', 'split_idx', 'fs_name', 'search_name', 'k', 'tau_abs',
                  'num_alternatives']
    # Define columns for evaluation metrics:
    metric_name_mapping = {'train_objective': '$Q_{\\mathrm{train}}$',
                           'test_objective': '$Q_{\\mathrm{test}}$',
                           'decision_tree_train_mcc': '$MCC_{\\mathrm{train}}^{\\mathrm{tree}}$',
                           'decision_tree_test_mcc': '$MCC_{\\mathrm{test}}^{\\mathrm{tree}}$'}
    # Number the alternatives (however, they only have a natural order in sequential search)
    results['n_alternative'] = results.groupby(group_cols).cumcount()

    print('\n-------- 6.3 Experimental Design --------')

    print('\n------ 6.3.3 Methods ------')

    print('\n---- 6.3.3.3 Alternatives (Constraints) ----')

    print('\n-- Timeout --')

    print('\nHow is the optimization status distributed (for solver-based search)?')
    print(results.loc[results['search_name'].isin(search_name_hue_order_solver),
                      'optimization_status'].value_counts(normalize=True).apply('{:.2%}'.format))

    print('\n------ 6.3.4 Datasets ------')

    print('\n## Table 6.2: Dataset overview ##\n')
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    dataset_overview = dataset_overview[['dataset', 'n_instances', 'n_features']]
    dataset_overview['Mean corr.'] = dataset_overview['dataset'].apply(
        lambda x: data_handling.mean_feature_corr(dataset_name=x, directory=data_dir)).round(2)
    dataset_overview.rename(columns={'dataset': 'Dataset', 'n_instances': '$m$',
                                     'n_features': '$n$'}, inplace=True)
    dataset_overview['Dataset'] = dataset_overview['Dataset'].str.replace('GAMETES', 'G')
    dataset_overview['Dataset'] = dataset_overview['Dataset'].str.replace('_Epistasis', 'E')
    dataset_overview['Dataset'] = dataset_overview['Dataset'].str.replace('_Heterogeneity', 'H')
    dataset_overview.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(dataset_overview.style.format(escape='latex', precision=2).hide(axis='index').to_latex(
        hrules=True))

    print('\n-------- 6.4 Evaluation --------')

    print('\n------ 6.4.1 Feature-Selection Methods ------')

    plot_results = results[(results['search_name'] == 'seq.') & (results['tau_abs'] == 1) &
                           (results['n_alternative'] == 0)]

    print('\n-- Prediction performance --')

    print('\nHow is the decision-tree test-set prediction performance distributed for different',
          'feature-selection methods (for the original feature sets of sequential search)?')
    print(plot_results.groupby('fs_name')['decision_tree_test_mcc'].describe().round(2))

    # Figure 6.1a: Test-set prediction performance by feature-set size "k" and feature-selection
    # method
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='k', y='decision_tree_test_mcc', hue='fs_name', data=plot_results,
                palette=DEFAULT_COL_PALETTE, fliersize=1, hue_order=fs_name_plot_order)
    plt.xlabel('Feature-set size $k$')
    plt.ylabel(metric_name_mapping['decision_tree_test_mcc'])
    plt.yticks(np.arange(start=-0.4, stop=1.1, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.15, -0.1),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.06, y=0.11, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-fs-method-k-decision-tree-test-mcc.pdf')

    print('\nHow is the iteration count of greedy-wrapper feature selection distributed (for the',
          'original feature sets of sequential search)?')
    print(plot_results['wrapper_iters'].describe().round(2))

    print('\nHow is the iteration count of greedy-wrapper feature selection distributed (for all',
          'experimental settings)?')
    print(results['wrapper_iters'].describe().round(2))

    print('\nHow is the optimization status distributed for different feature-selection methods',
          '(for solver-based search)?')
    solver_results = results[results['search_name'].isin(search_name_hue_order_solver)]
    print(pd.crosstab(solver_results['optimization_status'], solver_results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\nHow is the optimization status distributed for different feature-selection methods',
          '(for the original feature sets of sequential search)?')
    print(pd.crosstab(plot_results['optimization_status'], plot_results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\n-- Influence of feature-set size "k" --')

    # Figure 6.1b: Difference in feature-set quality between feature-set sizes "k" by evaluation
    # metric and feature-selection method
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
    sns.boxplot(x='Metric', y='Difference', hue='fs_name', data=plot_results,
                palette=DEFAULT_COL_PALETTE, fliersize=1, hue_order=fs_name_plot_order)
    plt.ylabel('Difference $k$=10 vs. $k$=5', y=0.45)  # moved a bit downwards to fit on plot
    plt.ylim(-0.65, 0.65)
    plt.yticks(np.arange(start=-0.6, stop=0.7, step=0.2))
    plt.legend(title=' ', edgecolor='white', loc='upper left', bbox_to_anchor=(-0.15, -0.1),
               columnspacing=1, framealpha=0, handletextpad=0.2, ncols=2)
    plt.figtext(x=0.06, y=0.11, s='Selection', rotation='vertical')
    plt.tight_layout()
    plt.savefig(plot_dir / 'afs-impact-fs-method-k-metric-diff.pdf')

    print('\nWhat is the median feature-set-quality difference per experimental setting between',
          'k=10 and k=5 for different feature-selection methods (for the original feature sets of',
          'sequential search)?')
    print(plot_results.groupby(['Metric', 'fs_name']).median().round(2))

    print('\n------ 6.4.2 Search Methods for Alternatives ------')

    comparison_results = results[(results['k'] == 5) &
                                 results['search_name'].isin(['sim. (min)', 'sim. (sum)', 'bal.'])]
    for num_alternatives in results.loc[results['search_name'].str.startswith('sim'),
                                        'num_alternatives'].unique():
        # Extract first "num_alternatives + 1" feature sets (solver-based sequential search and
        # Greedy Replacement are only run for one value of "num_alternatives", but you can get
        # results for smaller "a" by subsetting)
        seq_results = results[results['search_name'].isin(['seq.', 'rep.']) & (results['k'] == 5) &
                              (results['n_alternative'] <= num_alternatives)].copy()
        seq_results['num_alternatives'] = num_alternatives
        comparison_results = pd.concat([comparison_results, seq_results])
    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

    print('\n-- Variance in feature-set quality --')

    print('\nWhat is the median standard deviation of feature-set quality within one search run',
          'for different feature-selection methods, search methods, and numbers of alternatives',
          '(for k=5 and 1-5 alternatives)?')
    for metric in plot_metrics:
        print(comparison_results.groupby(group_cols)[metric].std().reset_index().rename(
            columns={'num_alternatives': 'a'}).groupby(['fs_name', 'search_name', 'a'])[
                metric].median().reset_index().pivot(
                    index=['fs_name', 'a'], columns='search_name').round(3))

        # Figures 6.2a-6.2c: Standard dev. of feature-set quality in search runs by search method
        plot_results = comparison_results[comparison_results['fs_name'] == 'Model Gain']
        plot_results = plot_results.groupby(group_cols)[metric].std().reset_index()
        plt.figure(figsize=(8, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette=DEFAULT_COL_PALETTE, fliersize=1,
                    hue_order=search_name_hue_order_all)
        plt.xlabel('Number of alternatives $a$')
        plt.ylabel(f'$\\sigma$ of {metric_name_mapping[metric]}')
        plt.yticks(np.arange(start=0, stop=0.35, step=0.1))
        plt.ylim(-0.05, 0.35)
        leg = plt.legend(title='Search', edgecolor='white', framealpha=0, loc='upper left',
                         bbox_to_anchor=(0, -0.1), columnspacing=1, handletextpad=0.3, ncol=5)
        leg.get_title().set_position((-262, -21))
        plt.tight_layout()
        plt.savefig(plot_dir / f'afs-impact-search-stddev-{metric.replace("_", "-")}.pdf')

    print('\n-- Average value of feature-set quality --')

    print('\nWhat is the median average value of feature-set quality within one search run for',
          'different feature-selection methods, search methods, and numbers of alternatives',
          '(for k=5 and 1-5 alternatives)?')
    for metric, ylim, min_tick in zip(
            plot_metrics, [(-0.05, 1.05), (-0.05, 0.65), (-0.3, 1.05)], [0, 0, -0.2]):
        print(comparison_results.groupby(group_cols)[metric].mean().reset_index().rename(
            columns={'num_alternatives': 'a'}).groupby(['fs_name', 'search_name', 'a'])[
                metric].median().reset_index().pivot(
                    index=['fs_name', 'a'], columns='search_name').round(3))

        # Figures 6.3a-6.3c: Average feature-set quality in search runs by search method
        plot_results = comparison_results[comparison_results['fs_name'] == 'Model Gain']
        plot_results = plot_results.groupby(group_cols)[metric].mean().reset_index()
        plt.figure(figsize=(8, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='num_alternatives', y=metric, hue='search_name', data=plot_results,
                    palette=DEFAULT_COL_PALETTE, fliersize=1,
                    hue_order=search_name_hue_order_all)
        plt.xlabel('Number of alternatives $a$')
        plt.ylabel(f'Mean of {metric_name_mapping[metric]}')
        plt.ylim(ylim)
        plt.yticks(np.arange(start=min_tick, stop=ylim[1], step=0.2))
        leg = plt.legend(title='Search', edgecolor='white', framealpha=0, loc='upper left',
                         bbox_to_anchor=(0, -0.1), columnspacing=1, handletextpad=0.3, ncol=5)
        leg.get_title().set_position((-262, -21))
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

    print('\n-- Quality difference of heuristics --')

    search_methods = [(('sim. (min)', 'bal.'), 'sim'), (('seq.', 'rep.'), 'seq')]
    for search_method_pair, search_method_file_infix in search_methods:
        # Filter all search settings where any search method has at least one invalid feature set
        # (while solver-based simultaneous search and Greedy Balancing always yield no or all
        # desired alternatives, solver-based sequential search and Greedy Replacement may differ in
        # their number of valid alternatives, which biases their mean quality within search runs):
        plot_results = comparison_results[comparison_results['fs_name'].isin(['MI', 'Model Gain'])]
        plot_results = plot_results.groupby([x for x in group_cols if x != 'search_name']).filter(
            lambda x: x['train_objective'].notna().all())

        plot_results = plot_results[plot_results['search_name'].isin(search_method_pair)]
        plot_results = plot_results.groupby(group_cols)[plot_metrics].mean().reset_index()
        plot_results = plot_results.pivot(index=[x for x in group_cols if x != 'search_name'],
                                          columns='search_name', values=plot_metrics).reset_index()
        plot_results = plot_results.rename(columns={'num_alternatives': 'a'})
        plot_results['tau'] = plot_results['tau_abs'] / plot_results['k']
        for metric in plot_metrics:
            plot_results[(metric, 'diff')] = (plot_results[(metric, search_method_pair[0])] -
                                              plot_results[(metric, search_method_pair[1])])
        plot_results = plot_results.loc[:, (slice(None), ['', 'diff'])]  # keep "diff" & non-search
        plot_results = plot_results.droplevel(level='search_name', axis='columns')

        # Not all feature-selection methods compared; still retain their global order and color:
        heu_fs_name_plot_order = [x for x in fs_name_plot_order
                                  if x in plot_results['fs_name'].unique()]
        heu_fs_col_palette = [col for col, fs_name in zip(
            sns.color_palette(DEFAULT_COL_PALETTE, len(fs_name_plot_order)), fs_name_plot_order
            ) if fs_name in heu_fs_name_plot_order]

        parameters = [('a', 'Number of alternatives $a$', 'num-alternatives'),
                      ('tau', 'Dissimilarity threshold $\\tau$', 'tau')]
        for parameter, parameter_label, parameter_file_infix in parameters:
            print('\nHow is the difference in feature-set quality per experimental setting',
                  f'between "{search_method_pair[0]}" search and "{search_method_pair[1]}" search',
                  f'distributed for different feature-selection methods and "{parameter}" (for',
                  'k=5 and 1-5 alternatives)?')
            for metric in plot_metrics:
                print('Metric:', metric)
                print(plot_results.groupby(['fs_name', parameter])[metric].describe().drop(
                    columns='count').round(3))

            # Figures 6.4a-6.4d: Difference in feature-set quality between exact and heuristic
            # search over user paramerers, by feature-selection method and search method
            metric = 'train_objective'
            plt.figure(figsize=(5, 5))
            plt.rcParams['font.size'] = 18
            sns.boxplot(x=parameter, y=metric, hue='fs_name', data=plot_results,
                        palette=heu_fs_col_palette, hue_order=heu_fs_name_plot_order, fliersize=1)
            plt.xlabel(parameter_label)
            plt.ylabel(f'$\\Delta${metric_name_mapping[metric]} ({search_method_pair[0]} ' +
                       f'vs. {search_method_pair[1]})')
            plt.ylim(-0.13, 0.13)
            plt.yticks(np.arange(start=-0.12, stop=0.14, step=0.04))
            leg = plt.legend(title='Selection', edgecolor='white', loc='upper left',
                             bbox_to_anchor=(0, -0.1), columnspacing=1, framealpha=0,
                             handletextpad=0.2, ncols=2)
            leg.get_title().set_position((-161, -26))
            plt.tight_layout()
            plt.savefig(plot_dir / ('afs-impact-search-heuristics-metric-diff-' +
                                    f'{search_method_file_infix}-{parameter_file_infix}.pdf'))

    print('\n-- Optimization status --')

    # To not bias analysis regarding the number of alternatives (simultaneous-search results
    # duplicate optimization statuses within search runs, sequential-search results with higher
    # "a" always contains results from lower "a" as well), we only extract one status for each
    # dataset, cross-validation fold, feature-selection method, search method, "a", and "tau"
    plot_results = comparison_results.loc[
        comparison_results['num_alternatives'] == comparison_results['n_alternative'],
        ['fs_name', 'search_name', 'num_alternatives', 'optimization_status']
    ]
    plot_results = plot_results[plot_results['fs_name'] != 'Greedy Wrapper']

    print('\nHow is the optimization status distributed for different feature-selection methods',
          '(excluding Greedy Wrapper) and search methods (for k=5 and 1-5 alternatives)?')
    print(plot_results.groupby(['fs_name', 'search_name'])['optimization_status'].value_counts(
        normalize=True).round(4).apply('{:.2%}'.format))

    print('\n## Table 6.3: Optimization status by search method and feature-selection method (for',
          'k=5 and 1-5 alternatives) ##\n')
    print_results = (plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_status'].value_counts(normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index=['fs_name', 'search_name'], values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    col_order = [x for x in status_order if x in print_results.columns]  # some might not occur
    print_results = print_results[print_results.columns[:2].tolist() + col_order]  # re-order
    print(print_results.style.format('{:.2f}\\%'.format, subset=col_order).hide(
        axis='index').to_latex(hrules=True))

    print('\nHow is the optimization status distributed for different numbers of alternatives',
          '(for k=5 and excluding Greedy Wrapper)?')
    for search_name in search_name_hue_order_all:
        print('\nSearch methhod:', search_name)
        print(pd.crosstab(plot_results.loc[plot_results['search_name'] == search_name,
                                           'optimization_status'],
                          plot_results.loc[plot_results['search_name'] == search_name,
                                           'num_alternatives'],
                          normalize='columns').applymap('{:.2%}'.format))

    print('\n## Table 6.4: Optimization status by number of alternatives (for simultaneous search',
          'with sum-aggregation, k=5, and excluding Greedy Wrapper) ##\n')
    print_results = plot_results[plot_results['search_name'] == 'sim. (sum)']
    print_results = (print_results.groupby('num_alternatives')['optimization_status'].value_counts(
        normalize=True) * 100).rename('Frequency').reset_index()
    print_results = print_results.pivot(index='num_alternatives', values='Frequency',
                                        columns='optimization_status').fillna(0).reset_index()
    col_order = [x for x in status_order if x in print_results.columns]  # some might not occur
    print_results = print_results[[print_results.columns[0]] + col_order]  # re-order
    print(print_results.style.format('{:.2f}\\%'.format, subset=col_order).hide(
        axis='index').to_latex(hrules=True))

    print('\n-- Optimization time --')

    # While sequential search has one optimization time per feature set, simultaneous search and
    # the two heuristic search methods duplicate the same runtime record for multiple feature sets
    # found by one search; for a fair comparison, we extract only one runtime for each of these
    # searches and sum the runtimes of sequential search runs
    assert ((comparison_results[comparison_results['search_name'] != 'seq.'].groupby(group_cols)[
        'optimization_time'].nunique() == 1).all())
    plot_results = pd.concat([
        comparison_results[comparison_results['search_name'] == 'seq.'].groupby(
            group_cols + ['n'])['optimization_time'].sum().reset_index()[
                ['n', 'fs_name', 'search_name', 'num_alternatives', 'optimization_time']
        ],
        comparison_results[comparison_results['search_name'] != 'seq.'].groupby(
            group_cols + ['n']).first().reset_index()[
                ['n', 'fs_name', 'search_name', 'num_alternatives', 'optimization_time']
        ]
    ])

    print('\nHow is the optimization time distributed for different feature-selection methods',
          '(for k=5 and 1-5 alternatives)?')
    for search_name in search_name_hue_order_all:
        print('\nSearch methhod:', search_name)
        print(plot_results[plot_results['search_name'] == search_name].groupby('fs_name')[
            'optimization_time'].describe().round(3))

    print('\n## Table 6.5: Mean optimization time by feature-selection method and search method',
          '(for k=5 and 1-5 alternatives) ##\n')
    print_results = plot_results.groupby(['fs_name', 'search_name'])[
        'optimization_time'].mean().reset_index()
    print_results = print_results.pivot(index='fs_name', columns='search_name')
    print(print_results.style.format('{:.2f}~s'.format, na_rep='---').to_latex(hrules=True))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'numbers of alternatives (for k=5 and 1-5 alternatives)?')
    for search_name in search_name_hue_order_all:
        print('\nSearch methhod:', search_name)
        print(plot_results[plot_results['search_name'] == search_name].groupby(
            ['fs_name', 'num_alternatives'])['optimization_time'].mean().reset_index().pivot(
                index='num_alternatives', columns='fs_name').round(3))

    print('\n## Table 6.6: Mean optimization time by number of alternatives and feature-selection',
          'method (for simultaneous search with sum-aggregation and k=5) ##\n')
    print_results = plot_results[plot_results['search_name'] == 'sim. (sum)'].groupby(
        ['fs_name', 'num_alternatives'])['optimization_time'].mean().reset_index()
    print_results = print_results.pivot(index='num_alternatives', columns='fs_name')
    print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'dataset dimensionalities "n" (for sequential search with k=5 and 1-5 alternatives)?')
    print(plot_results[plot_results['search_name'] == 'seq.'].groupby(['fs_name', 'n'])[
        'optimization_time'].mean().reset_index().pivot(index='n', columns='fs_name').round(3))

    print('\nWhat is the mean optimization time for different feature-selection methods and',
          'dataset dimensionalities "n" (for simultaneous search with sum-aggregation and k=5)?')
    print(plot_results[plot_results['search_name'] == 'sim. (sum)'].groupby(['fs_name', 'n'])[
        'optimization_time'].mean().reset_index().pivot(index='n', columns='fs_name').round(3))

    print('\n------ 6.4.3 User Parameters "a" and "tau" ------')

    print('\n-- Feature-set quality / Influence of feature-selection method --')

    plot_metrics = ['train_objective', 'test_objective', 'decision_tree_test_mcc']

    for fillna in (False, True):
        # Here, we use k=10 instead of k=5 to show more distinct values of "tau" (10 instead of 5)
        norm_results = results.loc[(results['search_name'] == 'seq.') & (results['k'] == 10),
                                   group_cols + plot_metrics + ['n', 'n_alternative']].copy()
        # Shift [-1, 1] metrics to [0, 1] first, since (1) normalizing with a negative max changes
        # order, e.g., [-0.5, -0.6, ..., -1] becomes [1, 1.2, ..., 2] (lower numbers get higher and
        # maximum can exceed 1) and (2) filling NAs with 0 (which we do for some of the plots to
        # account for infeasibility) makes most sense if 0 is the theoretical minimum of the metric
        condition = norm_results['fs_name'].isin(('mRMR', 'Greedy Wrapper'))
        norm_results.loc[condition, ['train_objective', 'test_objective']] = (
            norm_results.loc[condition, ['train_objective', 'test_objective']] + 1) / 2
        norm_results['decision_tree_test_mcc'] = (norm_results['decision_tree_test_mcc'] + 1) / 2
        if fillna:  # replace quality of infeasible feature sets with 0
            norm_results[plot_metrics] = norm_results[plot_metrics].fillna(0)
            normalization_name = 'max-fillna'
        else:
            normalization_name = 'max'
        norm_results[plot_metrics] = norm_results.groupby(group_cols)[plot_metrics].apply(
            lambda x: x / x.max())  # applies function to each column independently

        print(f'\nWhat is the mean feature-set quality ({normalization_name}-normalized per',
              'experimental setting) for different numbers of alternatives and feature-selection',
              'methods (for sequential search with k=10)?')
        for metric in plot_metrics:
            print(norm_results.groupby(['n_alternative', 'fs_name'])[metric].mean().reset_index(
                ).pivot(index='n_alternative', columns='fs_name').round(2))

        print(f'\nWhat is the mean feature-set quality ({normalization_name}-normalized per',
              'experimental setting) for different dissimilarity thresholds "tau" and',
              'feature-selection methods (for sequential search with k=10)?')
        for metric in plot_metrics:
            print(norm_results.groupby(['tau_abs', 'fs_name'])[metric].mean().reset_index(
                ).pivot(index='tau_abs', columns='fs_name').round(2))

        print(f'\nHow does the feature-set quality ({normalization_name}-normalized per',
              'experimental setting) (Spearman-)correlate with dataset dimensionality "n" for ',
              'each alternative and dissimilarity treshold "tau" (for sequential search with k=10',
              'and Model Gain as feature-selection method)?')
        for metric in plot_metrics:
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore',
                                        category=scipy.stats.SpearmanRConstantInputWarning)
                print('Metric:', metric)
                print(norm_results[norm_results['fs_name'] == 'Model Gain'].groupby(
                    ['n_alternative', 'tau_abs']).apply(lambda x: x[metric].corr(
                        x['n'], method='spearman')).rename('').reset_index().pivot(
                            index='tau_abs', columns='n_alternative').round(2))

            # Figures 6.5a-6.5f (Model Gain) and 6.7a-6.7d (other feature-selection methods):
            # Feature-set quality by number of alternatives and dissimilarity threshold "tau"
            if fillna or metric != 'train_objective':
                fs_names = ['Model Gain']
            else:
                fs_names = fs_name_plot_order  # all feature-selection methods
            for fs_name in fs_names:
                plot_results = norm_results[norm_results['fs_name'] == fs_name].groupby(
                    ['n_alternative', 'tau_abs'])[metric].mean().reset_index()
                plot_results['tau'] = plot_results['tau_abs'] / 10
                plt.figure(figsize=(4, 3))
                plt.rcParams['font.size'] = 15
                sns.lineplot(x='n_alternative', y=metric, hue='tau', data=plot_results,
                             palette=DEFAULT_COL_PALETTE, hue_norm=(-0.2, 1), legend=False)
                # Use color scale instead of standard line plot legend; start color scaling at
                # -0.2, so the color for the actual lowest value (tau=0) is more readable (darker):
                cbar = plt.colorbar(ax=plt.gca(), mappable=plt.cm.ScalarMappable(
                    cmap=DEFAULT_COL_PALETTE, norm=plt.Normalize(-0.2, 1)),
                    values=plot_results['tau'].unique())
                cbar.ax.invert_yaxis()  # put low values at top (like most lines are ordered)
                cbar.ax.set_title('$\\tau}$', y=0, pad=-20, loc='left')
                cbar.ax.set_yticks(np.arange(start=0.2, stop=1.1, step=0.2))
                plt.xlabel('Number of alternative')
                plt.ylabel(f'Normalized {metric_name_mapping[metric]}')
                plt.xticks(range(0, 11, 1))
                plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
                plt.ylim(-0.05, 1.05)
                plt.tight_layout()
                file_name = 'afs-impact-num-alternatives-tau'
                file_name = file_name + f'-{metric.replace("_", "-")}-{normalization_name}'
                if fs_name != 'Model Gain':
                    file_name = file_name + '-' + fs_name.lower().replace(' ', '-')
                file_name = file_name + '.pdf'
                plt.savefig(plot_dir / file_name)

    print('\nHow do the evaluation metrics (Spearman-)correlate for different feature-selection',
          'methods (for all experimental settings; averaged over datasets and cross-validation',
          'folds)?')
    print_metrics = ['train_objective', 'test_objective', 'decision_tree_train_mcc',
                     'decision_tree_test_mcc']
    for fs_name in results['fs_name'].unique():
        print('Feature-selection method:', fs_name)
        print_results = results[results['fs_name'] == fs_name]
        print_results = print_results.groupby(['dataset_name', 'split_idx'])[print_metrics].corr(
            method='spearman').reset_index().rename(columns={'level_2': 'Metric'})
        print_results = print_results.groupby('Metric', sort=False)[print_metrics].mean(
            ).round(2).reset_index().set_index('Metric')
        print_results = print_results.rename(columns=(lambda x: x.replace('decision_', '')),
                                             index=(lambda x: x.replace('decision_', '')))
        print(print_results)

    print('\n-- Optimization status --')

    for k in results['k'].unique():
        plot_results = results[(results['fs_name'] == 'Model Gain') & (results['k'] == k) &
                               (results['search_name'] == 'seq.')]

        # Figures 6.6a, 6.6b: Optimization status by number of alternatives and dissimilarity
        # threshold "tau"
        assert plot_results['optimization_status'].isin(['Infeasible', 'Optimal']).all()
        plot_results = plot_results.groupby(['tau_abs', 'n_alternative'])[
            'optimization_status'].agg(lambda x: (x == 'Optimal').sum() / len(x)).reset_index()
        plot_results['tau'] = plot_results['tau_abs'] / k
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.lineplot(x='n_alternative', y='optimization_status', hue='tau', data=plot_results,
                     palette=DEFAULT_COL_PALETTE, hue_norm=(-0.2, 1), legend=False)
        cbar = plt.colorbar(ax=plt.gca(), mappable=plt.cm.ScalarMappable(
            cmap=DEFAULT_COL_PALETTE, norm=plt.Normalize(-0.2, 1)),
            values=plot_results['tau'].unique())
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


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates the dissertations\'s plots and print statistics.',
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
