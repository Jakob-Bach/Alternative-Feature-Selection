"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
"""


import argparse
import ast
import pathlib

import matplotlib.pyplot as plt
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
    results['fs_name'] = results['fs_name'].str.removesuffix('Selector')
    results['search_name'] = results['search_name'].str.replace('(search_|ly)', '')
    results['optimization_status'].replace({0: 'Optimal', 1: 'Feasible', 2: 'Infeasible',
                                            6: 'Not solved'}, inplace=True)
    # Define columns for main experimental dimensions (corresponding to independent search runs):
    group_cols = ['dataset_name', 'split_idx', 'fs_name', 'search_name', 'k', 'tau_abs',
                  'num_alternatives']
    # Define columns for evaluation metrics:
    quality_metrics = [x for x in results.columns if 'train' in x or 'test' in x]
    prediction_metrics = [x for x in results.columns if '_mcc' in x]

    print('\n------ Experimental Design ------')

    print('\n---- Methods ----')

    print('\n-- Alternatives (Constraints) --')

    print('\nHow often do certain optimization statuses occur?')
    print(results['optimization_status'].value_counts(normalize=True).apply('{:.2%}'.format))

    print('\n---- Datasets ----')

    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    dataset_overview = dataset_overview[['dataset', 'n_instances', 'n_features']]
    dataset_overview.rename(columns={'dataset': 'Dataset', 'n_instances': 'm',
                                     'n_features': 'n'}, inplace=True)
    dataset_overview['Dataset'] = dataset_overview['Dataset'].str.replace('GAMETES', 'G')
    dataset_overview.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    with pd.option_context('max_colwidth', 1000):  # avoid truncation
        print(dataset_overview.to_latex(index=False))

    print('\n------ Evaluation ------')

    print('\n---- Datasets ----')

    print('\nHow does median feature-set quality differ between datasets?')
    print(results.groupby('dataset_name')[['train_objective', 'decision_tree_test_mcc']].median(
        ).describe().round(2))

    print('\nHow does median feature-set quality differ between "n" (dataset dimensionality)?')
    print(results.groupby('n')[['train_objective', 'decision_tree_test_mcc']].median().round(2))

    print('\nHow does feature set-quality (Spearman-)correlate with "n"?')
    print(results[quality_metrics].corrwith(results['n'], method='spearman').round(2))

    print('\nHow does feature set-quality (Spearman-)correlate with "k"/"n"?')
    print(results[quality_metrics].corrwith(results['k'] / results['n'], method='spearman').round(2))

    print('\n---- Prediction Models ----')

    print('\nHow is prediction performance distributed for different models?')
    print(results[prediction_metrics].describe().round(2).transpose())

    print('\nWhat\'s the median overfitting (train-test difference)?')
    results['objective_dif'] = results['train_objective'] - results['test_objective']
    results['tree_mcc_dif'] = results['decision_tree_train_mcc'] - results['decision_tree_test_mcc']
    print(results.groupby('fs_name')[['objective_dif', 'tree_mcc_dif']].median().round(2))
    results.drop(columns=['objective_dif', 'tree_mcc_dif'], inplace=True)

    plot_results = results[quality_metrics].corr(method='spearman').round(2)
    name_mapping = {'train_objective': '$Q_{\\mathrm{train}}$',
                    'test_objective': '$Q_{\\mathrm{test}}$',
                    'decision_tree_train_mcc': '$MCC_{\\mathrm{train}}^{\\mathrm{tree}}$',
                    'decision_tree_test_mcc': '$MCC_{\\mathrm{test}}^{\\mathrm{tree}}$',
                    'random_forest_train_mcc': '$MCC_{\\mathrm{train}}^{\\mathrm{forest}}$',
                    'random_forest_test_mcc': '$MCC_{\\mathrm{test}}^{\\mathrm{forest}}$'}
    plot_results.rename(columns=name_mapping, index=name_mapping, inplace=True)
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 16
    sns.heatmap(plot_results, vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'evaluation-metrics-correlation.pdf')

    print('\n---- Feature-Selection Methods ----')

    print('\nHow does the optimization status differ between feature-selection methods?')
    print(pd.crosstab(results['optimization_status'], results['fs_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\nHow does prediction performance differ between feature-selection methods?')
    for metric in prediction_metrics:
        print('\nMetric:', metric)
        print(results.groupby('fs_name')[metric].describe().round(2))

    print('\nHow do the results differ between k on median?')
    print(results.groupby('k')[['train_objective', 'decision_tree_test_mcc']].median().round(2))
    print(results.groupby(['fs_name', 'k'])[['train_objective', 'decision_tree_test_mcc']].median(
        ).round(2))

    print('\n---- Searching Alternatives ----')

    print('\n-- Search Method --')

    comparison_results = results[results['search_name'] == 'simultaneous']
    for num_alternatives in results.loc[results['search_name'] == 'simultaneous',
                                        'num_alternatives'].unique():
        # Extract first "num_alternatives + 1" feature sets (sequential search is only run for one
        # value of "num_alternatives", but you can get "smaller" results by subsetting)
        seq_results = results[results['search_name'] == 'sequential'].groupby(group_cols).nth(
            range(0, num_alternatives + 1)).reset_index()
        seq_results['num_alternatives'] = num_alternatives
        comparison_results = pd.concat([comparison_results, seq_results])

    print('\nHow does the standard deviation of feature-set quality within results from one search',
          'depend on search methods and number of alternatives (on median)?')
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        print(comparison_results.groupby(group_cols)[metric].std().reset_index().groupby(
                ['search_name', 'num_alternatives'])[metric].median().reset_index().pivot(
                    index='num_alternatives', columns='search_name').round(3))

    plot_results = comparison_results.groupby(group_cols)['train_objective'].std().reset_index()
    plot_results['search_name'] = plot_results['search_name'].str.replace('(uential|ultaneous)', '.')
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='num_alternatives', y='train_objective', hue='search_name', data=plot_results,
                palette='Set2', fliersize=0)
    plt.xlabel('Number of alternatives')
    plt.ylabel('$\\sigma$ of $Q_{\\mathrm{train}}$')
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.ylim(-0.05, 0.45)
    leg = plt.legend(title='Search', edgecolor='white', loc='upper left', bbox_to_anchor=(0, -0.1),
                     columnspacing=1, framealpha=0, ncol=2)
    leg.get_title().set_position((-117, -21))
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-search-stddev-objective.pdf')

    print('\nHow does the average of feature-set quality within results from one search',
          'depend on search methods and number of alternatives (on median)?')
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        print(comparison_results.groupby(group_cols)[metric].mean().reset_index().groupby(
                ['search_name', 'num_alternatives'])[metric].mean().reset_index().pivot(
                    index='num_alternatives', columns='search_name').round(2))

    plot_results = comparison_results.groupby(group_cols)['train_objective'].mean().reset_index()
    plot_results['search_name'] = plot_results['search_name'].str.replace('(uential|ultaneous)', '.')
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='num_alternatives', y='train_objective', hue='search_name', data=plot_results,
                palette='Set2', fliersize=0)
    plt.xlabel('Number of alternatives')
    plt.ylabel('Mean of $Q_{\\mathrm{train}}$')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    leg = plt.legend(title='Search', edgecolor='white', loc='upper left', bbox_to_anchor=(0, -0.1),
                     columnspacing=1, framealpha=0, ncol=2)
    leg.get_title().set_position((-117, -21))
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-search-mean-objective.pdf')

    print('\nHow is the difference in average feature-set quality between simultaneous search and',
          'sequential search (each comparison for the same search setting) distributed?')
    for metric in ['train_objective', 'test_objective', 'decision_tree_test_mcc']:
        reshaped_comparison_results = comparison_results.groupby(group_cols)[metric].mean(
            ).reset_index().pivot(index=[x for x in group_cols if x != 'search_name'],
                                  columns='search_name', values=metric)
        print('\nMetric:', metric)
        print((reshaped_comparison_results['simultaneous'] -
               reshaped_comparison_results['sequential']).describe().round(2))

    print('\nHow does the optimization status differ between search methods (excluding greedy FS,',
          'where the optimization status only describes the last solver run)?')
    print(pd.crosstab(results.loc[results['fs_name'] != 'GreedyWrapper', 'optimization_status'],
                      results.loc[results['fs_name'] != 'GreedyWrapper', 'search_name']))
    print(pd.crosstab(results.loc[results['fs_name'] != 'GreedyWrapper', 'optimization_status'],
                      results.loc[results['fs_name'] != 'GreedyWrapper', 'search_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\nHow does the optimization status depend on the number of alternatives in simultaneous',
          'search (excluding greedy FS)?')
    print(pd.crosstab(results.loc[(results['search_name'] == 'simultaneous') &
                                  (results['fs_name'] != 'GreedyWrapper'), 'optimization_status'],
                      results.loc[(results['search_name'] == 'simultaneous') &
                                  (results['fs_name'] != 'GreedyWrapper'), 'num_alternatives'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\n-- Number of Alternatives --')

    seq_results = results[(results['search_name'] == 'sequential')].copy()
    seq_results['n_alternative'] = seq_results.groupby(group_cols).cumcount()
    normalization_funcs = {'max': lambda x: x / x.max(),
                           'min-max': lambda x: (x - x.min()) / (x.max() - x.min())}
    norm_quality_metrics = ['norm_' + x for x in quality_metrics]
    for normalization_name, normalization_func in normalization_funcs.items():
        seq_results[norm_quality_metrics] = seq_results.groupby(group_cols)[quality_metrics].apply(
            normalization_func)  # applies function to each column independently

        print(f'\nHow does the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) develop over the iterations of sequential search?')
        for metric in ['norm_train_objective', 'norm_test_objective', 'norm_decision_tree_test_mcc']:
            print('\nMetric:', metric)
            print(seq_results.groupby(['n_alternative', 'fs_name'])[metric].median().reset_index(
                ).pivot(index='n_alternative', columns='fs_name').round(2))

        plot_results = seq_results[(seq_results['fs_name'] == 'MI')].melt(
            id_vars=['n_alternative'], value_vars=['norm_train_objective', 'norm_test_objective'],
            var_name='split', value_name='objective')
        plot_results['split'] = plot_results['split'].str.replace('(norm_|_objective)', '')
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.boxplot(x='n_alternative', y='objective', hue='split', data=plot_results,
                    palette='Set2', fliersize=0)
        plt.xlabel('Number of alternative')
        plt.ylabel('Normalized $Q_{\\mathrm{train}}$')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        leg = plt.legend(title='Split', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.1), columnspacing=1, framealpha=0, ncol=2)
        leg.get_title().set_position((-107, -21))
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-num-alternatives-objective-{normalization_name}.pdf')

    print('\nHow does the optimization status differ between the number of alternatives in',
          'sequential search for MI feature selection?')
    print(pd.crosstab(seq_results.loc[seq_results['fs_name'] == 'MI', 'n_alternative'],
                      seq_results.loc[seq_results['fs_name'] == 'MI', 'optimization_status'],
                      normalize='index').applymap('{:.2%}'.format))

    sim_results = results[(results['search_name'] == 'simultaneous')].copy()
    sim_results[quality_metrics] = sim_results.groupby(group_cols)[quality_metrics].apply(
        normalization_funcs['max'])

    print('\nHow does the median train objective (max-normalized per experimental setting) develop',
          'over the number of alternatives in simultaneous search?')
    print(sim_results.groupby(['num_alternatives', 'fs_name'])['train_objective'].median(
        ).reset_index().pivot(index='num_alternatives', columns='fs_name').round(2))

    print('\nHow does the optimization status differ between the number of alternatives in',
          'simultaneous search for MI feature selection?')
    print(pd.crosstab(sim_results.loc[sim_results['fs_name'] == 'MI', 'num_alternatives'],
                      sim_results.loc[sim_results['fs_name'] == 'MI', 'optimization_status'],
                      normalize='index').applymap('{:.2%}'.format))

    print('\n -- Dissimilarity Threshold --')

    seq_results[norm_quality_metrics] = seq_results.groupby(group_cols)[quality_metrics].apply(
        normalization_funcs['max'])
    for split in ['train', 'test']:
        plot_results = seq_results[(seq_results['fs_name'] == 'MI') & (seq_results['k'] == 10)].groupby(
            ['n_alternative', 'tau_abs'])[f'norm_{split}_objective'].median().reset_index()
        plot_results['tau'] = plot_results['tau_abs'] / 10
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 15
        sns.lineplot(x='n_alternative', y=f'norm_{split}_objective', hue='tau',
                     data=plot_results, palette='RdPu', hue_norm=(-0.2, 1), legend=False)
        # Use color scale instead of standard line plot legend; start color scaling at -0.2 such
        # that the color for the actual lowest value (tau=0) is more readable (darker):
        cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap="RdPu", norm=plt.Normalize(-0.2, 1)), values=plot_results['tau'].unique())
        cbar.ax.invert_yaxis()  # put low values at top (like most lines are ordered in the plot)
        cbar.ax.set_title('$\\tau}$', y=0, pad=-20, loc='left')
        cbar.ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel('Number of alternative')
        plt.ylabel('Normalized $Q_{\\mathrm{' + split + '}}$')
        plt.xticks(range(0, 11, 1))
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-num-alternatives-{split}-objective-tau.pdf')

    print('\nHow does the median feature-set quality (max-normalized per experimental setting)',
          'develop over the iterations and "tau" of sequential search for MI and k=10?')
    for metric in ['norm_train_objective', 'norm_test_objective', 'norm_decision_tree_test_mcc']:
        print(seq_results[(seq_results['fs_name'] == 'MI') & (seq_results['k'] == 10)].groupby(
            ['n_alternative', 'tau_abs'])[metric].median().reset_index(
            ).pivot(index='n_alternative', columns='tau_abs').round(2))

    print('\nHow does the optimization status differ between the dissimilarity thresholds in',
          'sequential search for MI feature selection?')
    print(pd.crosstab(seq_results.loc[(seq_results['fs_name'] == 'MI') & (seq_results['k'] == 10),
                                      'tau_abs'],
                      seq_results.loc[(seq_results['fs_name'] == 'MI') & (seq_results['k'] == 10),
                                      'optimization_status'],
                      normalize='index').applymap('{:.2%}'.format))

    print('\nHow does the median feature-set quality (max-normalized per experimental setting)',
          'develop over "tau" for different feature selectors in sequential search (k=10)?')
    for metric in ['norm_train_objective', 'norm_test_objective', 'norm_decision_tree_test_mcc']:
        print(seq_results[seq_results['k'] == 10].groupby(['tau_abs', 'fs_name'])[metric].median(
            ).reset_index().pivot(index='tau_abs', columns='fs_name').round(2))

    print('\nHow does the median train objective (max-normalized per experimental setting) develop',
          'over "tau" for different feature selectors in simultaneous search (k=10)?')
    print(sim_results[sim_results['k'] == 10].groupby(
        ['tau_abs', 'fs_name'])['train_objective'].median().reset_index().pivot(
            index='tau_abs', columns='fs_name').round(2))

    print('\nHow many new features occur in a feature set from one alternative to next in',
          'sequential search for MI as feature selector?')
    feature_diff_results = seq_results.copy()
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
