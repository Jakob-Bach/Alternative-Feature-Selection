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


plt.rcParams['font.family'] = 'Helvetica'  # IEEE template's sans-serif font


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
    print(results['optimization_status'].value_counts(normalize=True).apply('{:.2%}'.format))

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
                      normalize='columns').applymap('{:.2%}'.format))

    print('\nHow does prediction performance differ between feature-selection methods?')
    for metric in prediction_metrics:
        print('\nMetric:', metric)
        print(results.groupby('fs_name')[metric].describe().round(2))

    print('\nHow do the results differ between k on median?')
    print(results.groupby('k')[['train_objective', 'Decision tree_test_mcc']].median().round(2))
    print(results.groupby(['fs_name', 'k'])[['train_objective', 'Decision tree_test_mcc']].median(
        ).round(2))

    print('\n---- Searching Alternatives ----')

    print('\n-- Search Method --')

    comparison_results = results[results['search_name'] == 'search_simultaneously']
    for num_alternatives in results.loc[results['search_name'] == 'search_simultaneously',
                                        'num_alternatives'].unique():
        # Extract first "num_alternatives + 1" feature sets (sequential search is only run for one
        # value of "num_alternatives", but you can get "smaller" results by subsetting)
        seq_results = results[results['search_name'] == 'search_sequentially'].groupby(
            grouping).nth(range(0, num_alternatives + 1)).reset_index()
        seq_results['num_alternatives'] = num_alternatives
        comparison_results = pd.concat([comparison_results, seq_results])

    print('\nHow does the standard deviation of feature-set quality within results from one search',
          'depend on search methods and number of alternatives (on median)?')
    for metric in ['train_objective', 'test_objective', 'Decision tree_test_mcc']:
        print(comparison_results.groupby(grouping)[metric].std().reset_index().groupby(
                ['search_name', 'num_alternatives'])[metric].median().reset_index().pivot(
                    index='num_alternatives', columns='search_name').round(3))

    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    plot_results = comparison_results.groupby(grouping)['train_objective'].std().reset_index()
    plot_results['search_name'] = plot_results['search_name'].str.replace('search_', '')
    sns.boxplot(x='num_alternatives', y='train_objective', hue='search_name', data=plot_results,
                palette='Set2', fliersize=0)
    plt.xlabel('Number of alternatives')
    plt.ylabel('Std. dev. of $Q_{train}$')
    plt.ylim(-0.05, 0.55)
    leg = plt.legend(title='Search', edgecolor='white', loc='upper left',
                     bbox_to_anchor=(0, -0.1), framealpha=0, ncol=1)
    leg.get_title().set_position((-80, -24))
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-search-stddev-objective.pdf')

    print('\nHow does the average of feature-set quality within results from one search',
          'depend on search methods and number of alternatives (on median)?')
    for metric in ['train_objective', 'test_objective', 'Decision tree_test_mcc']:
        print(comparison_results.groupby(grouping)[metric].mean().reset_index().groupby(
                ['search_name', 'num_alternatives'])[metric].mean().reset_index().pivot(
                    index='num_alternatives', columns='search_name').round(2))

    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    plot_results = comparison_results.groupby(grouping)['train_objective'].mean().reset_index()
    plot_results['search_name'] = plot_results['search_name'].str.replace('search_', '')
    sns.boxplot(x='num_alternatives', y='train_objective', hue='search_name', data=plot_results,
                palette='Set2', fliersize=0)
    plt.xlabel('Number of alternatives')
    plt.ylabel('Mean of $Q_{train}$')
    leg = plt.legend(title='Search', edgecolor='white', loc='upper left',
                     bbox_to_anchor=(0, -0.1), framealpha=0, ncol=1)
    leg.get_title().set_position((-80, -24))
    plt.tight_layout()
    plt.savefig(plot_dir / 'impact-search-mean-objective.pdf')

    print('\nHow is the difference in feature-set quality between simultaneous search and',
          'sequential search (each comparison for the same search setting) distributed?')
    for metric in ['train_objective', 'test_objective', 'Decision tree_test_mcc']:
        reshaped_comparison_results = comparison_results.groupby(grouping)[metric].median(
            ).reset_index().pivot(index=[x for x in grouping if x != 'search_name'],
                                  columns='search_name', values=metric)
        print('\nMetric:', metric)
        print((reshaped_comparison_results['search_simultaneously'] -
               reshaped_comparison_results['search_sequentially']).describe().round(2))

    print('\nHow does the optimization status differ between search methods (excluding greedy FS)?')
    print(pd.crosstab(results.loc[results['fs_name'] != 'GreedyWrapper', 'optimization_status'],
                      results.loc[results['fs_name'] != 'GreedyWrapper', 'search_name']))
    print(pd.crosstab(results.loc[results['fs_name'] != 'GreedyWrapper', 'optimization_status'],
                      results.loc[results['fs_name'] != 'GreedyWrapper', 'search_name'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\nHow does the optimization status depend on the number of alternatives in simultaneous',
          'search (excluding greedy FS)?')
    print(pd.crosstab(results.loc[(results['search_name'] == 'search_simultaneously') &
                                  (results['fs_name'] != 'GreedyWrapper'), 'optimization_status'],
                      results.loc[(results['search_name'] == 'search_simultaneously') &
                                  (results['fs_name'] != 'GreedyWrapper'), 'num_alternatives'],
                      normalize='columns').applymap('{:.2%}'.format))

    print('\n-- Number of Alternatives --')

    seq_results = results[(results['search_name'] == 'search_sequentially')].copy()
    seq_results['n_alternative'] = seq_results.groupby(grouping).cumcount()
    normalization_funcs = {'max': lambda x: x / x.max(),
                           'min-max': lambda x: (x - x.min()) / (x.max() - x.min())}
    norm_quality_metrics = ['norm_' + x for x in quality_metrics]
    for normalization_name, normalization_func in normalization_funcs.items():
        seq_results[norm_quality_metrics] = seq_results.groupby(grouping)[quality_metrics].apply(
            normalization_func)

        print(f'\nHow does the median feature-set quality ({normalization_name}-normalized per',
              'experimental setting) develop over the iterations of sequential search?')
        for metric in ['norm_train_objective', 'norm_test_objective', 'norm_Decision tree_test_mcc']:
            print('\nMetric:', metric)
            print(seq_results.groupby(['n_alternative', 'fs_name'])[metric].median().reset_index(
                ).pivot(index='n_alternative', columns='fs_name').round(2))

        plot_results = seq_results[(seq_results['fs_name'] == 'MI')].melt(
            id_vars=['n_alternative'], value_vars=['norm_train_objective', 'norm_test_objective'],
            var_name='split', value_name='objective')
        plot_results['split'] = plot_results['split'].str.replace('(norm_|_objective)', '')
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 11
        sns.boxplot(x='n_alternative', y='objective', hue='split', data=plot_results,
                    palette='Set2', fliersize=0)
        plt.xlabel('Number of alternative')
        plt.ylabel('Normalized objective')
        leg = plt.legend(title='Split', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.1), framealpha=0, ncol=2)
        leg.get_title().set_position((-90, -15))
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-num-alternatives-objective-{normalization_name}.pdf')

    print('\nHow does the optimization status differ between the number of alternatives in',
          'sequential search for MI feature selection?')
    print(pd.crosstab(seq_results.loc[seq_results['fs_name'] == 'MI', 'n_alternative'],
                      seq_results.loc[seq_results['fs_name'] == 'MI', 'optimization_status'],
                      normalize='index').applymap('{:.2%}'.format))

    sim_results = results[(results['search_name'] == 'search_simultaneously')].copy()
    sim_results[quality_metrics] = sim_results.groupby(grouping)[quality_metrics].apply(
        lambda x: x / x.max())

    print('\nHow does the median train objective (max-normalized per experimental setting) develop',
          'over the number of alternatives in simultaneous search?')
    print(sim_results.groupby(['num_alternatives', 'fs_name'])['train_objective'].median(
        ).reset_index().pivot(index='num_alternatives', columns='fs_name').round(2))

    print('\nHow does the optimization status differ between the number of alternatives in',
          'simultaneous search for MI feature selection?')
    print(pd.crosstab(sim_results.loc[sim_results['fs_name'] == 'MI', 'num_alternatives'],
                      sim_results.loc[sim_results['fs_name'] == 'MI', 'optimization_status'],
                      normalize='index').applymap('{:.2%}'.format))

    print('\n -- Dissimilarity Threshold "tau"--')

    seq_results[norm_quality_metrics] = seq_results.groupby(grouping)[quality_metrics].apply(
        normalization_funcs['max'])
    for split in ['train', 'test']:
        plot_results = seq_results[(seq_results['fs_name'] == 'MI') & (seq_results['k'] == 10)].groupby(
            ['n_alternative', 'tau_abs'])[f'norm_{split}_objective'].median().reset_index()
        plot_results['tau'] = plot_results['tau_abs'] / 10
        plt.figure(figsize=(4, 3))
        plt.rcParams['font.size'] = 11
        sns.lineplot(x='n_alternative', y=f'norm_{split}_objective', hue='tau',
                     data=plot_results, palette='RdPu', hue_norm=(-0.2, 1), legend=False)
        cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap="RdPu", norm=plt.Normalize(-0.2, 1)), values=plot_results['tau'].unique())
        cbar.ax.invert_yaxis()  # put low values at top (like the lines are mainly ordered)
        cbar.ax.set_title('$\\tau}$', y=0, pad=-20, loc='left')
        plt.xlabel('Number of alternative')
        plt.ylabel('Normalized objective')
        plt.xticks(range(0, 11, 2))
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(plot_dir / f'impact-num-alternatives-{split}-objective-tau.pdf')

    print('\nHow does the median feature-set quality (max-normalized per experimental setting)',
          'develop over the iterations and "tau" of sequential search for MI and k=10?')
    for metric in ['norm_train_objective', 'norm_test_objective', 'norm_Decision tree_test_mcc']:
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
    for metric in ['norm_train_objective', 'norm_test_objective', 'norm_Decision tree_test_mcc']:
        print(seq_results[seq_results['k'] == 10].groupby(['tau_abs', 'fs_name'])[metric].median(
            ).reset_index().pivot(index='tau_abs', columns='fs_name').round(2))

    print('\nHow does the train objective (max-normalized per experimental setting) develop over',
          '"tau" in simultaneous search for MI feature selection?')
    print(sim_results[sim_results['k'] == 10].groupby(
        ['tau_abs', 'fs_name'])['train_objective'].mean().reset_index().pivot(
            index='tau_abs', columns='fs_name').round(2))

    print('How many new features occur in a feature set from one alternative to next in sequential',
          'search for MI as feature selector?')
    feature_diff_results = seq_results.copy()
    feature_diff_results['prev_selected_idxs'] = feature_diff_results.groupby(
        grouping)['selected_idxs'].shift().fillna('').apply(list)
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
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')
