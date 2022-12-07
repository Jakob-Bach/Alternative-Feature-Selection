# Alternative Feature Selection

This repository contains the code of two papers:

> Bach, Jakob. "Finding Optimal Solutions for Alternative Feature Selection"

(Currently under review at SDM 2023.
Use the tag `run-2022-05-13` for reproducing results since the code has evolved a bit afterwards.
In case the paper is accepted and published, we'll link it here.
We'll link the experimental data, too.)

> Bach, Jakob. "Finding Optimal Diverse Feature Sets via Alternative Feature Selection"

(To be submitted at arXiv.
Currently, there is no dedicated tag since the coding is in progress.
Once the paper is published, we'll link it here.
We'll link the experimental data, too.)

This document describes the repo structure, a short demo, and the steps to reproduce the experiments.

## Repo Structure and Developer Info

Currently, the repository contains six Python files and four non-code files.
The non-code files are:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

Four of the code files are directly related to our experiments (see below for details):

- `prepare_datasets.py`: First stage of the experiments
  (download prediction datasets).
- `run_experiments.py`: Second stage of the experiments
  (run feature selection, search for alternatives, and make predictions).
- `run_evaluation.py`: Third stage of the experiments
  (compute statistics and create plots for the paper).
- `data_handling.py`: Functions for working with prediction datasets and experimental data.

Two of the code files contain classes and functions for alternative feature selection.
If you want to use, modify, or extend alternative feature selection,
only these two files might be relevant for you:

- `afs.py`: Classes for alternative feature selection.
  `AlternativeFeatureSelector` is the abstract superclass.
  It contains code for solver handling, the dissimilarity-based definition of alternatives, and the
  two search procedures, i.e., sequential as well as simultaneous (sum-objective and min-objective).
  To integrate a new feature-selection method, you need to create a subclass.
  The subclass needs to define the optimization problem of the feature-selection method
  (the objective function and maybe constraints) in `initialize_solver()` and
  the process of solving the optimization problem in `select_and_evaluate()`.
  The search procedures for alternatives implemented in `AlternativeFeatureSelector` basically add
  further constraints (for alternatives) to the optimization problem and call the solving routine.
  We did this subclassing for the five feature-selection methods in our experiments, i.e.,
  mutual information (univariate filter), FCBF, model-based importance, mRMR, and greedy wrapper.
  There are further abstract superclasses extracting commonalities between feature selectors.
  In particular, `WhiteBoxFeatureSelector` is a good starting point if you want to optimize your
  objective with a solver (rather than only using the solver to check constraints while optimizing
  a black-box objective separately).
  `LinearQualityFeatureSelector` defines an objective that sums up the quality of individual
  features, so your subclass only has to define how to compute these qualities.
- `prediction.py`: Functions to make predictions for the experimental pipeline
  and two of our feature-selection methods that use prediction models (model-based and wrapper).

## Demo

Running alternative feature selection only requires three steps:

1) Create the feature selector (our code contains five different ones).
2) Set the dataset:
    - Four parameters: feature part and prediction target are separated, train-test split
    - Data types: `DataFrame` (feature parts) and `Series` (targets) from `pandas`
3) Run the search for alternatives:
    - Method name (`search_sequentially()` / `search_simultaneously()`) determines whether
      a sequential or a simultaneous search is run.
    - `k` determines the number of features to be selected.
    - `num_alternatives` determines ... you can guess what.
    - `tau_abs` determines by how many features the feature sets should differ.
      You can also provide a relative value (from the interval `[0,1]`) via `tau`,
      and change the dissimilarity `d_name` to `'jaccard'` (default is `'dice'`).
    - `objective_agg` switches between the min-objective and the sum-objective in a simultaneous
      search. Has no effect in sequential search (which only returns one feature set, so there is no
      need to aggregate feature-set quality over feature sets).

```python
import afs
import sklearn.datasets
import sklearn.model_selection

dataset = sklearn.datasets.load_iris(as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    dataset['data'], dataset['target'], train_size=0.8, random_state=25)
feature_selector = afs.MISelector()
feature_selector.set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
search_result = feature_selector.search_sequentially(k=3, num_alternatives=5, tau_abs=1)
print(search_result.drop(columns='optimization_time').round(2))
```

The search result is a `DataFrame` containing the indices of the selected features (can be used to
subset the columns in `X`), objective values on the training set and test set, optimization status,
and optimization time:

```
  selected_idxs  train_objective  test_objective  optimization_status
0     [0, 2, 3]             0.91            0.89                    0
1     [1, 2, 3]             0.83            0.78                    0
2     [0, 1, 3]             0.64            0.65                    0
3     [0, 1, 2]             0.62            0.68                    0
4            []              NaN             NaN                    2
5            []              NaN             NaN                    2
```

The search procedure ran out of features here, as the `iris` dataset only has four features.
The optimization statuses are:

- 0: `Optimal` (optimal solution found)
- 1: `Feasible` (a valid solution found till timeout, but may not be optimal)
- 2: `Infeasible` (there is no valid solution)
- 6: `Not solved` (no valid solution found till timeout, but there may be one)

## Setup

Before running the scripts to reproduce the experiments, you should

1) Set up an environment (optional but recommended).
2) Install all necessary dependencies.

Our code is implemented in Python (version 3.8; other versions, including lower ones, might work as well).

### Option 1: `conda` Environment

If you use `conda`, you can directly install the correct Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

Choose `<conda-env-name>` as you like.

To leave the environment, run

```bash
conda deactivate
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.7; other versions might work as well)
to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Let's assume the Python executable is located at `<path/to/python>`.
Next, you install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.7
```

To set up an environment with `virtualenv`, run

```bash
python -m virtualenv -p <path/to/python> <path/to/env/destination>
```

Choose `<path/to/env/destination>` as you like.

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

To leave the environment, run

```bash
deactivate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
Run

```bash
python -m prepare_datasets
```

to download and pre-process the input data for the experiments (prediction datasets from PMLB).

Next, start the experimental pipeline with

```bash
python -m run_experiments
```

Depending on your hardware, this might take several days.
We had a runtime of 141 hours on a server with an `AMD EPYC 7551` [CPU](https://www.amd.com/en/products/cpu/amd-epyc-7551)
(32 physical cores, base clock of 2.0 GHz).
In case the pipeline is nearly finished but doesn't make progress anymore,
the solver might have silently crashed (which happened in the past with `Cbc` as the solver, though
we didn't encounter the phenomenon with the current solver `SCIP`).
In this case, or if you had to abort the experimental run for other reasons, you could re-start the
experimental pipeline by calling the same script again; it automatically detects existing results
and only runs the remaining tasks.

To print statistics and create the plots for the paper, run

```bash
python -m run_evaluation
```

All scripts have a few command-line options, which you can see by running the scripts like

```bash
python -m prepare_datasets --help
```
