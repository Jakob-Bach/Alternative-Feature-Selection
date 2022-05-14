# Finding Optimal Solutions for Alternative Feature Selection

This repository contains the code of the paper

> Bach, Jakob. "Finding Optimal Solutions for Alternative Feature Selection"

(The paper is not published yet.
Once it's published, we'll add a link to it here.
We'll link the experimental data, too.)

This document describes the repo structure, a short demo, and the steps to reproduce the experiments.

## Repo Structure

At the moment, the repository contains six Python files and four non-code files.
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
  It contains solver configuration, the dissimilarity-based definition of alternatives,
  and the two search procedures, i.e., sequential as well as simultaneous.
  To integrate a particular feature selector, you need to create a subclass.
  The subclass needs to define the optimization problem of the feature selector
  (the objective function and maybe constraints) in `initialize_solver()` and
  the process of solving the optimization problem in `select_and_evaluate()`.
  The search procedures for alternatives implemented in `AlternativeFeatureSelector` basically add
  further constraints (for alternatives) to the optimization problem and call the solving routine.
  We did this subclassing for the four feature selectors in our experiments, i.e.,
  mutual information (univariate filter), FCBF, model-based importance, and greedy wrapper.
- `prediction.py`: Functions to make predictions for the experimental pipeline
  and two of our feature selectors that use prediction models (model-based and wrapper).

## Demo

Running alternative feature selection only requires three steps:

1) Create the feature selector (our code contains four different ones).
2) Set the dataset:
    - feature-part and prediction target separated, train-test split
    - data types: `DataFrame` and `Series` from `pandas`
3) Run the search for alternatives:
    - Method name determines whether sequential or simultaneous search is run.
    - `k` determines the number of features to be selected.
    - `num_alternatives` determines ... you can guess what.
    - `tau_abs` determines by how many features the feature sets should differ.
      You can also provide a relative value (from the interval `[0,1]`) via `tau`,
      and change the dissimilarity `d_name` to `'jaccard'` (default is `'dice'`).

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

The search result is a `DataFrame`:

```
  selected_idxs  train_objective  test_objective  optimization_status
0     [0, 2, 3]             0.91            0.89                    0
1     [1, 2, 3]             0.83            0.78                    0
2     [0, 1, 3]             0.64            0.65                    0
3     [0, 1, 2]             0.62            0.68                    0
4            []              NaN             NaN                    2
5            []              NaN             NaN                    2
```

The search procedure ran out of feature here, as the `iris` dataset only has four features.

## Setup

Before running the scripts to reproduce the experiments, you should

1) Set up an environment (optional, but recommended).
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

Depending on your hardware, this might take several hours or even days
(we had a runtime of several hours on a server with 32 CPU cores).
In case the pipeline is nearly finished, but doesn't make progress any more,
the solver might have silently crashed.
You can re-run the pipeline, in which case it detects existing results and focuses on missing ones.
To print statistics and create the plots for the paper, run

```bash
python -m run_evaluation
```

All scripts have a few command-line options, which you can see by running the scripts like

```bash
python -m prepare_datasets --help
```
