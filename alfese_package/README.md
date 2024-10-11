# `alfese` -- A Python Package for Alternative Feature Selection

The package `alfese` contains several methods for alternative feature selection.
Alternative feature selection is the problem of finding multiple feature sets (sequentially or simultaneously)
that optimize feature-set quality while being sufficiently dissimilar to each other.
Users can control the number of alternatives and a dissimilarity threshold.
The package can also be used for traditional feature selection (i.e., sequential search with zero alternatives)
but may not be the most efficient solution for this purpose.

This document provides:

- Steps for [setting up](#setup) the package.
- A short [overview](#functionality) of the (feature-selection) functionality.
- A [demo](#demo) of the functionality.
- [Guidelines for developers](#developer-info) who want to modify or extend the code base.

If you use this package for a scientific publication, please cite [our paper](https://doi.org/10.1007/s41060-024-00527-8)

```
@article{bach2024alternative,
  title={Alternative feature selection with user control},
  author={Bach, Jakob and B{\"o}hm, Klemens},
  journal={International Journal of Data Science and Analytics},
  year={2024},
  doi={10.1007/s41060-024-00527-8}
}
```

(is partially outdated regarding our current implementation, e.g., does not describe the heuristic search methods)
or [our other paper](https://doi.org/10.48550/arXiv.2307.11607)

```
@misc{bach2023finding,
	title={Finding Optimal Diverse Feature Sets with Alternative Feature Selection},
	author={Bach, Jakob},
	howpublished={arXiv:2307.11607 [cs.LG]},
	year={2023},
	doi={10.48550/arXiv.2307.11607}
}
```

(at least the most recent version is more up-to-date than the journal version).

## Setup

You can install our package from [PyPI](https://pypi.org/):

```
python -m pip install alfese
```

Alternatively, you can install the package from GitHub:

```bash
python -m pip install git+https://github.com/Jakob-Bach/Alternative-Feature-Selection.git#subdirectory=alfese_package
```

If you already have the source code for the package (i.e., the directory in which this `README` resides)
as a local directory on your computer (e.g., after cloning the project), you can also perform a local install:

```bash
python -m pip install .
```

## Functionality

`alfese.py` contains six feature-selection methods as classes:

- `FCBFSelector`: (adapted version of) FCBF, a multivariate filter method
- `GreedyWrapperSelector`: a wrapper method (by default, using a decision tree as prediction model)
- `ManualUnivariateQualitySelector`: a univariate filter method where you can enter each feature's utility directly
  (instead of computing it from a dataset)
- `MISelector`: a univariate filter method based on mutual information
- `ModelImportanceSelector`: a univariate filter method using feature importances from a prediction model
  (by default, a decision tree)
- `MRMRSelector`: mRMR, a multivariate filter method

The feature-selection method determines the notion of feature-set quality, i.e., the optimization objective.

Additionally, there are the following abstract superclasses:

- `AlternativeFeatureSelector`: highest superclass; defines solver, constraints for alternatives,
  and solver-based sequential/simultaneous search
- `LinearQualityFeatureSelector`:  super-class for feature-selection methods with a linear objective;
  defines heuristic search methods for alternatives (do not require a solver)
- `WhiteBoxFeatureSelector`: superclass for feature-selection methods with a white-box objective,
  i.e., optimizing purely with a solver rather than using the solver in an algorithmic search routine

All feature-selection methods support sequential and simultaneous search for alternatives,
as demonstrated next.

## Demo

Running alternative feature selection only requires three steps:

1) Create the feature selector (our code contains five different ones),
  thereby determining the notion of feature-set quality to be optimized.
2) Set the dataset (`set_data()`):
    - Four parameters: feature part and prediction target are separated, train-test split
    - Data types: `DataFrame` (feature parts) and `Series` (targets) from `pandas`
3) Run the search for alternatives:
    - Method name (`search_sequentially()` / `search_simultaneously()`) determines whether
      a (solver-based) sequential or a simultaneous search is run. `LinearQualityFeatureSelector`s
      (like "MI" and model-based importance) also support the heuristic procedures
      `search_greedy_replacement()` and `search_greedy_balancing()`.
    - `k` determines the number of features to be selected.
    - `num_alternatives` determines ... you can guess what.
    - `tau_abs` determines by how many features the feature sets should differ from each other.
      You can also provide a relative value (from the interval `[0,1]`) via `tau`,
      or change the dissimilarity `d_name` to `'jaccard'` (default is `'dice'`).
    - `objective_agg` switches between min-aggregation and sum-aggregation in solver-based simultaneous search.
      Has no effect in sequential search (which only returns one feature set, so there is no need to
      aggregate feature-set quality over feature sets) and the heuristic search methods.

```python
import alfese
import sklearn.datasets
import sklearn.model_selection

dataset = sklearn.datasets.load_iris(as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    dataset['data'], dataset['target'], train_size=0.8, random_state=25)
feature_selector = alfese.MISelector()
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

If you don't want to provide a dataset but use manually defined univariate qualities
(which result in the same optimization problem as "MI" and model importance), you can do so as well:

```python
import alfese

feature_selector = alfese.ManualQualityUnivariateSelector()
feature_selector.set_data(q_train=[1, 2, 3, 7, 8, 9])
search_result = feature_selector.search_sequentially(k=3, num_alternatives=3, tau_abs=2)
print(search_result.drop(columns='optimization_time').round(2))
```

## Developer Info

`AlternativeFeatureSelector` is the topmost abstract superclass.
It contains code for solver handling, the dissimilarity-based definition of alternatives, and the
two solver-based search procedures, i.e., sequential as well as simultaneous (sum-aggregation and min-aggregation).
For defining a new feature-selection method, you should create a subclass of `AlternativeFeatureSelector`.
In particular, you need to define how to solve the optimization problem of alternative feature selection
by overriding the abstract method `select_and_evaluate()`.
To this end, you may want to define the optimization problem
(objective function, which expresses feature-set quality, and maybe further constraints)
by overriding `initialize_solver()`.
You should also call the original implementation of this methods via `super().initialize_solver()`
to not override general initialization steps (solver configuration, cardinality constraints).
The sequential and simultaneous search procedures for alternatives implemented in `AlternativeFeatureSelector`
basically add further constraints (for alternatives) to the optimization problem and call `select_and_evaluate()`.
Thus, if the latter method is implemented properly, you do not need to override the search procedures,
as they should work as-is in new subclasses as well.

There are further abstract superclasses extracting commonalities between feature-selection methods:

-  `WhiteBoxFeatureSelector` is a good starting point if you want to optimize your objective with a solver
  (rather than using the solver in an algorithmic search routine with a black-box objective, like Greedy Wrapper does).
  When creating a subclass, you need to define the white-box objective by overriding the abstract method `create_objectives()`
  (define objectives separately for training set and test set, as they may use different constants for feature qualities).
  `select_and_evaluate()` and `initialize_solver()` need not be overridden in your subclass anymore.
- `LinearQualityFeatureSelector` is a good starting point if your objective is a plain sum of feature qualities.
  When creating a subclass, you need to provide these qualities by overriding the abstract method `compute_qualities()`.
  `select_and_evaluate()` and `initialize_solver()` need not be overridden in your subclass anymore.
