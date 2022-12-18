"""Alternative Feature Selection

Classes for alternative feature selection, i.e.,
- constraints and search routines for alternatives
- feature-selection methods (determining the objective of the search)
"""


from abc import ABCMeta, abstractmethod
import math
import time
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from ortools.linear_solver import pywraplp
import pandas as pd
import sklearn.base
import sklearn.feature_selection

import prediction


class AlternativeFeatureSelector(metaclass=ABCMeta):
    """Alternative Feature Selector

    The base class for alternative feature selection. Contains the search routines for alternatives
    (simultaneous, sequential) and various helper functions to formulate constraints. Class is
    abstract because an actual feature-selection method (used as subroutine in the search for
    alternatives) is not implemented here. Subclasses should at least override:
    1) "initialize_solver()" (add feature-selection-method-specific objective and constraints)
    2) "select_and_evaluate()" (find feature sets under the constraints stored in a solver)
    Optionally, you can override "set_data()" to do dataset-specific pre-computations (once per
    dataset instead of once for each solver initialization).
    """

    # Initialize all fields.
    def __init__(self):
        self._X_train = None  # pd.DataFrame
        self._X_test = None  # pd.DataFrame; for evaluation only, not optimization
        self._y_train = None  # pd.Series
        self._y_test = None  # pd.Series
        self._n = None  # int; total number of features

    # Set the data for the search for alternative feature sets. You can override this method if you
    # want to pre-compute some stuff for feature selection (so these computations can be reused if
    # feature selection is called multiple times for the same dataset); even if you override,
    # please still call this original method to make sure the data is properly set.
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        assert X_train.shape[1] == X_test.shape[1], 'Train and test need same number of features.'
        assert X_train.shape[0] == y_train.shape[0], 'Train X, y need same number of samples.'
        assert X_test.shape[0] == y_test.shape[0], 'Test X, y need same number of samples.'
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._n = X_train.shape[1]

    # Return the train-test split of the feature selector as (X_train, X_test, y_train, y_test).
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return self._X_train, self._X_test, self._y_train, self._y_test

    # Return a fresh solver in which the constraints for alternatives will be stored. Do some
    # initialization that is independent from the feature-selection method (while
    # "initialize_solver()" is feature-selection-specific). This routine will be called once at the
    # beginning of each search for alternative feature sets.
    @staticmethod
    def create_solver() -> pywraplp.Solver:
        solver = pywraplp.Solver.CreateSolver('SCIP')  # see documentation for available solvers
        solver.SetNumThreads(1)  # we already parallelize experimental pipeline
        return solver

    # Initialize a solver (to find alternative feature sets) specific to the feature-selection
    # method, e.g., add an objective or constraints (if the feature-selection method has these).
    # This routine will be called once at the beginning of the search for alternative feature sets.
    # - "solver": The solver that will hold the objective and the constraints for alternatives.
    # - "s_list": The feature-selection decision variables. For sequential search, len(s_list) == 1,
    #   while simultaneous search considers multiple feature sets at once. For each feature set,
    #   there should be as many decision variables as there are features in the dataset.
    # - "k": The number of features to be selected.
    # - "objective_agg": How to aggregate the feature sets' qualities in the objective (if there are
    #   multiple feature sets optimized at once, i.e., in simultaneous search). Not used here, but
    # should be handled appropriately when defining the objective in subclasses.
    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        solver.SetTimeLimit(60000 * len(s_list))  # measured in milliseconds; proportional to ...
        # ... the number of feature sets per solver call (i.e., higher for simultaneous search)
        for s in s_list:
            solver.Add(solver.Sum(s) == k)

    # Feature-selection method that is used (as subroutine) in the search for alternatives.
    # See "initialize_solver()" for a description of the parameters.
    # Should returns a table of results, where each row is a feature set (column "selected_idxs",
    # type "Sequence[int]") accompanied by metrics (columns: "train_objective", "test_objective",
    # "optimization_time", "optimization_status").
    @abstractmethod
    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        raise NotImplementedError('Abstract method.')

    # Return a variable for the product between two binary variables from the same "solver".
    # Add the necessary constraints to make sure the new variable really represents the product.
    # See https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators ("AND" operator)
    @staticmethod
    def create_product_var(solver: pywraplp.Solver, var1: pywraplp.Variable,
                           var2: pywraplp.Variable) -> pywraplp.Variable:
        var_name = var1.name() + '*' + var2.name()
        var = solver.BoolVar(name=var_name)
        solver.Add(var <= var1)
        solver.Add(var <= var2)
        solver.Add(1 + var >= var1 + var2)
        return var

    # Return a constraint such that the dissimilarity between two feature sets of desired size "k"
    # is over a threshold "tau". The selection decisions "s2" for the 2nd feature set are unknown
    # (thus, they are variables), while the selection decisions "s1" for the first feature set can
    # be either unknown (variables) or known (vector of 0/1 or false/true).
    # Currently supported for the dissimilarity "d_name" are "dice" and "jaccard". For "dice", you
    # can also provide an absolute number of differing features "tau_abs" instead of a relative
    # "tau" from the interval [0, 1].
    @staticmethod
    def create_pairwise_alternative_constraint(
            solver: pywraplp.Solver, s1: Sequence[Union[pywraplp.Variable, int, bool]],
            s2: Sequence[pywraplp.Variable], k: int, tau: Optional[float] = None,
            tau_abs: Optional[int] = None, d_name: str = 'dice') -> pywraplp.LinearConstraint:
        assert len(s1) == len(s2), 'Decision vectors s1 and s2 need to have same length.'
        if isinstance(s1[0], pywraplp.Variable):  # both feature sets undetermined
            overlap_size = solver.Sum([AlternativeFeatureSelector.create_product_var(
                solver=solver, var1=s1_j, var2=s2_j) for (s1_j, s2_j) in zip(s1, s2)])
        else:  # one feature set known, so plain sum
            overlap_size = solver.Sum([s2_j for (s1_j, s2_j) in zip(s1, s2) if s1_j])
        if d_name == 'dice':  # as same size of both sets, also equivalent to some other measures
            if tau_abs is not None:
                return overlap_size <= k - tau_abs
            return overlap_size <= (1 - tau) * k
        if d_name == 'jaccard':
            return overlap_size <= (1 - tau) / (2 - tau) * 2 * k
        raise ValueError('Unknown dissimilarity measure.')

    # Sequentially search for alternative feature sets, iteratively adding constraints and calling
    # a feature-selection method that need to be implemented in a subclass.
    # - "k": number of features to select
    # - "num_alternatives": number of returned feature sets - 1 (first set is considered "original")
    # - "tau": relative (i.e., in [0,1]) dissimilarity threshold for being alternative
    # - "tau_abs": absolute number of differing features (only works for "dice")
    # - "d_name": name of set dissimilarity measure (currently supported: "dice", "jaccard")
    # Return a table of results ("selected_idx", "train_objective", "test_objective",
    # "optimization_time", "optimization_status").
    def search_sequentially(self, k: int, num_alternatives: int, tau: Optional[float] = None,
                            tau_abs: Optional[int] = None, d_name: str = 'dice') -> pd.DataFrame:
        results = []
        solver = AlternativeFeatureSelector.create_solver()
        s = [solver.BoolVar(name=f's_{j}') for j in range(self._n)]
        s_list = [s]  # only search for one feature set at a time
        self.initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg='sum')
        results.append(self.select_and_evaluate(solver=solver, s_list=s_list))  # "original" set
        for _ in range(num_alternatives):
            if not math.isnan(results[-1]['train_objective'].iloc[0]):  # if not infeasible
                s_value = [j in results[-1]['selected_idxs'].iloc[0] for j in range(self._n)]
                # Feature set different to previous selection:
                solver.Add(AlternativeFeatureSelector.create_pairwise_alternative_constraint(
                    solver=solver, s1=s_value, s2=s, k=k, tau=tau, tau_abs=tau_abs, d_name=d_name))
            results.append(self.select_and_evaluate(solver=solver, s_list=s_list))
        return pd.concat(results, ignore_index=True)

    # Simultaneously search for alternative feature sets, only generating constraints once and then
    # calling a feature-selection method that need to be implemented in a subclass.
    # - "k": number of features to select
    # - "num_alternatives": number of returned feature sets - 1 (one set is considered "original")
    # - "tau": relative (i.e., in [0,1]) dissimilarity threshold for being alternative
    # - "tau_abs": absolute number of differing features (only works for "dice")
    # - "d_name": name of set dissimilarity measure (currently supported: "dice", "jaccard")
    # - "objective_agg": How to aggregate the feature sets' qualities in the objective.
    # Return a table of results ("selected_idx", "train_objective", "test_objective",
    # "optimization_time", "optimization_status").
    def search_simultaneously(self, k: int, num_alternatives: int, tau: Optional[float] = None,
                              tau_abs: Optional[int] = None, d_name: str = 'dice',
                              objective_agg: str = 'sum') -> pd.DataFrame:
        solver = AlternativeFeatureSelector.create_solver()
        s_list = []
        for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
            s = [solver.BoolVar(name=f's{i}_{j}') for j in range(self._n)]
            for s2 in s_list:
                solver.Add(AlternativeFeatureSelector.create_pairwise_alternative_constraint(
                    solver=solver, s1=s, s2=s2, k=k, tau=tau, tau_abs=tau_abs, d_name=d_name))
            s_list.append(s)
        self.initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        return self.select_and_evaluate(solver=solver, s_list=s_list)


class WhiteBoxFeatureSelector(AlternativeFeatureSelector, metaclass=ABCMeta):
    """Feature Selection as White-Box Problem

    (Abstract) white-box feature selector. Subclasses need to define the method for creating the
    (train and test) objectives.
    """

    # Initialize all fields.
    def __init__(self):
        super().__init__()
        self._Q_train_list = None  # Sequence[pywraplp.LinearExpr]; objectives for the feature sets
        self._Q_test_list = None  # Sequence[pywraplp.LinearExpr]; for evaluation, not optimization

    # Should return expressions for train objective and test objective (one per desired feature
    # set, i.e., element from "s_list"), using variables from "s_list", data stored in "self", and
    # potentially the number of features "k". Might add auxiliary variables and constraints to the
    # "solver".
    @abstractmethod
    def create_objectives(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]], k: int) \
            -> Tuple[Sequence[pywraplp.LinearExpr], Sequence[pywraplp.LinearExpr]]:
        raise NotImplementedError('Abstract method.')

    # Initialize the solver by creating an expression of the objective for each feature set in
    # "s_list" and aggregating this to an overall objective.
    # See the superclass for a description of the parameters.
    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        super().initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        self._Q_train_list, self._Q_test_list = self.create_objectives(solver=solver, s_list=s_list,
                                                                       k=k)
        if objective_agg == 'sum':
            objective = solver.Sum(self._Q_train_list)  # sum over all feature sets
        elif objective_agg == 'min':
            Q_min = solver.NumVar(name='Q_min', lb=float('-inf'), ub=float('inf'))
            for i in range(len(s_list)):
                solver.Add(Q_min <= self._Q_train_list[i])
            objective = Q_min
        else:
            raise ValueError('Unknown objective aggregation.')
        solver.Maximize(objective)

    # Run feature-selection with the solver. As this class represents a white-box objective (set in
    # "initialize_solver()"), we can directly run the optimization routine of the "solver".
    # See the superclass for a description of the parameters and the return value.
    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        start_time = time.process_time()
        optimization_status = solver.Solve()
        end_time = time.process_time()
        # As we limit the optimization time, the result might be feasible but suboptimal:
        if optimization_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            result = pd.DataFrame({
                'selected_idxs': [[j for (j, s_j) in enumerate(s) if s_j.solution_value()]
                                  for s in s_list],
                'train_objective': [Q_s.solution_value() for Q_s in self._Q_train_list],
                'test_objective': [Q_s.solution_value() for Q_s in self._Q_test_list]
            })
        else:
            result = pd.DataFrame({
                'selected_idxs': [[] for _ in s_list],
                'train_objective': [float('nan') for _ in self._Q_train_list],
                'test_objective': [float('nan') for _ in self._Q_test_list]
            })
        result['optimization_time'] = end_time - start_time
        result['optimization_status'] = optimization_status
        return result


class LinearQualityFeatureSelector(WhiteBoxFeatureSelector, metaclass=ABCMeta):
    """Feature Selection with Linear Quality Function

    (Abstract) white-box feature selector whose objective function is the sum of the individual
    features' qualities. This allows pre-computing the qualities when the data is set and re-using
    these qualities for multiple selection/alternative-search runs. Subclasses need to define the
    function for computing the qualities.
    """

    # Initialize all fields.
    def __init__(self):
        super().__init__()
        self._q_train = None  # Iterable[float]; qualities of the individual features
        self._q_test = None  # Iterable[float]; for evaluation only, not optimization

    # Should return a sequence of qualities with len(result) == X.shape[1], i.e., one quality value
    # for each feature.
    @abstractmethod
    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> Iterable[float]:
        raise NotImplementedError('Abstract method.')

    # Set the data for the search for alternative feature sets and pre-compute the features'
    # qualities for the objective.
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._q_train = self.compute_qualities(X=X_train, y=y_train)
        self._q_test = self.compute_qualities(X=X_test, y=y_test)

    # Create linear train and test objectives from the features' qualities.
    # See the superclass for a description of the parameters and the return value.
    def create_objectives(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]], k: int) \
            -> Tuple[Sequence[pywraplp.LinearExpr], Sequence[pywraplp.LinearExpr]]:
        return ([solver.Sum([q_j * s_j for (q_j, s_j) in zip(self._q_train, s)]) for s in s_list],
                [solver.Sum([q_j * s_j for (q_j, s_j) in zip(self._q_test, s)]) for s in s_list])

    # Greedy-replacement search for alternative feature sets. Optimal for univariate feature
    # qualities (objective: sum of qualities over features and feature sets) as long a there are
    # unused features. Works without a solver. Tailored to the Dice dissimilarity.
    # - "k": number of features to select
    # - "num_alternatives": number of returned feature sets - 1 (first set is considered "original")
    # - "tau": relative (i.e., in [0,1]) dissimilarity threshold for being alternative
    # - "tau_abs": absolute number of differing features
    # Return a table of results ("selected_idx", "train_objective", "test_objective",
    # "optimization_time", "optimization_status").
    def search_greedy_replacement(self, k: int, num_alternatives: int, tau: Optional[float] = None,
                                  tau_abs: Optional[int] = None) -> pd.DataFrame:
        if tau is not None:
            tau_abs = math.ceil(tau * k)
        if k > self._n:  # not enough features for selection
            return pd.DataFrame()
        results = []
        indices = np.argsort(self._q_train)[::-1]  # descending order by qualities
        s = [0] * self._n  # initial selection for all alternatives
        position = 0  # index of index of currently selected feature
        while position < k - tau_abs:
            j = indices[position]
            s[j] = 1
            position = position + 1
        i = 0  # number of current alternative
        while i <= num_alternatives and i <= ((self._n - k) / tau_abs):
            s_i = s.copy()  # select best "k - tau_abs" features
            for _ in range(tau_abs):  # select remaining "tau_abs" features
                j = indices[position]
                s_i[j] = 1
                position = position + 1
            results.append(s_i)
            i = i + 1
        # Transform into a result structure consistent to the other searches in the top-level class:
        results = [{
            'selected_idxs': [j for (j, s_j) in enumerate(s) if s_j],
            'train_objective': sum(q_j * s_j for (q_j, s_j) in zip(self._q_train, s)),
            'test_objective': sum(q_j * s_j for (q_j, s_j) in zip(self._q_test, s)),
            'optimization_status': pywraplp.Solver.OPTIMAL
        } for s in results]
        for i in range(i, num_alternatives + 1):  # in case algorithm ran out of features early
            results.append({
                'selected_idxs': [],
                'train_objective': float('nan'),
                'test_objective': float('nan'),
                'optimization_status': pywraplp.Solver.NOT_SOLVED  # not necessarily infeasible
            })
        results = pd.DataFrame(results)
        results['optimization_time'] = float('nan')  # not worth to measure it
        return results


class MISelector(LinearQualityFeatureSelector):
    """Feature Selection with Mutual Information

    Univariate filter feature selector based on mutual information between features and target.
    White-box optimization approach.
    """

    # Compute mutual information in a deterministic manner (fixed random state). Consistently use
    # regression estimator, as not only "X", but also "y" could be continuous (e.g., for regression
    # problems or if we call this function to consider feature-feature dependencies).
    @staticmethod
    def mutual_info(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        return sklearn.feature_selection.mutual_info_regression(X=X, y=y, random_state=25)

    # Compute the features' qualities as mutual information to the target. Normalize such that the
    # qualities sum up to 1 (so we have relative feature importances).
    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        qualities = MISelector.mutual_info(X=X, y=y)
        qualities = qualities / qualities.sum()
        return qualities


class FCBFSelector(MISelector):
    """Feature Selection with Fast Correlation-Based Filter

    Multivariate filter feature selector that uses mutual information between features and target as
    objective. Additionally, there are constraints on the mutual information between features (for
    each selected feature, dependency to target should be > than to each other selected feature).

    Inspired by Yu et al. (2003): "Feature Selection for High-Dimensional Data: A Fast Correlation-
    Based Filter Solution" (which searches heuristically rather than optimizing exactly).
    White-box optimization approach.
    """

    # Initialize all fields.
    def __init__(self):
        super().__init__()
        self._mi_target = None  # np.ndarray; dependency of each feature to target
        self._mi_features = None  # Sequence[np.ndarray]; dependency of each feature to each other

    # Set the data for the search for alternative feature sets. Pre-compute the feature-target and
    # feature-feature dependencies that will be used for constraints later.
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        # We already have the feature-target MI in "_q_train", but normalized, so compute again:
        self._mi_target = MISelector.mutual_info(X=self._X_train, y=self._y_train)
        self._mi_features = [MISelector.mutual_info(X=self._X_train, y=self._X_train[feature])
                             for feature in self._X_train.columns]

    # Initialize the "solver" by creating linear objectives from the features' qualities and
    # adding constraints on the feature-feature dependencies.
    # See the superclass for a description of the parameters.
    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        super().initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        # Note that the MI estimator in sklearn is not perfectly bivariate and symmetric, as it uses
        # *one* random-number generator to *iteratively* add some noise to *all* features and
        # target; e.g., if you re-order features in X, MI estimates change slightly, though the
        # procedure still is deterministic (if MI seeded) for the overall (X, y) combination.
        for j_1 in range(1, self._n):  # 1st feature
            for j_2 in range(j_1):  # 2nd feature
                # If inter-correlation >= target correlation, do not select both features at once:
                if ((self._mi_target[j_1] <= self._mi_features[j_1][j_2]) or
                        (self._mi_target[j_2] <= self._mi_features[j_2][j_1])):
                    for s in s_list:
                        solver.Add(s[j_1] + s[j_2] <= 1)


class ModelImportanceSelector(LinearQualityFeatureSelector):
    """Feature Selection based on Model Importance

    Post-hoc importances (here: from scikit-learn's feature_importances_) for feature selection.
    White-box optimization approach.
    """

    # Initialize all fields. If no "prediction_model" is provided, create a decision tree as default
    # one. A user-provided model should have an attribute "feature_importances_" after fitting (does
    # not need to be fit yet, though, ony initialized).
    def __init__(self, prediction_model: Optional[sklearn.base.BaseEstimator] = None):
        super().__init__()
        if prediction_model is None:
            prediction_model = prediction.create_model_for_fs()
        self._prediction_model = prediction_model

    # Compute the features' qualities by fitting a model and extracting importances from it.
    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        return self._prediction_model.fit(X=X, y=y).feature_importances_


class MRMRSelector(WhiteBoxFeatureSelector):
    """Feature Selection with Minimal Redundancy Maximal Relevance

    Multivariate filter feature selector that uses
    1) mutual information between features and target as relevance criterion
    2) mutual information between features as redundancy criterion
    White-box optimization approach.

    See Peng et al. (2005): "Feature Selection Based on Mutual Information: Criteria of
    Max-Dependency, Max-Relevance, and Min-Redundancy".
    """

    # Initialize all fields.
    def __init__(self):
        super().__init__()
        self._mi_target_train = None  # np.ndarray; feature-target dependencies
        self._mi_target_test = None  # np.ndarray; feature-target dependencies
        self._mi_features_train = None  # Sequence[np.ndarray]; feature-feature dependencies
        self._mi_features_test = None  # Sequence[np.ndarray]; feature-feature dependencies

    # Set the data that will be used during search for alternative feature sets. Pre-compute the
    # feature-target and feature-feature dependencies that will be used for objective later.
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._mi_target_train = MISelector.mutual_info(X=self._X_train, y=self._y_train)
        self._mi_target_test = MISelector.mutual_info(X=self._X_test, y=self._y_test)
        self._mi_features_train = [MISelector.mutual_info(X=self._X_train, y=self._X_train[feature])
                                   for feature in self._X_train.columns]
        self._mi_features_test = [MISelector.mutual_info(X=self._X_test, y=self._X_test[feature])
                                  for feature in self._X_test.columns]
        # Set self-redundancy to zero, to not penalize features for their intrinsic entropy (which
        # can differ between features). See Ngyuen et al. (2014): "Effective Global Approaches for
        # Mutual Information Based Feature Selection"
        for j in range(self._n):
            self._mi_features_train[j][j] = 0
            self._mi_features_test[j][j] = 0
        # Max-normalize MI values (min is 0 anyway, in theory due to definition of MI and in
        # practice due to our self-redundancy fix above); to be consistent to the other feature
        # selectors, normalize train and test with separate max values:
        max_mi_train = max(self._mi_target_train.max(),
                           max(x.max() for x in self._mi_features_train))
        self._mi_target_train = self._mi_target_train / max_mi_train
        self._mi_features_train = [x / max_mi_train for x in self._mi_features_train]
        max_mi_test = max(self._mi_target_test.max(), max(x.max() for x in self._mi_features_test))
        self._mi_target_test = self._mi_target_test / max_mi_test
        self._mi_features_test = [x / max_mi_test for x in self._mi_features_test]

    # Create objective by considering feature relevance and redundancy. Due to a special encoding of
    # interaction terms (using real variables specific to training-set feature qualities instead of
    # binary interaction variables), we cannot give a closed-form expression for the test objective
    # here, but return the train objective twice and calculate the actual test objective in
    # via "compute_test_objective()" called in "select_and_evaluate()".
    # See the superclass for a description of the parameters and the return value.
    def create_objectives(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]], k: int) \
            -> Tuple[Sequence[pywraplp.LinearExpr], Sequence[pywraplp.LinearExpr]]:
        objectives = []
        for i, s in enumerate(s_list):
            relevance = solver.Sum([q_j * s_j for (q_j, s_j) in zip(self._mi_target_train, s)])
            relevance = relevance / k
            redundancy_terms = []
            M = max(sum(x) for x in self._mi_features_train) + 1  # used to "deactivate" constraints
            for j_1 in range(len(s)):
                # Linearization: z_i = x_i * A_i(x) (follows Equation (14) in Nguyen et al. (2010)
                # "Towards a Generic Feature-Selection Measure for Intrusion Detection" except there
                # is no variable "y" since we do not have a fractional expression (we only divide by
                # constants); we have max instead of min objective, but since the redundancy
                # term is subtracted (should be minimized), we can still use Equation (14))
                z_j = solver.NumVar(name=f'z_{i}_{j_1}', lb=0, ub=M)
                A_j = solver.Sum([self._mi_features_train[j_1][j_2] * s[j_2]
                                  for j_2 in range(len(s))])
                solver.Add(M * (s[j_1] - 1) + A_j <= z_j)
                redundancy_terms.append(z_j)
            redundancy = solver.Sum(redundancy_terms) / (k * (k - 1))  # -1 as self-redundancy == 0
            objectives.append(relevance - redundancy)
        return (objectives, objectives)  # no closed-form expression for test objective

    # Compute the test-set objective for a given feature-selection "selected_idxs".
    def compute_test_objective(self, selected_idxs: Sequence[int]) -> float:
        k = len(selected_idxs)
        if k == 0:
            return float('nan')
        relevance = sum(self._mi_target_test[j] for j in selected_idxs) / k
        redundancy = sum(self._mi_features_test[j_1][j_2] for j_1 in selected_idxs
                         for j_2 in selected_idxs) / (k * (k - 1))  # -1 since self-redundancy == 0
        return relevance - redundancy

    # Run feature-selection with the solver. Due to the reason described in "create_objectives()",
    # we need to compute the test objectives manually.
    # See the superclass for a description of the parameters and the return value.
    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        result = super().select_and_evaluate(solver=solver, s_list=s_list)
        result['test_objective'] = [self.compute_test_objective(selected_idxs=selected_idxs)
                                    for selected_idxs in result['selected_idxs']]
        return result


class GreedyWrapperSelector(AlternativeFeatureSelector):
    """Greedy Wrapper Feature Selection

    Simple hill-climbing approach to select features. Feature-set quality is evaluated with a
    prediction model. Solver only checks validity of feature sets. Black-box optimization approach.
    """

    # Initialize all fields. If no "prediction_model" is provided, create a decision tree as default
    # one. "max_iters" is the maximum number of calls to the solver in the wrapper search
    # (solver might not only yield valid feature sets, which are evaluated, but also invalid ones).
    def __init__(self, prediction_model: Optional[sklearn.base.BaseEstimator] = None,
                 max_iters: int = 1000):
        super().__init__()
        if prediction_model is None:
            prediction_model = prediction.create_model_for_fs()
        self._prediction_model = prediction_model
        self._max_iters = max_iters
        self._objective_agg = None

    # Set a constant objective, as the solver should only check validity of solutions in this
    # approach (the actual objective, i.e., prediction performance, is a black box for the solver).
    # See the superclass for a description of the parameters.
    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        super().initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        self._objective_agg = objective_agg
        objective = 0
        solver.Maximize(objective)

    # Decide if the overall feature-set quality has improved based on some aggregation function
    # (which is stored in a field of "self").
    # - "old_Q_list": Qualities of the previously selected feature sets.
    # - "new_Q_list": Qualities of the newly selected feature sets.
    def has_objective_improved(self, old_Q_list: Iterable[float],
                               new_Q_list: Iterable[float]) -> bool:
        if self._objective_agg == 'sum':
            return sum(new_Q_list) > sum(old_Q_list)
        if self._objective_agg == 'min':
            return min(new_Q_list) > min(old_Q_list)
        raise ValueError('Unknown objective aggregation.')

    # Run a greedy hill-climbing procedure to select features. In particular, start with a feature
    # set satisfying all constraints and systematically try flipping the selection decisions of
    # individual features. Continue with a new solution if it has higher quality than the currently
    # best one. Use a prediction model to evaluate qualities of the feature sets. Stop after a fixed
    # number of iterations (= solver calls) or if no new valid feature set can be generated.
    # See the superclass for a description of the parameters and the return value.
    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        start_time = time.process_time()
        optimization_status = solver.Solve()  # actually only check constraint satisfcation
        iters = 1  # solver called for first time
        if optimization_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            result = pd.DataFrame({
                'selected_idxs': [[] for _ in s_list],
                'train_objective': [float('nan') for _ in s_list],
                'test_objective': [float('nan') for _ in s_list]
            })  # not even one valid alternative found
        else:
            s_value_list = [[bool(s_j.solution_value()) for s_j in s] for s in s_list]
            # Note that "training" quality actually is validation performance with a holdout split
            Q_train_list = [prediction.evaluate_wrapper(
                model=self._prediction_model, X=self._X_train.iloc[:, s_value], y=self._y_train)
                for s_value in s_value_list]
            j = 0  # other than pseudo-code in paper, 0-indexing here
            swap_variables = []
            while (iters < self._max_iters) and (j < self._n):
                # We can't add temporary constraints (no remove function), but we can fix variables
                for s, s_value in zip(s_list, s_value_list):  # fix s_j to inverse of previous value
                    s[j].SetBounds(1 - s_value[j], 1 - s_value[j])
                    swap_variables.append(s[j])
                optimization_status = solver.Solve()
                iters = iters + 1
                if optimization_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                    current_s_value_list = [[bool(s_j.solution_value()) for s_j in s]
                                            for s in s_list]
                    current_Q_train_list = [prediction.evaluate_wrapper(
                        model=self._prediction_model, X=self._X_train.iloc[:, s_value],
                        y=self._y_train) for s_value in current_s_value_list]
                    if self.has_objective_improved(old_Q_list=Q_train_list,
                                                   new_Q_list=current_Q_train_list):
                        s_value_list = current_s_value_list
                        Q_train_list = current_Q_train_list
                        j = 0  # re-start swapping with first feature (zero indexing!)
                    else:
                        j = j + 1
                else:
                    j = j + 1
                for s_j in swap_variables:
                    s_j.SetBounds(0, 1)  # revert fixing to one value (make regular binary again)
                swap_variables.clear()  # next iteration will swap at different position
            selected_idxs = [[j for (j, s_j_value) in enumerate(s_value) if s_j_value]
                             for s_value in s_value_list]
            result = pd.DataFrame({
                'selected_idxs': selected_idxs,
                'train_objective': Q_train_list,
                # To be consistent to other FS techniques, we evaluate the test objective by
                # considering test data only, i.e., we make a holdout split of the test data, train
                # a new prediction model on one part of it and evaluate the model on the rest of it;
                # the main experimental pipeline contains the "classic" evaluation with training
                # model on (full) training set and predicting on full train + test
                'test_objective': [prediction.evaluate_wrapper(
                    model=self._prediction_model, X=self._X_test.iloc[:, idxs],
                    y=self._y_test) for idxs in selected_idxs]
            })
        end_time = time.process_time()
        result['optimization_time'] = end_time - start_time
        result['optimization_status'] = optimization_status
        result['wrapper_iters'] = iters
        return result
