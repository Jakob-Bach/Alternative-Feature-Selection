"""Alternative Feature Selection

Classes for alternative feature selection, i.e.,
- constraints and search routines for alternatives
- feature-selection methods (determining the objective of the search)

In this experimental file, "pulp" replaces "mip" for optimization. To use this file,
checkout b032860a86 (newer versions might work as well) and replace "afs.py" with this file.
"""


from abc import ABCMeta, abstractmethod
import math
import time
from typing import Iterable, Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp
import sklearn.base
import sklearn.feature_selection

import prediction


class AlternativeFeatureSelector(metaclass=ABCMeta):
    """Alternative Feature Selector

    The base class for alternative feature selection. Contains the search routines for alternatives
    (simultaneous, sequential) and various helper functions to formulate constraints. Class is
    abstract because an actual feature-selection method (used as subroutine in the search for
    alternatives) is not implemented here.
    """

    # Initialize all fields.
    def __init__(self):
        self._X_train = None  # pd.DataFrame
        self._X_test = None  # pd.DataFrame; for evaluation only, not optimization
        self._y_train = None  # pd.Series
        self._y_test = None  # pd.Series
        self._n = None  # int; total number of features
        self._solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=60)

    # Set the data that will be used during search for alternative feature sets. You can override
    # this method if you want to pre-compute some stuff for feature-selection (so these computations
    # can be reused if feature selection is called multiple times for the same dataset).
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

    # Initialize a problem for alternative feature sets specific to the feature-selection method
    # used, e.g., add an objective or constraints (if the feature-selection method has these).
    # This routine will be called at the  beginning of the search for alternative feature sets.
    # -"problem": The PuLP problem that will hold the constraints for alternatives.
    # - "s_list": The feature-selection decision variables. For sequential search, len(s_list) == 1,
    #   while simultaneous search considers multiple feature sets at once. For each feature sets,
    #   there should be as many decision variables as there are features in the dataset.
    @abstractmethod
    def initialize_problem(self, problem: pulp.LpProblem,
                           s_list: Sequence[Sequence[pulp.LpVariable]]) -> None:
        raise NotImplementedError('Abstract method.')

    # Feature-selection method that is used (as subroutine) in the search for alternatives.
    # See "initialize_problem()" for a description of the parameters.
    # Should returns a table of results, where each row is a feature set (column "selected_idxs",
    # type "Sequence[int]") accompanied by metrics (columns: "train_objective", "test_objective",
    # "optimization_time", "optimization_status").
    @abstractmethod
    def select_and_evaluate(self, problem: pulp.LpProblem,
                            s_list: Sequence[Sequence[pulp.LpVariable]]) -> pd.DataFrame:
        raise NotImplementedError('Abstract method.')

    # Return a variable for the product between two binary variables from the same PuLP problem.
    # Add the necessary constraints to make sure the new variable really represents the product.
    # See https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators ("AND" operator)
    @staticmethod
    def create_product_var(problem: pulp.LpProblem, var1: pulp.LpVariable,
                           var2: pulp.LpVariable) -> pulp.LpVariable:
        var_name = var1.name + '*' + var2.name
        var = pulp.LpVariable(name=var_name, cat=pulp.const.LpBinary)
        problem.addConstraint(var <= var1)
        problem.addConstraint(var <= var2)
        problem.addConstraint(1 + var >= var1 + var2)
        return var

    # Return a constraint such that the dissimilarity between two feature sets of desired size "k"
    # is over a threshold "tau". The selection decisions "s2" for the 2nd feature set are unknown
    # (thus, they are PuLP variables), while the decisions "s1" for the first feature set can be
    # either unknown (PuLP variables) or known (vector of 0/1 or false/true).
    # Currently supported for "d_name" are "dice" and "jaccard". For "dice", you can also provide
    # an absolute number of differing features "tau_abs" instead of a relative "tau" from [0, 1].
    @staticmethod
    def create_pairwise_alternative_constraint(
            problem: pulp.LpProblem, s1: Sequence[Union[pulp.LpVariable, int, bool]],
            s2: Sequence[pulp.LpVariable], k: int, tau: Optional[float] = None,
            tau_abs: Optional[int] = None, d_name: str = 'dice') -> pulp.LpAffineExpression:
        assert len(s1) == len(s2), 'Decision vectors s1 and s2 need to have same length.'
        assert isinstance(s2[0], pulp.LpVariable), 's2 needs to contain variables.'
        if isinstance(s1[0], pulp.LpVariable):  # both features sets undetermined
            overlap_size = pulp.lpSum(AlternativeFeatureSelector.create_product_var(
                problem=problem, var1=s1_j, var2=s2_j) for (s1_j, s2_j) in zip(s1, s2))
        else:  # one feature set known, so plain sum
            overlap_size = pulp.lpSum(s2_j for (s1_j, s2_j) in zip(s1, s2) if s1_j)
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
        problem = pulp.LpProblem(sense=pulp.const.LpMaximize)
        s = [pulp.LpVariable(name=f's_{j}', cat=pulp.const.LpBinary) for j in range(self._n)]
        s_list = [s]  # only search for one feature set at a time
        problem.addConstraint(pulp.lpSum(s) == k)  # select exactly k
        self.initialize_problem(problem=problem, s_list=s_list)
        results.append(self.select_and_evaluate(problem=problem, s_list=s_list))  # "original" set
        for _ in range(num_alternatives):
            if not math.isnan(results[-1]['train_objective'].iloc[0]):  # if not infeasible
                s_value = [j in results[-1]['selected_idxs'].iloc[0] for j in range(self._n)]
                # Feature set different to previous selection:
                problem.addConstraint(AlternativeFeatureSelector.create_pairwise_alternative_constraint(
                    problem=problem, s1=s_value, s2=s, k=k, tau=tau, tau_abs=tau_abs, d_name=d_name))
            results.append(self.select_and_evaluate(problem=problem, s_list=s_list))
        return pd.concat(results, ignore_index=True)

    # Simultaneously search for alternative feature sets, only generating constraints once and then
    # calling a feature-selection method that need to be implemented in a subclass.
    # - "k": number of features to select
    # - "num_alternatives": number of returned feature sets - 1 (one set is considered "original")
    # - "tau": relative (i.e., in [0,1]) dissimilarity threshold for being alternative
    # - "tau_abs": absolute number of differing features (only works for "dice")
    # - "d_name": name of set dissimilarity measure (currently supported: "dice", "jaccard")
    # Return a table of results ("selected_idx", "train_objective", "test_objective",
    # "optimization_time", "optimization_status").
    def search_simultaneously(self, k: int, num_alternatives: int, tau: Optional[float] = None,
                              tau_abs: Optional[int] = None, d_name: str = 'dice') -> pd.DataFrame:
        problem = pulp.LpProblem(sense=pulp.const.LpMaximize)
        s_list = []
        for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
            s = [pulp.LpVariable(name=f's{i}_{j}', cat=pulp.const.LpBinary) for j in range(self._n)]
            problem.addConstraint(pulp.lpSum(s) == k)
            for s2 in s_list:
                problem.addConstraint(AlternativeFeatureSelector.create_pairwise_alternative_constraint(
                    problem=problem, s1=s, s2=s2, k=k, tau=tau, tau_abs=tau_abs, d_name=d_name))
            s_list.append(s)
        self.initialize_problem(problem=problem, s_list=s_list)
        return self.select_and_evaluate(problem=problem, s_list=s_list)


class LinearQualityFeatureSelector(AlternativeFeatureSelector):
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
        self._Q_train_list = None  # Sequence[mip.LinExpr]; objectives for the feature sets
        self._Q_test_list = None  # Sequence[mip.LinExpr]; for evaluation only, not optimization

    # Should return a sequence of qualities with len(result) == X.shape[1], i.e., one quality value
    # for each feature.
    @abstractmethod
    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> Iterable[float]:
        raise NotImplementedError('Abstract method.')

    # Set the data that will be used during search for alternative feature sets and pre-compute the
    # features' qualities.
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._q_train = self.compute_qualities(X=X_train, y=y_train)
        self._q_test = self.compute_qualities(X=X_test, y=y_test)

    # Initialize the PuLP problem by creating linear objectives from the features' qualities.
    # See the superclass for a description of the parameters.
    def initialize_problem(self, problem: pulp.LpProblem,
                           s_list: Sequence[Sequence[pulp.LpVariable]]) -> None:
        self._Q_train_list = [pulp.lpSum(q_j * s_j for (q_j, s_j) in zip(self._q_train, s))
                              for s in s_list]
        self._Q_test_list = [pulp.lpSum(q_j * s_j for (q_j, s_j) in zip(self._q_test, s))
                             for s in s_list]
        objective = pulp.lpSum(self._Q_train_list)  # sum over all feature sets
        problem += objective

    # Run feature-selection on the "problem." As this class represents a white-box objective, we can
    # directly run the optimization routine.
    # See the superclass for a description of the parameters and the return value.
    def select_and_evaluate(self, problem: pulp.LpProblem,
                            s_list: Sequence[Sequence[pulp.LpVariable]]) -> pd.DataFrame:
        start_time = time.perf_counter()
        problem.solve(solver=self._solver)
        end_time = time.perf_counter()
        # As we limit the optimization time, the result might be feasible but suboptimal
        if ((problem.sol_status == pulp.const.LpSolutionOptimal) or
                (problem.sol_status == pulp.const.LpSolutionIntegerFeasible)):
            result = pd.DataFrame({
                'selected_idxs': [[j for (j, s_j) in enumerate(s) if round(s_j.value())]
                                  for s in s_list],
                'train_objective': [Q_s.value() for Q_s in self._Q_train_list],
                'test_objective': [Q_s.value() for Q_s in self._Q_test_list]
            })
        else:
            result = pd.DataFrame({
                'selected_idxs': [[] for _ in s_list],
                'train_objective': [float('nan') for _ in self._Q_train_list],
                'test_objective': [float('nan') for _ in self._Q_test_list]
            })
        result['optimization_time'] = end_time - start_time
        result['optimization_status'] = problem.sol_status
        return result


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
    White-box optimization approach.
    """

    # Initialize all fields.
    def __init__(self):
        super().__init__()
        self._mi_target = None  # np.ndarray; dependency of each feature to target
        self._mi_features = None  # Sequence[np.ndarray]; dependency of each feature to each other

    # Set the data that will be used during search for alternative feature sets. Pre-compute the
    # feature-target and feature-feature dependencies that will be used for constraints later.
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._mi_target = MISelector.mutual_info(X=self._X_train, y=self._y_train)
        self._mi_features = [MISelector.mutual_info(X=self._X_train, y=self._X_train[feature])
                             for feature in self._X_train.columns]

    # Initialize the PuLP problem by creating linear objectives from the features' qualities and
    # adding constraints on the feature-feature depencies.
    # See the superclass for a description of the parameters.
    def initialize_problem(self, problem: pulp.LpProblem,
                           s_list: Sequence[Sequence[pulp.LpVariable]]) -> None:
        super().initialize_problem(problem=problem, s_list=s_list)
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
                        problem.addConstraint(s[j_1] + s[j_2] <= 1)


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

    # Don't need to set objective, as the solver should only check validity of solutions in this
    # approach (the actual objective, i.e., prediction performance, is a black-box).
    # See the superclass for a description of the parameters.
    def initialize_problem(self, problem: pulp.LpProblem,
                           s_list: Sequence[Sequence[pulp.LpVariable]]) -> None:
        pass

    # Run a greedy hill-climbing procedure to select features. In particular, start with a feature
    # set satisfying all constraints and systematically try flipping the selection decisions of
    # individual features. Continue with a new solution if it has higher quality then the currently
    # best one. Use a prediction model to evaluate qualities of the feature sets. Stop after a fixed
    # number of iterations (= solver calls) or if no new valid feature set can be generated.
    # See the superclass for a description of the parameters and the return value.
    def select_and_evaluate(self, problem: pulp.LpProblem,
                            s_list: Sequence[Sequence[pulp.LpVariable]]) -> pd.DataFrame:
        start_time = time.perf_counter()
        problem.solve(solver=self._solver)
        iters = 1  # solver called for first time
        if ((problem.sol_status != pulp.const.LpSolutionOptimal) and
                (problem.sol_status != pulp.const.LpSolutionIntegerFeasible)):  # no valid solution
            result = pd.DataFrame({
                'selected_idxs': [[] for _ in s_list],
                'train_objective': [float('nan') for _ in s_list],
                'test_objective': [float('nan') for _ in s_list]
            })
        else:
            s_value_list = [[bool(round(s_j.value())) for s_j in s] for s in s_list]
            # Note that "training" quality actually is validation performance with a holdout split
            Q_train_list = [prediction.evaluate_wrapper(
                model=self._prediction_model, X=self._X_train.iloc[:, s_value], y=self._y_train)
                for s_value in s_value_list]
            j = 0  # other than pseudo-code in paper, 0-indexing here
            while (iters < self._max_iters) and (j < self._n):
                for (i, (s, s_value)) in enumerate(zip(s_list, s_value_list)):
                    # Add temporary swap constraints
                    problem.addConstraint(s[j] == 0 if s_value[j] else s[j] == 1, name=f'swap_{i}')
                problem.solve(solver=self._solver)
                iters = iters + 1
                if ((problem.sol_status == pulp.const.LpSolutionOptimal) or
                        (problem.sol_status == pulp.const.LpSolutionIntegerFeasible)):
                    current_s_value_list = [[bool(round(s_j.value())) for s_j in s] for s in s_list]
                    current_Q_train_list = [prediction.evaluate_wrapper(
                        model=self._prediction_model, X=self._X_train.iloc[:, s_value],
                        y=self._y_train) for s_value in current_s_value_list]
                    if sum(current_Q_train_list) > sum(Q_train_list):
                        s_value_list = current_s_value_list
                        Q_train_list = current_Q_train_list
                        j = 0  # re-start swapping with first feature (zero indexing!)
                    else:
                        j = j + 1
                else:
                    j = j + 1
                for i in range(len(s_value_list)):  # remove temporary swap constraints
                    del problem.constraints[f'swap_{i}']  # is an ordered dictionary
                    problem.modifiedConstraints.pop()
            selected_idxs = [[j for (j, s_j_value) in enumerate(s_value) if s_j_value]
                             for s_value in s_value_list]
            result = pd.DataFrame({
                'selected_idxs': selected_idxs,
                'train_objective': Q_train_list,
                # To be consistent to other FS techniques, we evaluate the test objective by
                # considering test data (to be exact: the validation part of a holdout split of it)
                # only; main experimental pipeline contains "classic" evaluation with training model
                # on full (without holdout split) training set and predicting on full train + test
                'test_objective': [prediction.evaluate_wrapper(
                    model=self._prediction_model, X=self._X_test.iloc[:, idxs],
                    y=self._y_test) for idxs in selected_idxs]
            })
        end_time = time.perf_counter()
        result['optimization_time'] = end_time - start_time
        result['optimization_status'] = problem.sol_status
        result['iters'] = iters
        return result
