"""Alternative feature selection

Classes (and methods) for alternative feature selection, i.e.,
- constraints and search routines for alternatives (mostly in an abstract superclass), particularly
  solver-based sequential search and simultaneous search (supported by all feature-selection
  methods) and heuristic search methods (for univariate feature qualities)
- feature-selection methods (determining the objective of the search) from different categories,
 i.e., filter, wrapper, and post-hoc importance

Literature
----------
- Bach (2023): "Finding Optimal Diverse Feature Sets with Alternative Feature Selection"
- Bach & Böhm (2024): "Alternative feature selection with user control"
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
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree


class AlternativeFeatureSelector(metaclass=ABCMeta):
    """Alternative feature selection

    The base class for alternative feature selection. Contains the search routines for alternatives
    (simultaneous, sequential) and various helper functions to formulate constraints. Class is
    abstract because an actual feature-selection method (particularly the objective and the
    constrained optimization routine), which is used as subroutine in the search for alternatives,
    is not implemented here.

    Subclasses need to override:

    1) :meth:`initialize_solver` (add objective and (optionally) constraints of a particular
       feature-selection method)
    2) :meth:`select_and_evaluate` (find feature sets under the constraints stored in the solver)

    Optionally, you can override :meth:`set_data` to do dataset-specific pre-computations (once
    per dataset instead of once for each solver initialization, which may come handy if certain
    values of feature-set quality can be reused for multiple solver runs).

    Literature
    ----------
    - Bach (2023): "Finding Optimal Diverse Feature Sets with Alternative Feature Selection"
    - Bach & Böhm (2024): "Alternative feature selection with user control"
    """

    def __init__(self):
        """Initialize alternative feature selection

        Define all fields that are used later, without assigning a proper value yet. In particular,
        there are field to store a dataset with a train-test split.
        """

        self._X_train = None  # pd.DataFrame
        self._X_test = None  # pd.DataFrame; for evaluation only, not optimization
        self._y_train = None  # pd.Series
        self._y_test = None  # pd.Series
        self._n = None  # int; total number of features

    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        """Set data for searching alternative feature sets

        Assumes a supervised prediction scenario (i.e., features and target) with a predefined
        train-test split. You can override this method if you want to pre-compute some stuff
        (like feature qualities) for feature selection, so these computations can be reused if
        alternative feature selection is called multiple times for the same dataset; even if you
        override, please still call this original method to make sure the data is properly set.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature values for the training part of the dataset. Each row is a data object, each
            column a feature.
        X_test : pd.DataFrame
            Feature values for the test part of the dataset. Must have the same number of columns
            (features) as `X_train`.
        y_train : pd.Series
            Training part of the prediction target. Must have the same number of entries as
            `X_train` has rows.
        y_test : pd.Series
            Test part of the prediction target. Must have the same number of entries as `X_test`
            has rows.
        """

        assert X_train.shape[1] == X_test.shape[1], 'Train and test need same number of features.'
        assert X_train.shape[0] == y_train.shape[0], 'Train X, y need same number of samples.'
        assert X_test.shape[0] == y_test.shape[0], 'Test X, y need same number of samples.'
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._n = X_train.shape[1]

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Get data for searching alternative feature sets

        The data set with :meth:`set_data`, in the same order.

        Returns
        -------
        Tuple
            X_train, X_test, y_train, y_test.
        """

        return self._X_train, self._X_test, self._y_train, self._y_test

    @staticmethod
    def create_solver() -> pywraplp.Solver:
        """Create solver for alternative feature selection

        Return a fresh MILP solver in which the constraints for alternatives will be stored, and
        optionally (depending on the feature-selection method) also the optimization objective.
        Do some initialization that is independent of the feature-selection method, while
        :meth:`initialize_solver is feature-selection-specific. This routine will be called once at
        the beginning of each search for alternative feature sets.

        Returns
        -------
        solver : pywraplp.Solver
            The MILP solver.
        """

        solver = pywraplp.Solver.CreateSolver('SCIP')  # see documentation for available solvers
        solver.SetNumThreads(1)  # we already parallelize experimental pipeline
        return solver

    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        """Initialize solver for alternative feature selection

        Initialize a solver (to find alternative feature sets) specific to the feature-selection
        method, e.g., add an objective (if feature set optimized solver-based) or constraints (if
        the feature-selection method has these). Should be overridden in subclasses. This routine
        will be called once at the beginning of the search for alternative feature sets.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver that will hold the objective and the constraints.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The feature-selection decision variables. For sequential search, `len(s_list) == 1`,
            while simultaneous search considers multiple feature sets at once. For each feature
            set, there should be as many decision variables as there are features in the dataset.
        k : int
            The number of features to be selected.
        objective_agg : str, optional
            How to aggregate the feature sets' qualities in the objective (if there are multiple
            feature sets optimized at once, i.e., in simultaneous search). Not processed here, but
            should be handled appropriately when defining the objective in subclasses and
            distinguish between `"min"` and `"sum"`.
        """

        solver.SetTimeLimit(60000 * len(s_list))  # measured in milliseconds; proportional to ...
        # ... the number of feature sets per solver call (i.e., higher for simultaneous search)
        for s in s_list:
            solver.Add(solver.Sum(s) == k)

    @abstractmethod
    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        """Run alternative feature selection

        Subroutine in the search for alternative feature sets, representing one simultaneous search
        or one iteration of sequential search. Should use the `solver` and the decision variables
        in `s_list` (see :meth:`initialize_solver` for details) and return a summary of the result.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver holding the constraints and (for white-box optimization) the objective.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The decision variables for feature selection. One list per desired feature set.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (specific to the feature-selection method).

        Returns
        -------
        pd.DataFrame
            Table of results, where each row is a feature set (column `selected_idxs`, type
            :class:`Sequence[int]`) accompanied by evaluation metrics (columns: `train_objective`,
            `test_objective`, `optimization_time`, `optimization_status`).
        """

        raise NotImplementedError('Abstract method.')

    @staticmethod
    def create_product_var(solver: pywraplp.Solver, var1: pywraplp.Variable,
                           var2: pywraplp.Variable) -> pywraplp.Variable:
        """Create product variable

        Linearize the product between two binary decision variables by introducing a new variable
        and some constraints.

        Parameters
        ----------
        solver : pywraplp.Solver
            The MILP solver holding the variables and constraints.
        var1 : pywraplp.Variable
            The first binary decision variable.
        var2 : pywraplp.Variable
            The second binary decision variable.

        Returns
        -------
        var : pywraplp.Variable
            The auxiliary binary decision variable. Constraints are added to `solver` in-place.

        Literature
        ----------
        https://docs.mosek.com/modeling-cookbook/mio.html#boolean-operators ("AND" operator)
        """

        var_name = var1.name() + '*' + var2.name()
        var = solver.BoolVar(name=var_name)
        solver.Add(var <= var1)
        solver.Add(var <= var2)
        solver.Add(1 + var >= var1 + var2)
        return var

    @staticmethod
    def create_pairwise_alternative_constraint(
            solver: pywraplp.Solver, s1: Sequence[Union[pywraplp.Variable, int, bool]],
            s2: Sequence[pywraplp.Variable], k: int, tau: Optional[float] = None,
            tau_abs: Optional[int] = None, d_name: str = 'dice') -> pywraplp.LinearConstraint:
        """Create constraint for pairwise alternative feature sets

        Return a constraint such that the dissimilarity between two feature sets of desired size
        `k` is over a threshold `tau`.

        Parameters
        ----------
        solver : pywraplp.Solver
            The MILP solver used for formulating expressions and (if both feature sets are
            variable) additional linearization constraints that are added in-place.
        s1 : Sequence[Union[pywraplp.Variable, int, bool]]
            Selection decisions for the first feature set, which can be either unknown (variables)
            or known (vector of 0/1 or false/true).
        s2 : Sequence[pywraplp.Variable]
            Selection decisions for the second feature set, which are always unknown (variables).
        k : int
            The number of features to be selected.
        tau : Optional[float], optional
            The dissimilarity threshold between feature sets as number in [0, 1].
        tau_abs : Optional[int], optional
            For Dice dissimilarity, the dissimilarity threshold as absolute number of features
            (`tau * k`). You should only set either `tau` or `tau_abs`.
        d_name : str, optional
            The set-dissimilarity measure, either `"dice"` or `"jaccard"`.

        Raises
        ------
        ValueError
            Unknown set-dissimilarity measure.

        Returns
        -------
        pywraplp.LinearConstraint
            The dissimilarity constraint (further auxiliary constraints may be added automatically
            to `solver` if necessary).
        """

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

    def search_sequentially(self, k: int, num_alternatives: int, tau: Optional[float] = None,
                            tau_abs: Optional[int] = None, d_name: str = 'dice',
                            objective_agg: Optional[str] = None) -> pd.DataFrame:
        """Sequential search for alternative feature sets

        Sequentially search for alternative feature sets, iteratively adding constraints and
        calling a feature-selection method (:meth:`select_and_evaluate`) that needs to be
        implemented in a subclass.

        Parameters
        ----------
        k : int
            The number of features to be selected.
        num_alternatives : int
            The number of alternatives `a`, which is the number of feature sets + 1 (the first set
            is the "original").
        tau : Optional[float], optional
            The dissimilarity threshold between feature sets as number in [0, 1].
        tau_abs : Optional[int], optional
            For Dice dissimilarity, the dissimilarity threshold as absolute number of features
            (`tau * k`). You should only set either `tau` or `tau_abs`.
        d_name : str, optional
            The set-dissimilarity measure, either `"dice"` or `"jaccard"`.
        objective_agg : Optional[str], optional
            How to aggregate the feature sets' qualities in the objective. Parameter only exists
            for consistency to simultaneous search but is irrelevant here (as we only find one
            feature set per iteration, we do not need to aggregate quality over feature sets).

        Returns
        -------
        pd.DataFrame
            Table of results (see :meth:`select_and_evaluate`).
        """

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

    def search_simultaneously(self, k: int, num_alternatives: int, tau: Optional[float] = None,
                              tau_abs: Optional[int] = None, d_name: str = 'dice',
                              objective_agg: str = 'sum') -> pd.DataFrame:
        """Simultaneous search for alternative feature sets

        Simultaneously search for alternative feature sets, only generating constraints once and
        then calling a feature-selection method (:meth:`select_and_evaluate`) that needs to be
        implemented in a subclass.

        Parameters
        ----------
        k : int
            The number of features to be selected.
        num_alternatives : int
            The number of alternatives `a`, which is the number of feature sets + 1 (one set is
            deemed the "original").
        tau : Optional[float], optional
            The dissimilarity threshold between feature sets as number in [0, 1].
        tau_abs : Optional[int], optional
            For Dice dissimilarity, the dissimilarity threshold as absolute number of features
            (`tau * k`). You should only set either `tau` or `tau_abs`.
        d_name : str, optional
            The set-dissimilarity measure, either `"dice"` or `"jaccard"`.
        objective_agg : Optional[str], optional
            How to aggregate the feature sets' qualities in the objective, which can be `"min"`
            or `"sum"`.

        Returns
        -------
        pd.DataFrame
            Table of results (see :meth:`select_and_evaluate`).
        """

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
    """Alternative feature selection as white-box problem

    (Abstract) white-box feature-selection method, i.e., whose optimization objective (feature-set
    quality) is formulated in the solver and optimized with the latter (under the constraints for
    alternatives). Instead of :meth:`select_and_evaluate`, subclasses need to define the method
    :meth:`create_objectives`, which is specific to the feature-selection method.
    """

    def __init__(self):
        """Initialize alternative feature selection

        In addition to the fields defined by the superclass, store the objective functions for the
        training set and test set.
        """

        super().__init__()
        self._Q_train_list = None  # Sequence[pywraplp.LinearExpr]; objectives for the feature sets
        self._Q_test_list = None  # Sequence[pywraplp.LinearExpr]; for evaluation, not optimization

    @abstractmethod
    def create_objectives(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]], k: int) \
            -> Tuple[Sequence[pywraplp.LinearExpr], Sequence[pywraplp.LinearExpr]]:
        """Create objectives for alternative feature selection

        Should return expressions for train objective and test objective (one expression per
        desired feature set, i.e., element from `s_list`), using the decision variables from
        `s_list`, data stored in `self`, and potentially the number of features `k`. Might add
        auxiliary variables and constraints to the `solver` in-place.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver holding the constraints and the objective (though you should not add the
            latter to the `solver` here; this will be done automatically later).
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The decision variables for feature selection. One list per desired feature set.
        k : int
            The number of features to be selected.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (specific to the feature-selection method).

        Returns
        -------
        Sequence[pywraplp.LinearExpr]
            Objectives (feature-set quality) on the training set, one per feature set.
        Sequence[pywraplp.LinearExpr]
            Objectives (feature-set quality) on the test set, one per feature set.
        """

        raise NotImplementedError('Abstract method.')

    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        """Initialize solver for white-box alternative feature selection

        Initialize the solver (to find alternative feature sets) by creating an expression of the
        white-box objective for each feature set in `s_list` and aggregating this with
        `objective_agg` to an overall objective.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver that will hold the objective and the constraints.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The feature-selection decision variables. For sequential search, `len(s_list) == 1`,
            while simultaneous search considers multiple feature sets at once. For each feature
            set, there should be as many decision variables as there are features in the dataset.
        k : int
            The number of features to be selected.
        objective_agg : str, optional
            How to aggregate the feature sets' qualities in the objective (only matters in
            simultaneous search, with options being `"min"` and `"sum"`).
        """

        super().initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        self._Q_train_list, self._Q_test_list = self.create_objectives(
            solver=solver, s_list=s_list, k=k)
        if objective_agg == 'sum':
            objective = solver.Sum(self._Q_train_list)  # sum over all feature sets
        elif objective_agg == 'min':  # additional linearization
            Q_min = solver.NumVar(name='Q_min', lb=float('-inf'), ub=float('inf'))
            for i in range(len(s_list)):
                solver.Add(Q_min <= self._Q_train_list[i])
            objective = Q_min
        else:
            raise ValueError('Unknown objective aggregation.')
        solver.Maximize(objective)

    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        """Run alternative white-box feature selection

        Subroutine in the search for alternative feature sets, representing one simultaneous search
        or one iteration of sequential search. Uses the `solver` and the decision variables in
        `s_list` (see :meth:`initialize_solver` for details) for white-box optimization and returns
        a summary of the result.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver holding the constraints and the optimization objective.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The decision variables for feature selection. One list per desired feature set.

        Returns
        -------
        pd.DataFrame
            Table of results, where each row is a feature set (column `selected_idxs`, type
            :class:`Sequence[int]`) accompanied by evaluation metrics (columns: `train_objective`,
            `test_objective`, `optimization_time`, `optimization_status`).
        """

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
    """Alternative feature selection with linear quality function

    (Abstract) white-box feature-selection method whose objective function (feature-set quality) is
    the sum of the individual features' qualities. This allows pre-computing the qualities when the
    data is set and re-using these qualities for multiple selection/alternative-search runs.
    Subclasses need to implement :meth:`compute_qualities` instead of :meth:`create_objectives`
    from the superclass.

    Adding to the solver-based optimization methods :meth:`search_sequentially` and
    :meth:`search_simultaneously` from the superclass, this class provides two heuristic search
    methods (:meth:`search_greedy_balancing` and :meth:`search_greedy_replacement`).
    """

    def __init__(self):
        """Initialize alternative feature selection

        In addition to the fields defined by the superclass, store the univariate feature qualities
        for the training set and test set.
        """

        super().__init__()
        self._q_train = None  # Iterable[float]; qualities of the individual features
        self._q_test = None  # Iterable[float]; for evaluation only, not optimization

    @abstractmethod
    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> Iterable[float]:
        """Compute univariate feature qualities

        Compute one numeric quality value for each feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature values of the dataset. Each row is a data object, each column a feature.
        y : pd.Series
            Prediction target. Must have the same number of entries as `X` has rows.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (specific to the feature-selection method).

        Returns
        -------
        Iterable[float]
            One quality per feature, i.e., as many qualities as `X` has columns.
        """

        raise NotImplementedError('Abstract method.')

    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        """Set data for searching alternative feature sets

        Set the data for the search for alternative feature sets and pre-compute the features'
        qualities (:meth:`compute_qualities`) for the objective, so they can be used in multiple
        search runs for alternatives.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature values for the training part of the dataset. Each row is a data object, each
            column a feature.
        X_test : pd.DataFrame
            Feature values for the test part of the dataset. Must have the same number of columns
            (features) as `X_train`.
        y_train : pd.Series
            Training part of the prediction target. Must have the same number of entries as
            `X_train` has rows.
        y_test : pd.Series
            Test part of the prediction target. Must have the same number of entries as `X_test`
            has rows.
        """

        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._q_train = self.compute_qualities(X=X_train, y=y_train)
        self._q_test = self.compute_qualities(X=X_test, y=y_test)

    def create_objectives(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]], k: int) \
            -> Tuple[Sequence[pywraplp.LinearExpr], Sequence[pywraplp.LinearExpr]]:
        """Create linear objectives for alternative feature selection

        Formulate training-set and test-set objective (feature-set quality) for each feature set as
        the sum of the selected features' qualities.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver holding the constraints and the objective.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The decision variables for feature selection. One list per desired feature set.
        k : int
            The number of features to be selected.

        Returns
        -------
        Sequence[pywraplp.LinearExpr]
            Objectives (feature-set quality) on the training set, one per feature set.
        Sequence[pywraplp.LinearExpr]
            Objectives (feature-set quality) on the test set, one per feature set.
        """

        return ([solver.Sum([q_j * s_j for (q_j, s_j) in zip(self._q_train, s)]) for s in s_list],
                [solver.Sum([q_j * s_j for (q_j, s_j) in zip(self._q_test, s)]) for s in s_list])

    def search_greedy_replacement(self, k: int, num_alternatives: int,
                                  tau: Optional[float] = None, tau_abs: Optional[int] = None,
                                  objective_agg: Optional[str] = None) -> pd.DataFrame:
        """'Greedy Replacement' search for alternative feature sets

        A heuristic search method for alternative feature sets with univariate feature qualities.
        Selects the top `(1 - tau) * k` features in each alternative and sequentially replaces the
        remaining features from alternative to alternative, iterating over features by decreasing
        quality. Works without a solver. Only supports the Dice dissimilarity and cannot handle
        additional constraints on feature sets (is tailored a specific constraint type).

        Parameters
        ----------
        k : int
            The number of features to be selected.
        num_alternatives : int
            The number of alternatives `a`, which is the number of feature sets + 1 (the first set
            is the "original").
        tau : Optional[float], optional
            The dissimilarity threshold between feature sets as number in [0, 1].
        tau_abs : Optional[int], optional
            The dissimilarity threshold as absolute number of features (`tau * k`). You should only
            set either `tau` or `tau_abs`.
        objective_agg : Optional[str], optional
            How to aggregate the feature sets' qualities in the objective. Parameter only exists
            for consistency to exact simultaneous search but is irrelevant here (as we only find
            one feature set per iteration, we do not need to aggregate quality over feature sets).

        Returns
        -------
        pd.DataFrame
            Table of results (see :meth:`select_and_evaluate`).
        """

        if tau is not None:
            tau_abs = math.ceil(tau * k)
        start_time = time.process_time()
        s_list = []
        indices = np.argsort(self._q_train)[::-1]  # descending order by qualities
        s = [0] * self._n  # initial selection for all alternatives
        feature_position = 0  # index of index of currently selected feature
        while feature_position < k - tau_abs:  # "(1-tau) * k" features same in all alternatives
            j = indices[feature_position]
            s[j] = 1
            feature_position = feature_position + 1
        i = 0  # number of current alternative
        while i <= num_alternatives and i <= ((self._n - k) / tau_abs):
            s_i = s.copy()  # select best "k - tau_abs" features
            for _ in range(tau_abs):  # select remaining "tau_abs" features
                j = indices[feature_position]
                s_i[j] = 1
                feature_position = feature_position + 1
            s_list.append(s_i)
            i = i + 1
        end_time = time.process_time()  # result-preparation time also not measured in exact search
        # Transform into result structure consistent to the other searches in the top-level class:
        results = [{
            'selected_idxs': [j for (j, s_j) in enumerate(s) if s_j],
            'train_objective': sum(q_j * s_j for (q_j, s_j) in zip(self._q_train, s)),
            'test_objective': sum(q_j * s_j for (q_j, s_j) in zip(self._q_test, s)),
            'optimization_status': pywraplp.Solver.FEASIBLE  # heuristic -> potentially suboptimal
        } for s in s_list]
        for i in range(i, num_alternatives + 1):  # in case algorithm ran out of features early
            results.append({
                'selected_idxs': [],
                'train_objective': float('nan'),
                'test_objective': float('nan'),
                'optimization_status': pywraplp.Solver.NOT_SOLVED  # heuristic: solution may exist
            })
        results = pd.DataFrame(results)
        results['optimization_time'] = end_time - start_time
        return results

    def search_greedy_balancing(self, k: int, num_alternatives: int,
                                tau: Optional[float] = None, tau_abs: Optional[int] = None,
                                objective_agg: Optional[str] = None) -> pd.DataFrame:
        """'Greedy Balancing' search for alternative feature sets

        A heuristic search method for alternative feature sets with univariate feature qualities.
        Selects the top `(1 - tau) * k` features in each alternative and employs a
        Longest-Processing-Time-first (LPT) heuristic for the remaining features (i.e., sort
        features decreasingly by quality and always add to the alternative with the currently
        lowest quality). Works without a solver. Only supports the Dice dissimilarity and cannot
        handle additional constraints on feature sets (is tailored a specific constraint type).

        Parameters
        ----------
        k : int
            The number of features to be selected.
        num_alternatives : int
            The number of alternatives `a`, which is the number of feature sets + 1 (the first set
            is the "original").
        tau : Optional[float], optional
            The dissimilarity threshold between feature sets as number in [0, 1].
        tau_abs : Optional[int], optional
            The dissimilarity threshold as absolute number of features (`tau * k`). You should only
            set either `tau` or `tau_abs`.
        objective_agg : Optional[str], optional
            How to aggregate the feature sets' qualities in the objective. Parameter only exists
            for consistency to exact simultaneous search but is irrelevant here (as we only find
            one feature set per iteration, we do not need to aggregate quality over feature sets).

        Returns
        -------
        pd.DataFrame
            Table of results (see :meth:`select_and_evaluate`).
        """

        if tau is not None:
            tau_abs = math.ceil(tau * k)
        if tau_abs * num_alternatives + k > self._n:  # not enough features for alternatives
            return pd.DataFrame([{
                'selected_idxs': [],
                'train_objective': float('nan'),
                'test_objective': float('nan'),
                'optimization_status': pywraplp.Solver.NOT_SOLVED,  # heuristic: solution may exist
                'optimization_time': 0  # not worth to measure it
            }] * (num_alternatives + 1))  # all alternatives in table/list identical (NA values)
        start_time = time.process_time()
        s_list = [[0] * self._n for _ in range(num_alternatives + 1)]  # initial selection
        indices = np.argsort(self._q_train)[::-1]  # descending order by qualities
        feature_position = 0  # index of index of currently selected feature
        while feature_position < k - tau_abs:  # "(1-tau) * k" features same in all alternatives
            j = indices[feature_position]
            for i in range(num_alternatives + 1):
                s_list[i][j] = 1
            feature_position = feature_position + 1
        Q_list = [0] * (num_alternatives + 1)  # same relative quality of all sets (same features)
        while feature_position < tau_abs * num_alternatives + k:  # LPT heuristic for the rest
            Q_min = float('inf')
            i_min = -1
            for i in range(num_alternatives + 1):  # find lowest-quality set with space remaining
                if (Q_list[i] < Q_min) and (sum(s_list[i]) < k):
                    Q_min = Q_list[i]
                    i_min = i
            j = indices[feature_position]
            s_list[i_min][j] = 1
            Q_list[i_min] = Q_list[i_min] + self._q_train[j]
            feature_position = feature_position + 1
        end_time = time.process_time()  # result-preparation time also not measured in exact search
        # Transform into result structure consistent to the other searches in the top-level class:
        return pd.DataFrame([{
            'selected_idxs': [j for (j, s_j) in enumerate(s) if s_j],
            'train_objective': sum(q_j * s_j for (q_j, s_j) in zip(self._q_train, s)),
            'test_objective': sum(q_j * s_j for (q_j, s_j) in zip(self._q_test, s)),
            'optimization_status': pywraplp.Solver.FEASIBLE,  # heuristic -> potentially suboptimal
            'optimization_time': end_time - start_time
        } for s in s_list])


class ManualQualityUnivariateSelector(LinearQualityFeatureSelector):
    """Alternative feature selection with manually-defined univariate qualities

    Univariate filter feature-selection method (linear objective summing the selected features'
    qualities) whose feature qualities are directly provided rather than computing them from a
    dataset. Useful to inspect examples for a theoretical analysis of the univariate objective.
    """

    def set_data(self, q_train: Sequence[float], q_test: Optional[Sequence[float]] = None) -> None:
        """Set data for searching alternative feature sets

        Set the data for the search for alternative feature sets. Unlike the superclass'
        implementation, this method directly takes the given univariate feature qualities rather
        computing them from a given dataset.

        Parameters
        ----------
        q_train : Sequence[float]
            Feature qualities on the training set. One quality per feature.
        q_test : Optional[Sequence[float]], optional
            Feature qualities on the test set. If not provided, take training-set ones.
        """

        if q_test is None:
            q_test = q_train
        assert len(q_train) == len(q_test), 'Train and test need same number of features.'
        self._q_train = q_train
        self._q_test = q_test
        self._n = len(q_train)

    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> Iterable[float]:
        """Compute univariate feature qualities

        This inherited method is irrelevant for the current class, since feature qualities should
        be directly provided in :meth:`set_data` rather than computed. We only implement the method
        to make it non-abstract, but it should not be used (slightly dubious software design).

        Raises
        ------
        TypeError
            Always raised since method should not be called.
        """

        raise TypeError('Qualities are directly set in set_data(), computation unnecessary.')


class MISelector(LinearQualityFeatureSelector):
    """Alternative feature selection with mutual information

    Univariate filter feature-selection method that uses the mutual information between each
    feature and the prediction target in the linear objective function.
    """

    @staticmethod
    def mutual_info(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute mutual information

        Determine the mutual information between each feature and the prediction target in a
        deterministic manner (i.e., with a fixed random seed). Consistently use a regression
        estimator, as not only `X`, but also `y` could be continuous (e.g., for regression problems
        or if we call this function to consider feature-feature dependencies).

        Parameters
        ----------
        X : pd.DataFrame
            Feature values of the dataset. Each row is a data object, each column a feature.
        y : pd.Series
            Prediction target. Must have the same number of entries as `X` has rows.

        Returns
        -------
        np.ndarray
            One quality per feature, i.e., as many entries as `X` has columns.
        """

        return sklearn.feature_selection.mutual_info_regression(X=X, y=y, random_state=25)

    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute univariate feature qualities

        Compute the quality of each feature as the mutual information between it and the prediction
        target. Normalize such that selecting all features yields a feature-set quality of 1.

        Parameters
        ----------
        X : pd.DataFrame
            Feature values of the dataset. Each row is a data object, each column a feature.
        y : pd.Series
            Prediction target. Must have the same number of entries as `X` has rows.

        Returns
        -------
        np.ndarray
            One quality per feature, i.e., as many qualities as `X` has columns.
        """

        qualities = MISelector.mutual_info(X=X, y=y)
        qualities = qualities / qualities.sum()
        return qualities


class FCBFSelector(MISelector):
    """Alternative feature selection with Fast Correlation-Based Filter (FCBF)

    Multivariate filter feature-selection method that uses the mutual information between each
    feature and the prediction target in a linear objective function. Additionally, there are
    constraints on the mutual information between features (for each selected feature, dependency
    to target should be > than to each other selected feature).

    Inspired by Yu et al. (2003), which searches heuristically rather than optimizing exactly and
    has a quality threshold on features instead of the linear objective.

    Literature
    ----------
    Yu et al. (2003): "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter
    Solution"
    """

    def __init__(self):
        """Initialize alternative feature selection

        In addition to the fields defined by the superclass, store the unnormalized feature-target
        and feature-feature dependency values (for the additional redundancy constraint).
        """

        super().__init__()
        self._mi_target = None  # np.ndarray; dependency of each feature to target
        self._mi_features = None  # Sequence[np.ndarray]; dependency of each feature to each other

    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        """Set data for searching alternative feature sets

        Set the data for the search for alternative feature sets and pre-compute the features'
        qualities (:meth:`compute_qualities`) for the objective as well as the feature-feature
        dependencies for the additional redundancy constraints, so these values can be used in
        multiple search runs for alternatives.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature values for the training part of the dataset. Each row is a data object, each
            column a feature.
        X_test : pd.DataFrame
            Feature values for the test part of the dataset. Must have the same number of columns
            (features) as `X_train`.
        y_train : pd.Series
            Training part of the prediction target. Must have the same number of entries as
            `X_train` has rows.
        y_test : pd.Series
            Test part of the prediction target. Must have the same number of entries as `X_test`
            has rows.
        """

        super().set_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        # We already have the feature-target MI in "_q_train", but normalized, so compute again:
        self._mi_target = MISelector.mutual_info(X=self._X_train, y=self._y_train)
        self._mi_features = [MISelector.mutual_info(X=self._X_train, y=self._X_train[feature])
                             for feature in self._X_train.columns]

    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        """Initialize solver for FCBF feature selection

        Initialize the solver (to find alternative feature sets) by creating expressions for the
        linear objective and the constraints on feature redundancy.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver that will hold the objective and the constraints.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The feature-selection decision variables. For sequential search, `len(s_list) == 1`,
            while simultaneous search considers multiple feature sets at once. For each feature
            set, there should be as many decision variables as there are features in the dataset.
        k : int
            The number of features to be selected.
        objective_agg : str, optional
            How to aggregate the feature sets' qualities in the objective (only matters in
            simultaneous search, with options being `"min"` and `"sum"`).
        """

        super().initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        # Note that the MI estimator in sklearn is not perfectly bivariate and symmetric, as it
        # uses *one* random-number generator to *iteratively* add some noise to *all* features and
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
    """Alternative feature selection with post-hoc importance

    Use post-hoc feature-importance scores from a trained prediction model as univariate feature
    qualities in a linear objective function.
    """

    def __init__(self, prediction_model: Optional[sklearn.base.BaseEstimator] = None):
        """Initialize alternative feature selection

        In addition to the fields defined by the superclass, store the prediction model whose
        importance score will be extracted.

        Parameters
        ----------
        prediction_model : Optional[sklearn.base.BaseEstimator], optional
            The prediction model for computing feature qualities. If none provided, create a
            decision tree as default one. A user-provided model should have an attribute
            `feature_importances_` after fitting (though the model passed here does not need to be
            fit yet, though, only initialized).
        """

        super().__init__()
        if prediction_model is None:
            prediction_model = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                                   random_state=25)
        self._prediction_model = prediction_model

    def compute_qualities(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute univariate feature qualities

        Compute the quality of each feature as its importance in a trained prediction model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature values of the dataset. Each row is a data object, each column a feature.
        y : pd.Series
            Prediction target. Must have the same number of entries as `X` has rows.

        Returns
        -------
        np.ndarray
            One quality per feature, i.e., as many qualities as `X` has columns.
        """

        return self._prediction_model.fit(X=X, y=y).feature_importances_


class MRMRSelector(WhiteBoxFeatureSelector):
    """Alternative feature selection with Minimal Redundancy Maximal Relevance (mRMR)

    Multivariate filter feature-selection method that uses

    1) mutual information between selected features and prediction target as relevance criterion
    2) mutual information between selected features as redundancy criterion

    mRMR compute the difference between these two quantities.

    Literature
    ----------
    Peng et al. (2005): "Feature Selection Based on Mutual Information: Criteria of Max-Dependency,
    Max-Relevance, and Min-Redundancy".
    """

    # Initialize all fields.
    def __init__(self):
        """Initialize alternative feature selection

        In addition to the fields defined by the superclass, store the feature-target and
        feature-feature dependency values (for computing relevance and redundancy).
        """

        super().__init__()
        self._mi_target_train = None  # np.ndarray; feature-target dependencies
        self._mi_target_test = None  # np.ndarray; feature-target dependencies
        self._mi_features_train = None  # Sequence[np.ndarray]; feature-feature dependencies
        self._mi_features_test = None  # Sequence[np.ndarray]; feature-feature dependencies

    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 y_test: pd.Series) -> None:
        """Set data for searching alternative feature sets

        Set the data for the search for alternative feature sets and pre-compute the feature-target
        and feature-feature dependency values for the objective, so these values can be used in
        multiple search runs for alternatives. We use mutual information as dependency measure and
        normalize the resulting values such that relevance and redundancy cannot exceed 1 (another
        normalization, inherent in mRMR, is dividing the relevance and redundancy values by the
        number of selected features in the objective).

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature values for the training part of the dataset. Each row is a data object, each
            column a feature.
        X_test : pd.DataFrame
            Feature values for the test part of the dataset. Must have the same number of columns
            (features) as `X_train`.
        y_train : pd.Series
            Training part of the prediction target. Must have the same number of entries as
            `X_train` has rows.
        y_test : pd.Series
            Test part of the prediction target. Must have the same number of entries as `X_test`
            has rows.
        """

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

    def create_objectives(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]], k: int) \
            -> Tuple[Sequence[pywraplp.LinearExpr], Sequence[pywraplp.LinearExpr]]:
        """Create mRMR objectives for alternative feature selection

        Formulate training-set and test-set objective (feature-set quality) for each feature set by
        considering feature relevance and redundancy. Due to a special encoding of interaction
        terms (using real variables specific to training-set feature qualities instead of binary
        interaction variables), we cannot give a closed-form expression for the test objective
        here, but return the train objective twice and calculate the actual test objective via
        :meth:`compute_test_objective` called in :meth:`select_and_evaluate`.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver holding the constraints and the objective.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The decision variables for feature selection. One list per desired feature set.
        k : int
            The number of features to be selected.

        Returns
        -------
        Sequence[pywraplp.LinearExpr]
            Objectives (feature-set quality) on the training set, one per feature set.
        Sequence[pywraplp.LinearExpr]
            Also training-set objective, see explanation above.

        Literature
        ----------
        Nguyen et al. (2010): "Towards a Generic Feature-Selection Measure for Intrusion Detection"
        """

        objectives = []
        for i, s in enumerate(s_list):
            relevance = solver.Sum([q_j * s_j for (q_j, s_j) in zip(self._mi_target_train, s)])
            relevance = relevance / k
            redundancy_terms = []
            M = max(sum(x) for x in self._mi_features_train) + 1  # can "deactivate" constraints
            for j_1 in range(len(s)):
                # Linearization: z_i = x_i * A_i(x) (follows Equation (14) in Nguyen et al. (2010)
                # "Towards a Generic Feature-Selection Measure for Intrusion Detection" except
                # there is no variable "y" since we do not have a fractional expression (we only
                # divide by constants); we have max instead min objective, but since the redundancy
                # term is subtracted (should be minimized), we can still use Equation (14))
                z_j = solver.NumVar(name=f'z_{i}_{j_1}', lb=0, ub=M)
                A_j = solver.Sum([self._mi_features_train[j_1][j_2] * s[j_2]
                                  for j_2 in range(len(s))])
                solver.Add(M * (s[j_1] - 1) + A_j <= z_j)
                redundancy_terms.append(z_j)
            redundancy = solver.Sum(redundancy_terms) / (k * (k - 1))  # -1 as self-redundancy == 0
            objectives.append(relevance - redundancy)
        return (objectives, objectives)  # no closed-form expression for test objective

    def compute_test_objective(self, selected_idxs: Sequence[int]) -> float:
        """Compute mRMR test-set objective

        Compute the feature-set quality according to mRMR on the internally stored test set with
        a given feature selection.

        Parameters
        ----------
        selected_idxs : Sequence[int]
            Liste of selected features (indices).

        Returns
        -------
        float
            mRMR feature-set quality on the test set.
        """

        k = len(selected_idxs)
        if k == 0:
            return float('nan')
        relevance = sum(self._mi_target_test[j] for j in selected_idxs) / k
        redundancy = sum(self._mi_features_test[j_1][j_2] for j_1 in selected_idxs
                         for j_2 in selected_idxs) / (k * (k - 1))  # -1 since self-redundancy == 0
        return relevance - redundancy

    # Due to the reason described in "create_objectives()", we need to compute the test objectives
    # manually, but no difference for the user.
    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        result = super().select_and_evaluate(solver=solver, s_list=s_list)
        result['test_objective'] = [self.compute_test_objective(selected_idxs=selected_idxs)
                                    for selected_idxs in result['selected_idxs']]
        return result


class GreedyWrapperSelector(AlternativeFeatureSelector):
    """Alternative feature selection with 'Greedy Wrapper' algorithm

    Simple hill-climbing approach to select features. Feature-set quality is evaluated with a
    prediction model (using Matthews Correlation Coefficient as the metric, so this method is
    limited to classification tasks), while the solver only searches for valid solution candidates
    (one new feature set in sequential search, multiple ones in simultaneous search) instead of
    optimizing quality. In particular, we start with a random valid solution candidate. Given the
    currently best solution candidate, we 'swap' (flip) two feature-selection decisions and let the
    solver find the closest (according to Hamming distance on the feature-selection decisions)
    valid solution candidate under these additional constraints. A better solution candidate
    replaces the previous one. The procedure runs for a fixed number of iterations or till no swap
    improves feature-set quality.
    """

    def __init__(self, prediction_model: Optional[sklearn.base.BaseEstimator] = None,
                 max_iters: int = 1000):
        """Initialize alternative feature selection

        In addition to the fields defined by the superclass, store the maximum iteration count and
        the prediction model we use to evaluate feature-set quality of new solution candidates.

        Parameters
        ----------
        prediction_model : Optional[sklearn.base.BaseEstimator], optional
            The prediction model for evaluating feature-set quality. If none provided, create a
            decision tree as default one.
        max_iters : int, optional
            The maximum number of calls to the solver in the wrapper search (as the solver might
            not always yield valid feature sets, this is an upper bound on the number of solution
            candidates evaluated).
        """
        super().__init__()
        if prediction_model is None:
            prediction_model = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                                   random_state=25)
        self._prediction_model = prediction_model
        self._max_iters = max_iters
        self._objective_agg = None

    def initialize_solver(self, solver: pywraplp.Solver,
                          s_list: Sequence[Sequence[pywraplp.Variable]],
                          k: int, objective_agg: str = 'sum') -> None:
        """Initialize solver for 'Greedy Wrapper' feature selection

        Set a constant objective. The actual quality objective, i.e., prediction performance, is a
        black box for the solver. Instead, the first solver call has no objective at all (just
        finds a valid solution) and all subsequent solver calls dynamically set their objective
        (similarity of the new solution to the currently best one).

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver that will hold the objective and the constraints.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The feature-selection decision variables. For sequential search, `len(s_list) == 1`,
            while simultaneous search considers multiple feature sets at once. For each feature
            set, there should be as many decision variables as there are features in the dataset.
        k : int
            The number of features to be selected.
        objective_agg : str, optional
            How to aggregate the feature sets' qualities in the objective (only matters in
            simultaneous search, with options being `"min"` and `"sum"`).
        """

        super().initialize_solver(solver=solver, s_list=s_list, k=k, objective_agg=objective_agg)
        self._objective_agg = objective_agg
        objective = 0
        solver.Maximize(objective)

    def has_objective_improved(self, old_Q_list: Iterable[float],
                               new_Q_list: Iterable[float]) -> bool:
        """Has wrapper objective improved?

        Checks whether the overall feature-set quality, i.e., aggregated prediction performance
        over alternatives (for sequential search, just one), has improved from the previously best
        solution candidate to the current solution candidate. The prediction performance of
        individual feature sets is computed somewhere else, while this method only aggregates over
        feature sets (the aggregation method is stored in the field :attr:`_objective_agg`).

        Parameters
        ----------
        old_Q_list : Iterable[float]
            Qualities of the feature sets in the previously best solution candidate.
        new_Q_list : Iterable[float]
            Qualities of the feature sets in the new/current solution candidate.

        Raises
        ------
        ValueError
            Unknown aggregation method (neither `"min"` nor `"sum"`).

        Returns
        -------
        bool
            Is the new aggregated feature-set quality higher than the old one?
        """

        if self._objective_agg == 'sum':
            return sum(new_Q_list) > sum(old_Q_list)
        if self._objective_agg == 'min':
            return min(new_Q_list) > min(old_Q_list)
        raise ValueError('Unknown objective aggregation.')

    def evaluate_wrapper(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate feature-set quality in wrapper

        Given a dataset with features selected already, evaluate feature-set quality with the
        wrapped prediction model (:attr:`_prediction_model`) and a stratified holdout split on the
        passed data. Return the prediction performance on the validation set of the split.

        Parameters
        ----------
        X : pd.DataFrame
            Feature values of the dataset. Each row is a data object, each column a feature.
        y : pd.Series
            Prediction target. Must have the same number of entries as `X` has rows.

        Returns
        -------
        float
            Feature-set quality in terms of prediction performance.
        """

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=y, random_state=25)
        if len(X_train.columns) == 0:  # no features selected (no valid solution found)
            return float('nan')
        self._prediction_model.fit(X=X_train, y=y_train)
        pred_test = self._prediction_model.predict(X=X_test)
        return sklearn.metrics.matthews_corrcoef(y_true=y_test, y_pred=pred_test)

    def select_and_evaluate(self, solver: pywraplp.Solver,
                            s_list: Sequence[Sequence[pywraplp.Variable]]) -> pd.DataFrame:
        """Run alternative 'Greedy Wrapper' feature selection

        Subroutine in the search for alternative feature sets, representing one simultaneous search
        or one iteration of sequential search. Uses the `solver` and the decision variables in
        `s_list` (see :meth:`initialize_solver` for details) for finding valid solution candidates
        (close to the currently best solution) and returns a summary of the result. See the class
        description of :class:`GreedyWrapperSelector` for more details on the algorithm itself.

        Parameters
        ----------
        solver : pywraplp.Solver
            The solver holding the constraints for alternatives.
        s_list : Sequence[Sequence[pywraplp.Variable]]
            The decision variables for feature selection. One list per desired feature set.

        Returns
        -------
        pd.DataFrame
            Table of results, where each row is a feature set (column `selected_idxs`, type
            :class:`Sequence[int]`) accompanied by evaluation metrics (columns: `train_objective`,
            `test_objective`, `optimization_time`, `optimization_status`).
        """

        start_time = time.process_time()
        optimization_status = solver.Solve()  # actually only check constraint satisfaction
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
            Q_train_list = [self.evaluate_wrapper(
                X=self._X_train.iloc[:, s_value], y=self._y_train) for s_value in s_value_list]
            j_1 = 0  # other than pseudo-code in paper, 0-indexing here
            j_2 = j_1 + 1
            swap_variables = []
            while (iters < self._max_iters) and (j_1 < self._n - 1):
                # We can't add temporary constraints (no remove function), but we can fix variables
                for s, s_value in zip(s_list, s_value_list):  # fix s_j to inverse of prev. value
                    s[j_1].SetBounds(1 - s_value[j_1], 1 - s_value[j_1])
                    swap_variables.append(s[j_1])
                    s[j_2].SetBounds(1 - s_value[j_2], 1 - s_value[j_2])
                    swap_variables.append(s[j_2])
                objective = solver.Sum([s_j if s_value_j else 1 - s_j
                                        for s, s_value in zip(s_list, s_value_list)
                                        for s_j, s_value_j in zip(s, s_value)])
                solver.Maximize(objective)  # Hamming similarity to previous selection
                optimization_status = solver.Solve()
                iters = iters + 1
                restart_indexing = False
                if optimization_status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                    current_s_value_list = [[bool(s_j.solution_value()) for s_j in s]
                                            for s in s_list]
                    current_Q_train_list = [self.evaluate_wrapper(
                        X=self._X_train.iloc[:, s_value], y=self._y_train)
                        for s_value in current_s_value_list]
                    if self.has_objective_improved(old_Q_list=Q_train_list,
                                                   new_Q_list=current_Q_train_list):
                        s_value_list = current_s_value_list
                        Q_train_list = current_Q_train_list
                        restart_indexing = True
                if restart_indexing:
                    j_1 = 0  # re-start swapping with first feature (zero indexing!) ...
                    j_2 = j_1 + 1  # ... and second feature
                else:
                    if j_2 < self._n - 1:  # "inner loop": only increase index of second feature
                        j_2 = j_2 + 1
                    else:  # "outer loop": increase index of first feature and reset second
                        j_1 = j_1 + 1
                        j_2 = j_1 + 1
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
                # a new prediction model on one part and evaluate the model on the rest; the main
                # experimental pipeline contains the "classic" evaluation with training model on
                # (full) training set and predicting on full train + test
                'test_objective': [self.evaluate_wrapper(
                    X=self._X_test.iloc[:, idxs], y=self._y_test) for idxs in selected_idxs]
            })
        end_time = time.process_time()
        result['optimization_time'] = end_time - start_time
        result['optimization_status'] = optimization_status
        result['wrapper_iters'] = iters
        return result
