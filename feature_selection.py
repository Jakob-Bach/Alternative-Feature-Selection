"""Feature selection

Functions to express feature-set quality for different feature-selection methods.
"""


import time
from typing import Sequence, Tuple

import mip
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.feature_selection

import prediction


# Compute mutual information in a deterministic manner (fixed random state). Consistently use
# regression estimator, as "y" can be continuous (e.g., to consider feature-feature dependencies).
def mutual_info(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    return sklearn.feature_selection.mutual_info_regression(X=X, y=y, random_state=25)


# Compute mutual information as feature-set quality Q(s,X,y) and add it to "mip_model" as objective.
# Used for MI and FCBF as feature selectors. See mi() for details on parameters.
# Returns expressions for train- and test-qualities of each feature set (parts of the objective).
def add_mi_objective(mip_model: mip.Model, s_list: Sequence[Sequence[mip.Var]],
                     X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                     y_test: pd.Series) -> Tuple[Sequence[mip.LinExpr], Sequence[mip.LinExpr]]:
    q_train = mutual_info(X=X_train, y=y_train)
    q_train = q_train / q_train.sum()  # normalize to a sum of 1
    Q_train_list = [mip.xsum(q_j * s_j for (q_j, s_j) in zip(q_train, s)) for s in s_list]
    q_test = mutual_info(X=X_test, y=y_test)
    q_test = q_test / q_test.sum()
    Q_test_list = [mip.xsum(q_j * s_j for (q_j, s_j) in zip(q_test, s)) for s in s_list]
    mip_model.objective = mip.xsum(Q_train_list)  # sum over all feature sets
    return Q_train_list, Q_test_list


# Optimize a problem contained in "mip_model". The decision variables in "s_list" represent
# feature-selection decisions s for one or multiple feature sets (depending on whether we conduct
# sequential or simultaneous search). "Q_train_list" and "Q_test_list" are expressions for the
# feature sets' qualities Q(s,X,y), which are part of the objective in "mip_model" (we need them to
# evaluate individual feature sets rather than just the overall objective).
# Return a table with the selected features and some metrics. Note that the optimization problem
# might have no solution, in which case the returned table contains empty feature sets and NAs.
def optimize_mip(mip_model: mip.Model, s_list: Sequence[Sequence[mip.Var]],
                 Q_train_list: mip.LinExpr, Q_test_list: mip.LinExpr) -> pd.DataFrame:
    assert len(s_list) == len(Q_train_list)
    assert len(Q_train_list) == len(Q_test_list)
    start_time = time.process_time()
    optimization_status = mip_model.optimize()
    end_time = time.process_time()
    if optimization_status == mip.OptimizationStatus.OPTIMAL:
        result = pd.DataFrame({
            'selected_idxs': [[j for (j, s_j) in enumerate(s) if s_j.x] for s in s_list],
            'train_objective': [Q_s.x for Q_s in Q_train_list],
            'test_objective': [Q_s.x for Q_s in Q_test_list],
        })
    else:
        result = pd.DataFrame({
            'selected_idxs': [[] for _ in s_list],
            'train_objective': [float('nan') for _ in Q_train_list],
            'test_objective': [float('nan') for _ in Q_test_list],
        })
    result['optimization_time'] = end_time - start_time
    return result


# Mutual information as univariate filter feature-selector.
# "mip_model" should already contain the decision variables for feature sets that are also in
# "s_list" (for easier assess, so we don't need to query them from the model, which might also
# contain other variables). Also, "mip_model" should already contain constraints on alternatives and
# on the size of the feature sets.
# "X" and "y" are used to compute feature set qualities Q(s,X,y) for the objective. While
# optimization only uses train qualities, evaluation uses test qualities as well.
# Return a table with the selected features and some metrics.
def fs_mi(mip_model: mip.Model, s_list: Sequence[Sequence[mip.Var]], X_train: pd.DataFrame,
          y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    Q_train_list, Q_test_list = add_mi_objective(mip_model=mip_model, s_list=s_list, X_train=X_train,
                                                 y_train=y_train, X_test=X_test, y_test=y_test)
    return optimize_mip(mip_model=mip_model, s_list=s_list, Q_train_list=Q_train_list,
                        Q_test_list=Q_test_list)


# FCBF as multivariate filter feature-selector. Uses mutual information for feature qualities and
# adds constraints on inter-feature dependencies.
# See mi() for details on parameters and return value.
def fs_fcbf(mip_model: mip.Model, s_list: Sequence[Sequence[mip.Var]], X_train: pd.DataFrame,
            y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    Q_train_list, Q_test_list = add_mi_objective(mip_model=mip_model, s_list=s_list, X_train=X_train,
                                                 y_train=y_train, X_test=X_test, y_test=y_test)
    fcbf_constraints = []
    mi_target = mutual_info(X=X_train, y=y_train)
    # Note that MI estimator in sklearn is not perfectly bivariate and symmetric, as it uses *one*
    # random-number generator to *iteratively* add some noise to *all* features and target;
    # e.g., if you re-order features in X, MI estimates change slightly, though the procedure still
    # is deterministic (if seeded, as we do) for the overall (X, y) combination.
    # Here, we only consider each pair of features once.
    for j_1 in range(1, X_train.shape[1]):
        mi_feature = mutual_info(X=X_train.iloc[:, :j_1], y=X_train.iloc[:, j_1])
        for j_2 in range(j_1):
            # If inter-correlation > target correlation, do not select both features:
            if (mi_target[j_1] <= mi_feature[j_2]) or (mi_target[j_2] <= mi_feature[j_2]):
                for s in s_list:
                    fcbf_constraints.append(mip_model.add_constr(s[j_1] + s[j_2] <= 1))
    result = optimize_mip(mip_model=mip_model, s_list=s_list, Q_train_list=Q_train_list,
                          Q_test_list=Q_test_list)
    mip_model.remove(fcbf_constraints)
    return result


# A gredy hill-climbing search as wrapper feature-selector.
# See mi() for details on most parameters and the return value. "model" is the untrained prediction
# model. "max_iters" is the number of evaluated feature sets.
def fs_greedy_wrapper(
        mip_model: mip.Model, s_list: Sequence[Sequence[mip.Var]],
        X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
        prediction_model: sklearn.base.BaseEstimator, max_iters: int = 1000) -> pd.DataFrame:
    start_time = time.process_time()
    mip_model.objective = 0  # don't optimize, only search for valid solutions
    optimization_status = mip_model.optimize()
    iters = 1
    if optimization_status != mip.OptimizationStatus.OPTIMAL:  # no valid solution given constraints
        result = pd.DataFrame({
            'selected_idxs': [[] for _ in s_list],
            'train_objective': [float('nan') for _ in s_list],
            'test_objective': [float('nan') for _ in s_list],
        })
    else:
        s_value_list = [[bool(s_j.x) for s_j in s] for s in s_list]
        # Note that "training" quality actually is validation performance with a holdout split
        Q_train_list = [prediction.evaluate_wrapper(
            model=prediction_model, X=X_train.iloc[:, s_value], y=y_train)
            for s_value in s_value_list]
        j = 0  # other than pseudo-code in paper, 0-indexing here
        while (iters < max_iters) and (j < X_train.shape[1]):
            swap_constraints = [mip_model.add_constr(s[j] == 0 if s_value[j] else s[j] == 1)
                                for (s, s_value) in zip(s_list, s_value_list)]
            optimization_status = mip_model.optimize()
            iters = iters + 1
            if optimization_status == mip.OptimizationStatus.OPTIMAL:
                current_s_value_list = [[bool(s_j.x) for s_j in s] for s in s_list]
                current_Q_train_list = [prediction.evaluate_wrapper(
                    model=prediction_model, X=X_train.iloc[:, s_value], y=y_train)
                    for s_value in current_s_value_list]
                if sum(current_Q_train_list) > sum(Q_train_list):
                    s_value_list = current_s_value_list
                    Q_train_list = current_Q_train_list
                    j = 1
                else:
                    j = j + 1
            else:
                j = j + 1
            mip_model.remove(swap_constraints)
        selected_idxs = [[j for (j, s_j_value) in enumerate(s_value) if s_j_value]
                         for s_value in s_value_list]
        result = pd.DataFrame({
            'selected_idxs': selected_idxs,
            'train_objective': Q_train_list,
            # To be consistent to other FS techniques, we evaluate the test objective by considering
            # test data (to be exact: the validation part of a holdout split of it) only;
            # main experimental pipeline contains "classic" evaluation with training model on full
            # (without holdout split) training set and predicting on full train + full test
            'test_objective': [prediction.evaluate_wrapper(
                model=prediction_model, X=X_test.iloc[:, idxs], y=y_test) for idxs in selected_idxs],
        })
    end_time = time.process_time()
    result['optimization_time'] = end_time - start_time
    result['iters'] = iters
    return result


# Model-based feature importance (for tree-based models: reduction of information gain) as post-hoc
# importance measure, forming a linear objective, like for univariate filter feature-selection.
# See mi() for details on most parameters and the return value. "model" is the untrained prediction
# model.
def fs_model_gain(
        mip_model: mip.Model, s_list: Sequence[Sequence[mip.Var]],
        X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
        prediction_model: sklearn.base.BaseEstimator) -> pd.DataFrame:
    q_train = prediction_model.fit(X=X_train, y=y_train).feature_importances_
    Q_train_list = [mip.xsum(q_j * s_j for (q_j, s_j) in zip(q_train, s)) for s in s_list]
    # Fitting to test set might look fishy, but test importances don't influence feature selection /
    # optimization (we only evaluate optimization objective with test importances retroactively)
    q_test = prediction_model.fit(X=X_test, y=y_test).feature_importances_
    Q_test_list = [mip.xsum(q_j * s_j for (q_j, s_j) in zip(q_test, s)) for s in s_list]
    mip_model.objective = mip.xsum(Q_train_list)
    return optimize_mip(mip_model=mip_model, s_list=s_list, Q_train_list=Q_train_list,
                        Q_test_list=Q_test_list)
