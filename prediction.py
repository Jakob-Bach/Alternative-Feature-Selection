"""Prediction

Functions for making and evaluating predictions.
"""


from typing import Dict, Generator, Tuple

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree


MODELS = [
    {'name': 'Decision tree', 'func': sklearn.tree.DecisionTreeClassifier,
     'args': {'criterion': 'entropy', 'random_state': 25}},
    {'name': 'Random forest', 'func': sklearn.ensemble.RandomForestClassifier,
     'args': {'n_estimators': 100, 'criterion': 'entropy', 'random_state': 25}}
]


# Split a dataset for the experimental pipeline. Return the split indices.
def split_for_pipeline(X: pd.DataFrame, y: pd.Series) -> Generator[Tuple[np.ndarray, np.ndarray],
                                                                   None, None]:
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=25)
    return splitter.split(X=X, y=y)


# Train and evaluate a model. Return a dictionary with prediction performances.
def train_and_evaluate(
        model: sklearn.base.BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    if len(X_train.columns) == 0:  # no features selected (no valid solution found)
        return {'train_mcc': float('nan'), 'test_mcc': float('nan')}
    model.fit(X=X_train, y=y_train)
    pred_train = model.predict(X=X_train)
    pred_test = model.predict(X=X_test)
    results = {}
    results['train_mcc'] = sklearn.metrics.matthews_corrcoef(y_true=y_train, y_pred=pred_train)
    results['test_mcc'] = sklearn.metrics.matthews_corrcoef(y_true=y_test, y_pred=pred_test)
    return results


# Evaluate a model (for wrapper feature selection) with a stratified holdout split. Return the
# prediction performance on the validation part of the split.
def evaluate_wrapper(model: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=25)
    return train_and_evaluate(model=model, X_train=X_train, y_train=y_train, X_test=X_test,
                              y_test=y_test)['test_mcc']
