"""Alternatives

Functions to express alternatives and search for them.
"""


import math
from typing import Any, Callable, Dict, Sequence, Union

import mip
import pandas as pd


# Return a variable representing the product between two binary variables from the same model.
# Add the necessary constraints to make sure the new variable really represents the product.
def create_product_var(var1: mip.Var, var2: mip.Var) -> mip.Var:
    assert var1.model is var2.model
    mip_model = var1.model
    var_name = var1.name + '*' + var2.name
    var = mip_model.add_var(name=var_name, var_type=mip.BINARY)
    mip_model.add_constr(var <= var1)
    mip_model.add_constr(var <= var2)
    mip_model.add_constr(1 + var >= var1 + var2)
    return var


# Create and add a constraint such that the Jaccard distance between two feature sets of size "k"
# is over a threshold "tau". The selection decisions "s2" for the 2nd feature set are unknown,
# while the decisions "s1" for the first feature set can be unknown or known.
def add_pairwise_alternative_constraint(
        mip_model: mip.Model, s1: Sequence[Union[mip.Var, int, bool]], s2: Sequence[mip.Var],
        k: int, tau: float) -> None:
    assert len(s1) == len(s2)
    assert isinstance(s2[0], mip.Var)  # one feature set needs to be undetermined
    if isinstance(s1[0], mip.Var):  # both features sets undetermined
        overlap_size = mip.xsum(create_product_var(s1_j, s2_j) for (s1_j, s2_j) in zip(s1, s2))
    else:  # one feature set known
        overlap_size = mip.xsum(s2_j for (s1_j, s2_j) in zip(s1, s2) if s1_j)
    mip_model.add_constr(overlap_size <= (1 - tau) / (2 - tau) * 2 * k)


# Configure the MIP optimizer that we use in the search routines.
def initialize_optimizer() -> mip.Model:
    mip_model = mip.Model(sense=mip.MAXIMIZE)
    mip_model.seed = 25
    mip_model.verbose = 0
    mip_model.threads = 1
    mip_model.emphasis = mip.SearchEmphasis.OPTIMALITY
    mip_model.max_mip_gap = 0  # without this, solutions might be slightly sub-optimal
    return mip_model


# Sequentially search for alternative feature sets. This routine represents the overall procedure
# and generates the constraints, while the actual optimization is delegated to "optimization_func",
# which is called with "optimization_args".
# - "n": original number of features
# - "k": number of features to select
# - "tau": (Jaccard) distance threshold for being alternative
# - "num_alternatives": number of returned feature sets - 1 (first set is considered "original")
# Return a table of results, where each row is a feature set accompanied by some metrics.
def search_sequentially(optimization_func: Callable, optimization_args: Dict[str, Any],
                        n: int, k: int, tau: float, num_alternatives: int) -> pd.DataFrame:
    results = []
    mip_model = initialize_optimizer()
    s = [mip_model.add_var(name=f's_{j}', var_type=mip.BINARY) for j in range(n)]  # selections
    mip_model.add_constr(mip.xsum(s) == k)  # select exactly k
    optimization_args = {'mip_model': mip_model, 's_list': [s], **optimization_args}
    results.append(optimization_func(**optimization_args))  # results for original feature set
    for _ in range(num_alternatives):
        if not math.isnan(results[-1]['train_objective'].iloc[0]):  # if not infeasible
            s_value = [j in results[-1]['selected_idxs'].iloc[0] for j in range(n)]  # last solution
            add_pairwise_alternative_constraint(mip_model=mip_model, s1=s_value, s2=s,
                                                k=k, tau=tau)  # alternative to all prior solutions
        results.append(optimization_func(**optimization_args))
    return pd.concat(results, ignore_index=True)


# Simultaneously search for alternative feature sets. This routine represents the overall procedure
# and generates the constraints, while the actual optimization is delegated to "optimization_func",
# which is called with "optimization_args".
# - "n": original number of features
# - "k": number of features to select
# - "tau": (Jaccard) distance threshold for being alternative
# - "num_alternatives": number of returned feature sets - 1 (one set is considered "original")
# Return a table of results, where each row is a feature set accompanied by some metrics.
def search_simultaneously(optimization_func: Callable, optimization_args: Dict[str, Any],
                          n: int, k: int, tau: float, num_alternatives: int) -> pd.DataFrame:
    mip_model = initialize_optimizer()
    s_list = []
    for i in range(num_alternatives + 1):  # find "num_alternatives" + 1 feature sets
        s = [mip_model.add_var(name=f's{i}_{j}', var_type=mip.BINARY) for j in range(n)]
        mip_model.add_constr(mip.xsum(s) == k)
        for s2 in s_list:
            add_pairwise_alternative_constraint(mip_model, s1=s, s2=s2, k=k, tau=tau)
        s_list.append(s)
    optimization_args = {'mip_model': mip_model, 's_list': s_list, **optimization_args}
    return optimization_func(**optimization_args)
