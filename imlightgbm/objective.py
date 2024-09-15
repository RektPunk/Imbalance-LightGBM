from functools import partial
from typing import Callable

import numpy as np
from lightgbm import Dataset
from sklearn.utils.multiclass import type_of_target

EvalLike = Callable[[np.ndarray, Dataset], tuple[str, float, bool]]
ObjLike = Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]


def binary_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    is_higher_better = False
    return "binary_focal", ..., is_higher_better


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    # TODO
    return ...


def multiclass_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    # TODO
    return


def multiclass_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    # TODO
    return


def set_fobj_feval(
    train_set: Dataset, alpha: float, gamma: float
) -> tuple[ObjLike, EvalLike]:
    inferred_task = type_of_target(train_set.get_label())
    if inferred_task not in {"binary", "multiclass"}:
        raise ValueError(
            f"Invalid target type: {inferred_task}. Supported types are 'binary' or 'multiclass'."
        )
    objective_mapper: dict[str, ObjLike] = {
        "binary": partial(binary_focal_objective, alpha=alpha, gamma=gamma),
        "multiclass": partial(multiclass_focal_objective, alpha=alpha, gamma=gamma),
    }
    eval_mapper: dict[str, EvalLike] = {
        "binary": partial(binary_focal_eval, alpha=alpha, gamma=gamma),
        "multiclass": partial(multiclass_focal_eval, alpha=alpha, gamma=gamma),
    }
    fobj = objective_mapper[inferred_task]
    feval = eval_mapper[inferred_task]

    return fobj, feval
