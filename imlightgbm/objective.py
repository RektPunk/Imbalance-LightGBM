from copy import deepcopy
from functools import partial
from typing import Any, Callable

import numpy as np
from lightgbm import Dataset
from sklearn.utils.multiclass import type_of_target

from imlightgbm.utils import logger

EvalLike = Callable[[np.ndarray, Dataset], tuple[str, float, bool]]
ObjLike = Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]
ALPHA_DEFAULT: float = 0.05
GAMMA_DEFAULT: float = 0.05
OBJECTIVE_STR: str = "objective"


def binary_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[str, float, bool]:
    is_higher_better = False
    return "binary_focal", ..., is_higher_better


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    # TODO
    return ...


def multiclass_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[str, float, bool]:
    # TODO
    return


def multiclass_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
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


def set_params(
    params: dict[str, Any], train_set: Dataset
) -> tuple[dict[str, Any], EvalLike]:
    _params = deepcopy(params)
    if OBJECTIVE_STR in _params:
        logger.warning(f"'{OBJECTIVE_STR}' exists in params will not used.")
        del _params[OBJECTIVE_STR]

    _alpha = _params.pop("alpha", ALPHA_DEFAULT)
    _gamma = _params.pop("gamma", GAMMA_DEFAULT)

    fobj, feval = set_fobj_feval(train_set=train_set, alpha=_alpha, gamma=_gamma)
    _params.update({OBJECTIVE_STR: fobj})
    return _params, feval
