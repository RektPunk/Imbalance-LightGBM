from copy import deepcopy
from functools import partial
from typing import Any, Callable

import numpy as np
import optuna
from lightgbm import Dataset
from sklearn.utils.multiclass import type_of_target

from imlightgbm.utils import logger

EvalLike = Callable[[np.ndarray, Dataset], tuple[str, float, bool]]
ObjLike = Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]
ALPHA_DEFAULT: float = 0.25
GAMMA_DEFAULT: float = 2.0
OBJECTIVE_STR: str = "objective"
METRIC_STR: str = "metric"
IS_HIGHER_BETTER: bool = False


def _power(num_base: np.ndarray, num_pow: float):
    """Safe power."""
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


def _log(array: np.ndarray, is_prob: bool = False) -> np.ndarray:
    """Safe log."""
    _upper = 1 if is_prob else None
    return np.log(np.clip(array, 1e-6, _upper))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert raw predictions to probabilities in binary task."""
    return 1 / (1 + np.exp(-x))


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary focal objective."""
    label = train_data.get_label()
    pred_prob = _sigmoid(pred)

    # gradient
    g1 = pred_prob * (1 - pred_prob)
    g2 = label + ((-1) ** label) * pred_prob
    g3 = pred_prob + label - 1
    g4 = 1 - label - ((-1) ** label) * pred_prob
    g5 = label + ((-1) ** label) * pred_prob
    grad = gamma * g3 * _power(g2, gamma) * _log(g4) + ((-1) ** label) * _power(
        g5, (gamma + 1)
    )

    # hess
    h1 = _power(g2, gamma) + gamma * ((-1) ** label) * g3 * _power(g2, (gamma - 1))
    h2 = ((-1) ** label) * g3 * _power(g2, gamma) / g4
    hess = ((h1 * _log(g4) - h2) * gamma + (gamma + 1) * _power(g5, gamma)) * g1
    return grad, hess


def binary_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[str, float, bool]:
    """Return binary focal eval."""
    label = train_data.get_label()
    pred_prob = _sigmoid(pred)
    p_t = np.where(label == 1, pred_prob, 1 - pred_prob)
    loss = -alpha * ((1 - p_t) ** gamma) * _log(p_t, True)

    focal_loss = np.mean(loss)
    return "focal", focal_loss, IS_HIGHER_BETTER


def multiclass_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    # TODO
    return


def multiclass_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[str, float, bool]:
    # TODO
    return


def _set_fobj_feval(
    train_set: Dataset, alpha: float, gamma: float
) -> tuple[ObjLike, EvalLike]:
    """Return obj and eval with respect to task type."""
    inferred_task = type_of_target(train_set.get_label())
    if inferred_task not in {"binary", "multiclass"}:
        raise ValueError(
            f"Invalid target type: {inferred_task}. Supported types are 'binary' or 'multiclass'."
        )
    objective_mapper: dict[str, ObjLike] = {
        "binary": partial(binary_focal_objective, gamma=gamma),
        "multiclass": partial(multiclass_focal_objective, alpha=alpha, gamma=gamma),
    }
    eval_mapper: dict[str, EvalLike] = {
        "binary": "binary_logloss",
        "multiclass": "multi_logloss",
    }
    fobj = objective_mapper[inferred_task]
    feval = eval_mapper[inferred_task]

    return fobj, feval


def set_params(params: dict[str, Any], train_set: Dataset) -> dict[str, Any]:
    """Set params and eval finction, objective in params."""
    _params = deepcopy(params)
    if OBJECTIVE_STR in _params:
        logger.warning(f"'{OBJECTIVE_STR}' exists in params will not used.")
        del _params[OBJECTIVE_STR]

    _alpha = _params.pop("alpha", ALPHA_DEFAULT)
    _gamma = _params.pop("gamma", GAMMA_DEFAULT)

    fobj, feval = _set_fobj_feval(train_set=train_set, alpha=_alpha, gamma=_gamma)
    _params.update({OBJECTIVE_STR: fobj, METRIC_STR: feval})
    return _params


def get_params(trial: optuna.Trial) -> dict[str, Any]:
    """Get default params."""
    return {
        "alpha": trial.suggest_float("alpha", 0.25, 0.75),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }
