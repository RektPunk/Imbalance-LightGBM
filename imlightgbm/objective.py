from copy import deepcopy
from functools import partial
from typing import Any, Callable

import numpy as np
from lightgbm import Dataset
from sklearn.utils.multiclass import type_of_target

from imlightgbm.base import Metric, Objective, SupportedTask

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
    """Return grad, hess for binary focal objective."""
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


def binary_weighted_objective(pred: np.ndarray, train_data: Dataset, alpha: float):
    """Return grad, hess for binary weighted objective."""
    label = train_data.get_label()
    pred_prob = _sigmoid(pred)
    grad = -(alpha**label) * (label - pred_prob)
    hess = (alpha**label) * pred_prob * (1.0 - pred_prob)
    return grad, hess


def multiclass_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    # TODO
    return


def multiclass_weighted_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[str, float, bool]:
    # TODO
    return


def _get_metric(task_enum: SupportedTask, metric: str | None) -> str:
    """Retrieve the appropriate metric function based on task."""
    metric_mapper: dict[SupportedTask, list[Metric]] = {
        SupportedTask.binary: [Metric.auc, Metric.binary_error, Metric.binary_logloss],
        SupportedTask.multiclass: [
            Metric.auc_mu,
            Metric.multi_logloss,
            Metric.multi_error,
        ],
    }
    if metric:
        metric_enum = Metric.get(metric)
        metric_enums = metric_mapper[task_enum]
        if metric_enum not in metric_enums:
            valid_metrics = ", ".join([m.value for m in metric_enums])
            raise ValueError(f"Invalid metric: Supported metrics are {valid_metrics}")
        return metric_enum.value

    return metric_mapper[task_enum][0].value


def _get_objective(
    task_enum: SupportedTask, objective: str | None, alpha: float, gamma: float
) -> ObjLike:
    """Retrieve the appropriate objective function based on task and objective type."""
    objective_mapper: dict[SupportedTask, dict[Objective, ObjLike]] = {
        SupportedTask.binary: {
            Objective.focal: partial(binary_focal_objective, gamma=gamma),
            Objective.weighted: partial(binary_weighted_objective, alpha=alpha),
        },
        SupportedTask.multiclass: {
            Objective.focal: partial(
                multiclass_focal_objective, alpha=alpha, gamma=gamma
            ),
            Objective.weighted: partial(
                multiclass_weighted_objective, alpha=alpha, gamma=gamma
            ),
        },
    }
    if objective:
        objective_enum = Objective.get(objective)
        return objective_mapper[task_enum][objective_enum]

    return objective_mapper[task_enum][Objective.focal]


def _get_fobj_feval(
    train_set: Dataset,
    alpha: float,
    gamma: float,
    objective: str | None,
    metric: str | None,
) -> tuple[ObjLike, str]:
    """Return obj and eval with respect to task type."""
    _task = type_of_target(train_set.get_label())
    task_enum = SupportedTask.get(_task)
    # FIXME: remove after developing multiclass objective
    if task_enum != SupportedTask.binary:
        raise ValueError(
            "Inferred task is not binary. Multiclass classification will be supported starting from version 0.1.0."
        )
    feval = _get_metric(task_enum=task_enum, metric=metric)
    fobj = _get_objective(
        task_enum=task_enum, objective=objective, alpha=alpha, gamma=gamma
    )
    return fobj, feval


def set_params(params: dict[str, Any], train_set: Dataset) -> dict[str, Any]:
    """Set params and eval finction, objective in params."""
    _params = deepcopy(params)
    _objective = _params.pop(OBJECTIVE_STR, None)
    _metric = _params.pop(METRIC_STR, None)

    if _metric and not isinstance(_metric, str):
        raise ValueError("metric must be str")

    fobj, feval = _get_fobj_feval(
        train_set=train_set,
        alpha=_params.pop("alpha", ALPHA_DEFAULT),
        gamma=_params.pop("gamma", GAMMA_DEFAULT),
        objective=_objective,
        metric=_metric,
    )
    _params.update({OBJECTIVE_STR: fobj, METRIC_STR: feval})
    return _params
