from copy import deepcopy
from functools import partial
from typing import Any, Callable

import numpy as np
from lightgbm import Dataset
from sklearn.utils.multiclass import type_of_target

from imlightgbm.base import (
    ALPHA_DEFAULT,
    GAMMA_DEFAULT,
    Metric,
    Objective,
    SupportedTask,
)
from imlightgbm.objective.core import (
    binary_focal_objective,
    binary_weighted_objective,
    multiclass_focal_objective,
    multiclass_weighted_objective,
)
from imlightgbm.utils import validate_positive_number

ObjLike = Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]

OBJECTIVE_STR: str = "objective"
METRIC_STR: str = "metric"


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
            Objective.binary_focal: partial(binary_focal_objective, gamma=gamma),
            Objective.binary_weighted: partial(binary_weighted_objective, alpha=alpha),
        },
        SupportedTask.multiclass: {
            Objective.multiclass_focal: partial(
                multiclass_focal_objective, alpha=alpha, gamma=gamma
            ),
            Objective.multiclass_weighted: partial(
                multiclass_weighted_objective, alpha=alpha, gamma=gamma
            ),
        },
    }
    if objective:
        objective_enum = Objective.get(objective)
        return objective_mapper[task_enum][objective_enum]

    focal_key = [key for key in objective_mapper[task_enum] if key.endswith("focal")][0]
    return objective_mapper[task_enum][focal_key]


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

    _alpha = _params.pop("alpha", ALPHA_DEFAULT)
    _gamma = _params.pop("gamma", GAMMA_DEFAULT)

    validate_positive_number(_alpha)
    validate_positive_number(_gamma)

    fobj, feval = _get_fobj_feval(
        train_set=train_set,
        alpha=_alpha,
        gamma=_gamma,
        objective=_objective,
        metric=_metric,
    )
    _params.update({OBJECTIVE_STR: fobj, METRIC_STR: feval})
    return _params
