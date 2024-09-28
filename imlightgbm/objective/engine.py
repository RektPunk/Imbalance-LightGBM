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

_OBJECTIVE_STR: str = "objective"
_METRIC_STR: str = "metric"
_NUM_CLASS_STR: str = "num_class"


def _get_metric(task_enum: SupportedTask, metric: str | None) -> str:
    """Retrieve the appropriate metric function based on task.
    Defaults to auc (binary), auc_mu (multiclass).
    """
    metric_mapper: dict[SupportedTask, list[Metric]] = {
        SupportedTask.binary: [
            Metric.auc,
            Metric.binary_error,
            Metric.binary_logloss,
        ],
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
    task_enum: SupportedTask,
    objective: str,
    alpha: float,
    gamma: float,
    num_class: int | None,
) -> ObjLike:
    """Retrieve the appropriate objective function based on task and objective type."""
    objective_mapper: dict[SupportedTask, dict[Objective, ObjLike]] = {
        SupportedTask.binary: {
            Objective.binary_focal: partial(
                binary_focal_objective,
                gamma=gamma,
            ),
            Objective.binary_weighted: partial(
                binary_weighted_objective,
                alpha=alpha,
            ),
        },
        SupportedTask.multiclass: {
            Objective.multiclass_focal: partial(
                multiclass_focal_objective,
                gamma=gamma,
                num_class=num_class,
            ),
            Objective.multiclass_weighted: partial(
                multiclass_weighted_objective,
                alpha=alpha,
                num_class=num_class,
            ),
        },
    }
    objective_enum = Objective.get(objective)
    return objective_mapper[task_enum][objective_enum]


def _get_fobj_feval(
    train_set: Dataset,
    alpha: float,
    gamma: float,
    objective: str,
    metric: str | None,
    num_class: int | None,
) -> tuple[ObjLike, str]:
    """Return obj and eval with respect to task type.
    Raise ValueError when multiclass task without num_class.
    """
    _task = type_of_target(train_set.get_label())
    task_enum = SupportedTask.get(_task)
    if task_enum == SupportedTask.multiclass and num_class is None:
        raise ValueError(f"{_NUM_CLASS_STR} must be provided for multiclass.")

    feval = _get_metric(task_enum=task_enum, metric=metric)
    fobj = _get_objective(
        task_enum=task_enum,
        objective=objective,
        alpha=alpha,
        gamma=gamma,
        num_class=num_class,
    )
    return fobj, feval


def set_params(params: dict[str, Any], train_set: Dataset) -> dict[str, Any]:
    """Set params and eval finction, objective in params."""
    _params = deepcopy(params)
    if _OBJECTIVE_STR not in params:
        raise ValueError(f"{_OBJECTIVE_STR} must be included in params.")

    _objective: str = _params[_OBJECTIVE_STR]
    _metric = _params.pop(_METRIC_STR, None)

    if _metric and not isinstance(_metric, str):
        raise ValueError(f"{_METRIC_STR} must be str.")

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
        num_class=_params.get(_NUM_CLASS_STR, None),
    )
    _params.update({_OBJECTIVE_STR: fobj, _METRIC_STR: feval})
    return _params
