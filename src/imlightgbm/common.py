from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any, cast

import numpy as np
from lightgbm import Dataset
from scipy.special import expit, softmax
from sklearn.utils.multiclass import type_of_target

from imlightgbm.base import (
    ALPHA_DEFAULT,
    GAMMA_DEFAULT,
    Metric,
    Objective,
    SupportedTask,
)
from imlightgbm.objective import (
    binary_focal_lgb_objective,
    binary_weighted_lgb_objective,
    multiclass_focal_lgb_objective,
    multiclass_weighted_lgb_objective,
)
from imlightgbm.utils import validate_positive_number

ObjLike = Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]


def _get_metric(task_enum: SupportedTask, metric: str | None) -> str:
    """Retrieve the appropriate metric function based on task."""
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
        metric_enum = Metric[metric]
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
                binary_focal_lgb_objective,
                gamma=gamma,
            ),
            Objective.binary_weighted: partial(
                binary_weighted_lgb_objective,
                alpha=alpha,
            ),
        },
        SupportedTask.multiclass: {
            Objective.multiclass_focal: partial(
                multiclass_focal_lgb_objective,
                gamma=gamma,
                num_class=num_class,
            ),
            Objective.multiclass_weighted: partial(
                multiclass_weighted_lgb_objective,
                alpha=alpha,
                num_class=num_class,
            ),
        },
    }
    objective_enum = Objective[objective]
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
    task_enum = SupportedTask[_task]
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
    """Set params and eval function, objective in params."""
    _params = deepcopy(params)
    if "objective" not in params:
        raise ValueError("objective must be included in params.")

    _objective: str = _params["objective"]
    _metric = _params.pop("metric", None)

    if _metric and not isinstance(_metric, str):
        raise ValueError("metric must be str.")

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
        num_class=_params.get("num_class", None),
    )
    _params.update({"objective": fobj, "metric": feval})
    return _params
