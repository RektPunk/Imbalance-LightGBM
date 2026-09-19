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
from imlightgbm.utils import validate_positive_number

ObjLike = Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]

_OBJECTIVE_STR: str = "objective"
_METRIC_STR: str = "metric"
_NUM_CLASS_STR: str = "num_class"


def _safe_power(num_base: np.ndarray, num_pow: float) -> np.ndarray:
    """Safe power."""
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


def _safe_log(array: np.ndarray, min_value: float = 1e-6) -> np.ndarray:
    """Safe log."""
    return np.log(np.clip(array, min_value, None))


def _weighted_grad_hess(
    y_true: np.ndarray, pred_prob: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted grad hess."""
    grad = -(alpha**y_true) * (y_true - pred_prob)
    hess = (alpha**y_true) * pred_prob * (1.0 - pred_prob)
    return grad, hess


def _focal_grad_hess(
    y_true: np.ndarray, pred_prob: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return focal grad hess."""
    prob_product = pred_prob * (1 - pred_prob)
    true_diff_pred = y_true + ((-1) ** y_true) * pred_prob
    focal_grad_term = pred_prob + y_true - 1
    focal_log_term = 1 - y_true - ((-1) ** y_true) * pred_prob
    focal_grad_base = y_true + ((-1) ** y_true) * pred_prob
    grad = gamma * focal_grad_term * _safe_power(true_diff_pred, gamma) * _safe_log(
        focal_log_term
    ) + ((-1) ** y_true) * _safe_power(focal_grad_base, (gamma + 1))

    hess_term1 = _safe_power(true_diff_pred, gamma) + gamma * (
        (-1) ** y_true
    ) * focal_grad_term * _safe_power(true_diff_pred, (gamma - 1))
    hess_term2 = (
        ((-1) ** y_true)
        * focal_grad_term
        * _safe_power(true_diff_pred, gamma)
        / focal_log_term
    )
    hess = (
        (hess_term1 * _safe_log(focal_log_term) - hess_term2) * gamma
        + (gamma + 1) * _safe_power(focal_grad_base, gamma)
    ) * prob_product
    return grad, hess


def binary_focal_objective(
    y_true: np.ndarray, y_pred: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective."""
    pred_prob = expit(y_pred)
    return _focal_grad_hess(y_true=y_true, pred_prob=pred_prob, gamma=gamma)


def binary_weighted_objective(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary weighted objective."""
    pred_prob = expit(y_pred)
    return _weighted_grad_hess(y_true=y_true, pred_prob=pred_prob, alpha=alpha)


def binary_focal_lgb_objective(
    pred: np.ndarray, train_data: Dataset, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective for engine."""
    label = cast(np.ndarray, train_data.get_label())
    return binary_focal_objective(y_true=label, y_pred=pred, gamma=gamma)


def binary_weighted_lgb_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary weighted objective for engine."""
    label = cast(np.ndarray, train_data.get_label())
    return binary_weighted_objective(y_true=label, y_pred=pred, alpha=alpha)


def multiclass_focal_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true.astype(int)]
    return _focal_grad_hess(y_true=y_true_onehot, pred_prob=pred_prob, gamma=gamma)


def multiclass_weighted_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass weighted objective."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true.astype(int)]
    return _weighted_grad_hess(y_true=y_true_onehot, pred_prob=pred_prob, alpha=alpha)


def multiclass_focal_lgb_objective(
    pred: np.ndarray,
    train_data: Dataset,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective for engine."""
    label = cast(np.ndarray, train_data.get_label())
    return multiclass_focal_objective(
        y_true=label,
        y_pred=pred,
        gamma=gamma,
        num_class=num_class,
    )


def multiclass_weighted_lgb_objective(
    pred: np.ndarray,
    train_data: Dataset,
    alpha: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass weighted objective for engine."""
    label = cast(np.ndarray, train_data.get_label())
    return multiclass_weighted_objective(
        y_true=label,
        y_pred=pred,
        alpha=alpha,
        num_class=num_class,
    )


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
