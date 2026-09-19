from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import Dataset
from scipy.sparse import spmatrix
from scipy.special import expit, softmax

from imlightgbm.base import ALPHA_DEFAULT, GAMMA_DEFAULT, validate_positive_number
from imlightgbm.objective import (
    binary_focal_lgb_objective,
    binary_weighted_lgb_objective,
    multiclass_focal_lgb_objective,
    multiclass_weighted_lgb_objective,
)

SUPPORTED_TASKS = {"binary", "multiclass"}
BINARY_OBJECTIVES = {"binary_focal", "binary_weighted"}
MULTICLASS_OBJECTIVES = {"multiclass_focal", "multiclass_weighted"}

BINARY_METRICS = {"auc", "binary_error", "binary_logloss"}
MULTICLASS_METRICS = {"auc_mu", "multi_logloss", "multi_error"}


def select_metric(objective: str, metric: str | None) -> str:
    """Retrieve the appropriate metric function based on task."""
    return (
        metric or BINARY_METRICS.pop()
        if objective in BINARY_OBJECTIVES
        else MULTICLASS_METRICS.pop()
    )


def select_objective(
    objective: str,
    alpha: float,
    gamma: float,
    num_class: int,
) -> Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]:
    """Retrieve the appropriate objective function based on task and objective type."""
    objective_mapper: dict[
        str, Callable[[np.ndarray, Dataset], tuple[np.ndarray, np.ndarray]]
    ] = {
        "binary_focal": partial(binary_focal_lgb_objective, gamma=gamma),
        "binary_weighted": partial(binary_weighted_lgb_objective, alpha=alpha),
        "multiclass_focal": partial(
            multiclass_focal_lgb_objective,
            gamma=gamma,
            num_class=num_class,
        ),
        "multiclass_weighted": partial(
            multiclass_weighted_lgb_objective,
            alpha=alpha,
            num_class=num_class,
        ),
    }
    return objective_mapper[objective]


def set_params(params: dict[str, Any]) -> dict[str, Any]:
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

    feval = select_metric(objective=_objective, metric=_metric)
    fobj = select_objective(
        objective=_objective,
        alpha=_alpha,
        gamma=_gamma,
        num_class=_params.get("num_class", -1),
    )
    _params.update({"objective": fobj, "metric": feval})
    return _params


class ImbalancedBooster(lgb.Booster):
    def predict(
        self,
        data: lgb.basic._LGBM_PredictDataType,
        *args,
        **kwargs,
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        _predict = super().predict(data, *args, **kwargs)
        if (
            kwargs.get("raw_score", False)
            or kwargs.get("pred_leaf", False)
            or kwargs.get("pred_contrib", False)
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if _predict.ndim == 1:
            return expit(_predict)

        return softmax(_predict, axis=1)


def train(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    *args,
    **kwargs,
) -> ImbalancedBooster:
    """Perform the training with given parameters."""
    _params = set_params(params=params)
    _booster = lgb.train(_params, train_set, *args, **kwargs)

    return ImbalancedBooster(model_str=_booster.model_to_string())


def cv(
    params: dict[str, Any], train_set: lgb.Dataset, *args, **kwargs
) -> dict[str, list[float] | lgb.CVBooster]:
    """Perform the cross-validation with given parameters."""
    _params = set_params(params=params)

    return lgb.cv(_params, train_set, *args, **kwargs)
