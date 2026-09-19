from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit, softmax

from imlightgbm.objective import (
    binary_focal_lgb_objective,
    binary_weighted_lgb_objective,
    multiclass_focal_lgb_objective,
    multiclass_weighted_lgb_objective,
)
from imlightgbm.parameters import select_alpha, select_gamma


def select_metric(
    objective: str,
    metric: str | list[str] | tuple[str, ...] | set[str] | None,
) -> str | list[str] | tuple[str, ...] | set[str]:
    """Select the metric for the objective."""
    if metric is not None:
        return metric

    return (
        "binary_logloss"
        if objective in {"binary_focal", "binary_weighted"}
        else "multi_logloss"
    )


def select_objective(
    objective: str,
    alpha: float,
    gamma: float,
    num_class: int,
) -> Callable:
    """Retrieve the appropriate objective function based on task and objective type."""
    objective_mapper: dict[str, Callable] = {
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
    if callable(_metric) or (
        isinstance(_metric, (list, tuple, set)) and any(callable(m) for m in _metric)
    ):
        raise ValueError("custom metric are not supported.")

    if _metric is not None and not isinstance(_metric, (str, list, tuple, set)):
        raise TypeError(
            f"metric must be a string or collection of strings, but got {type(_metric).__name__}."
        )

    _alpha = select_alpha(_objective, _params.pop("alpha", None))
    _gamma = select_gamma(_objective, _params.pop("gamma", None))
    fobj = select_objective(
        objective=_objective,
        alpha=_alpha,
        gamma=_gamma,
        num_class=_params.get("num_class", -1),
    )
    feval = select_metric(objective=_objective, metric=_metric)
    _params.update({"objective": fobj, "metric": feval})
    return _params


class ImbalancedBooster(lgb.Booster):
    def predict(
        self,
        data: lgb.basic._LGBM_PredictDataType,
        *args,
        **kwargs,
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        """Make a prediction."""
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
