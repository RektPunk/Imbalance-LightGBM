from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit, softmax

from imlightgbm.common import set_params


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
    _params = set_params(params=params, train_set=train_set)
    _booster = lgb.train(_params, train_set, *args, **kwargs)

    return ImbalancedBooster(model_str=_booster.model_to_string())


def cv(
    params: dict[str, Any], train_set: lgb.Dataset, *args, **kwargs
) -> dict[str, list[float] | lgb.CVBooster]:
    """Perform the cross-validation with given parameters."""
    _params = set_params(params=params, train_set=train_set)

    return lgb.cv(_params, train_set, *args, **kwargs)
