from collections.abc import Iterable
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit, softmax
from sklearn.model_selection import BaseCrossValidator

from imlightgbm.docstring import add_docstring
from imlightgbm.objective.engine import set_params


class ImbalancedBooster(lgb.Booster):
    def predict(
        self,
        data: lgb.basic._LGBM_PredictDataType,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        _predict = super().predict(
            data=data,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            data_has_header=data_has_header,
            validate_features=validate_features,
            **kwargs,
        )
        if (
            raw_score
            or pred_leaf
            or pred_contrib
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if len(_predict.shape) == 1:
            return expit(_predict)
        else:
            return softmax(_predict, axis=1)


@add_docstring("train")
def train(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    num_boost_round: int = 100,
    valid_sets: list[lgb.Dataset] = None,
    valid_names: list[str] = None,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    keep_training_booster: bool = False,
    callbacks: list[Callable] | None = None,
) -> ImbalancedBooster:
    _params = set_params(params=params, train_set=train_set)
    _booster = lgb.train(
        params=_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        init_model=init_model,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
    )
    _booster_str = _booster.model_to_string()
    return ImbalancedBooster(model_str=_booster_str)


@add_docstring("cv")
def cv(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    num_boost_round: int = 100,
    folds: Iterable[tuple[np.ndarray, np.ndarray]] | BaseCrossValidator | None = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    fpreproc: Callable[
        [lgb.Dataset, lgb.Dataset, dict[str, Any]],
        tuple[lgb.Dataset, lgb.Dataset, dict[str, Any]],
    ]
    | None = None,
    seed: int = 0,
    callbacks: list[Callable] | None = None,
    eval_train_metric: bool = False,
    return_cvbooster: bool = False,
) -> dict[str, list[float] | lgb.CVBooster]:
    _params = set_params(params=params, train_set=train_set)
    return lgb.cv(
        params=_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        folds=folds,
        nfold=nfold,
        stratified=stratified,
        shuffle=shuffle,
        init_model=init_model,
        fpreproc=fpreproc,
        seed=seed,
        callbacks=callbacks,
        eval_train_metric=eval_train_metric,
        return_cvbooster=return_cvbooster,
    )
