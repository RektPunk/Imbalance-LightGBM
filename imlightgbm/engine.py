from collections.abc import Iterable
from typing import Any, Callable, Literal

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import BaseCrossValidator

from imlightgbm.objective import set_fobj_feval
from imlightgbm.utils import docstring, logger


@docstring(lgb.train.__doc__)
def train(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    valid_sets: list[lgb.Dataset] = None,
    valid_names: list[str] = None,
    num_boost_round: int = 100,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    feature_name: list[str] | Literal["auto"] = "auto",
    categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
    keep_training_booster: bool = False,
    callbacks: list[Callable] | None = None,
) -> lgb.Booster:
    if "objective" in params:
        logger.warning("'objective' exists in params will not used.")
        del params["objective"]

    _alpha = params.pop("alpha", 0.05)
    _gamma = params.pop("gamma", 0.01)

    fobj, feval = set_fobj_feval(train_set=train_set, alpha=_alpha, gamma=_gamma)
    params.update({"objective": fobj})

    return lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        num_boost_round=num_boost_round,
        init_model=init_model,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
    )


@docstring(lgb.cv.__doc__)
def cv(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    num_boost_round: int = 100,
    folds: Iterable[tuple[np.ndarray, np.ndarray]] | BaseCrossValidator | None = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    metrics: str | list[str] | None = None,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    feature_name: list[str] | Literal["auto"] = "auto",
    categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
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
    if "objective" in params:
        logger.warning("'objective' exists in params will not used.")
        del params["objective"]

    _alpha = params.pop("alpha", 0.05)
    _gamma = params.pop("gamma", 0.01)

    fobj, feval = set_fobj_feval(train_set=train_set, alpha=_alpha, gamma=_gamma)
    params.update({"objective": fobj})
    return lgb.cv(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        folds=folds,
        nfold=nfold,
        stratified=stratified,
        shuffle=shuffle,
        metrics=metrics,
        feavl=feval,
        init_model=init_model,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        fpreproc=fpreproc,
        seed=seed,
        callbacks=callbacks,
        eval_train_metric=eval_train_metric,
        return_cvbooster=return_cvbooster,
    )
