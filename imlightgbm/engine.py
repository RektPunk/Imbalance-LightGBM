from collections.abc import Iterable
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import BaseCrossValidator

from imlightgbm.docstring import add_docstring
from imlightgbm.objective import get_params, set_params


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
) -> lgb.Booster:
    _params = set_params(params=params, train_set=train_set)
    return lgb.train(
        params=_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        init_model=init_model,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
    )


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


@add_docstring("optimize")
def optimize(
    train_set: lgb.Dataset,
    num_trials: int = 10,
    num_boost_round: int = 100,
    folds: Iterable[tuple[np.ndarray, np.ndarray]] | BaseCrossValidator | None = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    get_params: Callable[[optuna.Trial], dict[str, Any]] = get_params,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    fpreproc: Callable[
        [lgb.Dataset, lgb.Dataset, dict[str, Any]],
        tuple[lgb.Dataset, lgb.Dataset, dict[str, Any]],
    ]
    | None = None,
    seed: int = 0,
    callbacks: list[Callable] | None = None,
) -> optuna.Study:
    def _objective(trial: optuna.Trial):
        """Optuna objective function."""
        params = get_params(trial)
        cv_results = cv(
            params=params,
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
        )
        _keys = [_ for _ in cv_results.keys() if _.endswith("mean")]
        assert len(_keys) == 1
        return min(cv_results[_keys[0]])

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=num_trials)
    return study
