from collections.abc import Iterable
from typing import Any, Callable, Literal

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import BaseCrossValidator

from imlightgbm.objective import set_params
from imlightgbm.utils import docstring, optimize_doc


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
    _params = set_params(params=params, train_set=train_set)
    return lgb.train(
        params=_params,
        train_set=train_set,
        valid_sets=valid_sets,
        valid_names=valid_names,
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
    _params = set_params(params=params, train_set=train_set)
    return lgb.cv(
        params=_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        folds=folds,
        nfold=nfold,
        stratified=stratified,
        shuffle=shuffle,
        metrics=metrics,
        init_model=init_model,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        fpreproc=fpreproc,
        seed=seed,
        callbacks=callbacks,
        eval_train_metric=eval_train_metric,
        return_cvbooster=return_cvbooster,
    )


def get_params(trial: optuna.Trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.25, 0.75),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }


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
    feature_name: list[str] | Literal["auto"] = "auto",
    categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
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
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            fpreproc=fpreproc,
            seed=seed,
            callbacks=callbacks,
            eval_train_metric=False,
            return_cvbooster=False,
        )
        _keys = [_ for _ in cv_results.keys() if _.endswith("mean")]
        assert len(_keys) == 1
        return min(cv_results[_keys[0]])

    study = optuna.create_study(direction="minimize")
    study.optimize(_objective, n_trials=num_trials)
    return study


optimize.__doc__ = optimize_doc
