from collections.abc import Iterable
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
from scipy.sparse import spmatrix
from scipy.special import expit, softmax
from sklearn.model_selection import BaseCrossValidator

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

    predict.__doc__ = lgb.Booster.predict.__doc__


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
    """Perform the training with given parameters.
    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    valid_sets : list of Dataset, or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of str, or None, optional (default=None)
        Names of ``valid_sets``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.

    Returns
    -------
    booster : ImbalancedBooster
        The trained Booster model.
    """
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
    """Perform the cross-validation with given parameters.
    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=True)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    eval_train_metric : bool, optional (default=False)
        Whether to display the train metric in progress.
        The score of the metric is calculated again after each training step, so there is some impact on performance.
    return_cvbooster : bool, optional (default=False)
        Whether to return Booster models trained on each fold through ``CVBooster``.

    Returns
    -------
    eval_results : dict
        History of evaluation results of each metric.
        The dictionary has the following format:
        {'valid metric1-mean': [values], 'valid metric1-stdv': [values],
        'valid metric2-mean': [values], 'valid metric2-stdv': [values],
        ...}.
        If ``return_cvbooster=True``, also returns trained boosters wrapped in a ``CVBooster`` object via ``cvbooster`` key.
        If ``eval_train_metric=True``, also returns the train metric history.
        In this case, the dictionary has the following format:
        {'train metric1-mean': [values], 'valid metric1-mean': [values],
        'train metric2-mean': [values], 'valid metric2-mean': [values],
        ...}.
    """
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
