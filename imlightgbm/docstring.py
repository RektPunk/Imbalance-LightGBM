from typing import Callable

_TRAIN_DOC = """Perform the training with given parameters.
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
booster : Booster
    The trained Booster model.
"""

_CV_DOC = """Perform the cross-validation with given parameters.
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


_PARAMS_MAPPER: dict[str, tuple[str, str]] = {
    "train": _TRAIN_DOC,
    "cv": _CV_DOC,
}


def add_docstring(func_name: str) -> Callable:
    """Decorator to add a docstring to a function."""

    def decorator(func: Callable) -> Callable:
        func.__doc__ = _PARAMS_MAPPER[func_name]
        return func

    return decorator
