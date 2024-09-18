import logging
from typing import Callable


def _modify_docstring(docstring: str) -> str:
    lines = docstring.splitlines()

    feval_start = next(i for i, line in enumerate(lines) if "feval" in line)
    init_model_start = next(i for i, line in enumerate(lines) if "init_model" in line)
    del lines[feval_start:init_model_start]

    note_start = next(i for i, line in enumerate(lines) if "Note" in line)
    returns_start = next(i for i, line in enumerate(lines) if "Returns" in line)
    del lines[note_start:returns_start]
    return "\n".join(lines)


def docstring(doc: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        func.__doc__ = _modify_docstring(doc)
        return func

    return decorator


def init_logger() -> logging.Logger:
    logger = logging.getLogger("imlightgbm")
    logger.setLevel(logging.INFO)
    return logger


logger = init_logger()


optimize_doc = """Perform the cross-validation with given parameters.
Parameters
----------
train_set : Dataset
    Data to be trained on.
num_trials : int, optional (default=10)
    Number of hyperparameter search trials.
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
get_params : callable, optional (default=get_params)
    def get_params(trial: optuna.Trial):
        return {
            'alpha': trial.suggest_float('alpha', .25, .75),
            'gamma': trial.suggest_float('gamma', .0, 3.),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }
init_model : str, pathlib.Path, Booster or None, optional (default=None)
    Filename of LightGBM model or Booster instance used for continue training.
feature_name : list of str, or 'auto', optional (default="auto")
    **Deprecated.** Set ``feature_name`` on ``train_set`` instead.
    Feature names.
    If 'auto' and data is pandas DataFrame, data columns names are used.
categorical_feature : list of str or int, or 'auto', optional (default="auto")
    **Deprecated.** Set ``categorical_feature`` on ``train_set`` instead.
    Categorical features.
    If list of int, interpreted as indices.
    If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
    If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
    All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
    Large values could be memory consuming. Consider using consecutive integers starting from zero.
    All negative values in categorical features will be treated as missing values.
    The output cannot be monotonically constrained with respect to a categorical feature.
    Floating point numbers in categorical features will be rounded towards 0.
fpreproc : callable or None, optional (default=None)
    Preprocessing function that takes (dtrain, dtest, params)
    and returns transformed versions of those.
seed : int, optional (default=0)
    Seed used to generate the folds (passed to numpy.random.seed).
callbacks : list of callable, or None, optional (default=None)
    List of callback functions that are applied at each iteration.
    See Callbacks in Python API for more information.

Returns
-------
study: optuna.Study
"""
