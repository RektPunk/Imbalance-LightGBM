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


_CLASSIFIER_DOC = """Construct a gradient boosting model.
Parameters
----------
objective : str
    Specify the learning objective
    'binary_focal', 'binary_weighted'.
alpha: float
gamma: float
boosting_type : str, optional (default='gbdt')
    'gbdt', traditional Gradient Boosting Decision Tree.
    'dart', Dropouts meet Multiple Additive Regression Trees.
    'rf', Random Forest.
num_leaves : int, optional (default=31)
    Maximum tree leaves for base learners.
max_depth : int, optional (default=-1)
    Maximum tree depth for base learners, <=0 means no limit.
    If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
learning_rate : float, optional (default=0.1)
    Boosting learning rate.
    You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
    in training using ``reset_parameter`` callback.
    Note, that this will ignore the ``learning_rate`` argument in training.
n_estimators : int, optional (default=100)
    Number of boosted trees to fit.
subsample_for_bin : int, optional (default=200000)
    Number of samples for constructing bins.
class_weight : dict, 'balanced' or None, optional (default=None)
    Weights associated with classes in the form ``{class_label: weight}``.
    Use this parameter only for multi-class classification task;
    for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
    Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
    You may want to consider performing probability calibration
    (https://scikit-learn.org/stable/modules/calibration.html) of your model.
    The 'balanced' mode uses the values of y to automatically adjust weights
    inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
    If None, all classes are supposed to have weight one.
    Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
    if ``sample_weight`` is specified.
min_split_gain : float, optional (default=0.)
    Minimum loss reduction required to make a further partition on a leaf node of the tree.
min_child_weight : float, optional (default=1e-3)
    Minimum sum of instance weight (Hessian) needed in a child (leaf).
min_child_samples : int, optional (default=20)
    Minimum number of data needed in a child (leaf).
subsample : float, optional (default=1.)
    Subsample ratio of the training instance.
subsample_freq : int, optional (default=0)
    Frequency of subsample, <=0 means no enable.
colsample_bytree : float, optional (default=1.)
    Subsample ratio of columns when constructing each tree.
reg_alpha : float, optional (default=0.)
    L1 regularization term on weights.
reg_lambda : float, optional (default=0.)
    L2 regularization term on weights.
random_state : int, RandomState object or None, optional (default=None)
    Random number seed.
    If int, this number is used to seed the C++ code.
    If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
    If None, default seeds in C++ code are used.
n_jobs : int or None, optional (default=None)
    Number of parallel threads to use for training (can be changed at prediction time by
    passing it as an extra keyword argument).

    For better performance, it is recommended to set this to the number of physical cores
    in the CPU.

    Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
    scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
    threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
    to using the number of physical cores in the system (its correct detection requires
    either the ``joblib`` or the ``psutil`` util libraries to be installed).

    .. versionchanged:: 4.0.0

importance_type : str, optional (default='split')
    The type of feature importance to be filled into ``feature_importances_``.
    If 'split', result contains numbers of times the feature is used in a model.
    If 'gain', result contains total gains of splits which use the feature.
**kwargs
    Other parameters for the model.
    Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

    .. warning::

        \*\*kwargs is not supported in sklearn, it may cause unexpected issues.
"""


_PARAMS_MAPPER: dict[str, tuple[str, str]] = {
    "train": _TRAIN_DOC,
    "cv": _CV_DOC,
    "classifier": _CLASSIFIER_DOC,
}


def add_docstring(func_name: str) -> Callable:
    """Decorator to add a docstring to a function."""

    def decorator(func: Callable) -> Callable:
        func.__doc__ = _PARAMS_MAPPER[func_name]
        return func

    return decorator
