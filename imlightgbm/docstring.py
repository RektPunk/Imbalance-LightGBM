from typing import Callable

_space = "\n        "
ALL_PARAMS = {
    "params": f"dict{_space}Parameters for training. Values passed through ``params`` take precedence over those supplied via arguments.",
    "train_set": f"Dataset{_space}Data to be trained on.",
    "num_boost_round": f"int, optional (default=100){_space}Number of boosting iterations.",
    "valid_sets": f"list of Dataset, or None, optional (default=None){_space}List of data to be evaluated on during training.",
    "valid_names": f"list of str, or None, optional (default=None){_space}Names of ``valid_sets``.",
    "folds": f"generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None){_space}If generator or iterator, it should yield the train and test indices for each fold.{_space}If object, it should be one of the scikit-learn splitter classes{_space}(https://scikit-learn.org/stable/modules/classes.html#splitter-classes){_space}and have ``split`` method.{_space}This argument has highest priority over other data split arguments.",
    "nfold": f"int, optional (default=5){_space}Number of folds in CV.",
    "stratified": f"bool, optional (default=True){_space}Whether to perform stratified sampling.",
    "shuffle": f"bool, optional (default=True){_space}Whether to shuffle before splitting data.",
    "init_model": f"str, pathlib.Path, Booster or None, optional (default=None){_space}Filename of LightGBM model or Booster instance used for continue training.",
    "fpreproc": f"callable or None, optional (default=None){_space}Preprocessing function that takes (dtrain, dtest, params) and returns transformed versions of those.",
    "seed": f"int, optional (default=0){_space}Seed used to generate the folds (passed to numpy.random.seed).",
    "keep_training_booster": f"bool, optional (default=False){_space}Whether the returned Booster will be used to keep training.{_space}If False, the returned value will be converted into _InnerPredictor before returning.{_space}This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.{_space}When your model is very large and cause the memory error,{_space}you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.{_space}You can still use _InnerPredictor as ``init_model`` for future continue training.",
    "callbacks": f"list of callable, or None, optional (default=None){_space}List of callback functions that are applied at each iteration.{_space}See Callbacks in Python API for more information.",
    "eval_train_metric": f"bool, optional (default=False){_space}Whether to display the train metric in progress.",
    "return_cvbooster": f"bool, optional (default=False){_space}Whether to return Booster models trained on each fold through ``CVBooster``.",
}


PARAMS_MAPPER = {
    "train": {
        "description": "Perform the training with given parameters.",
        "selected_params": [
            "params",
            "train_set",
            "num_boost_round",
            "valid_sets",
            "valid_names",
            "init_model",
            "keep_training_booster",
            "callbacks",
        ],
        "return_description": f"booster: Booster{_space}The trained Booster model.",
    },
    "cv": {
        "description": "Perform the cross-validation with given parameters.",
        "selected_params": [
            "params",
            "train_set",
            "num_boost_round",
            "folds",
            "nfold",
            "stratified",
            "shuffle",
            "init_model",
            "fpreproc",
            "seed",
            "callbacks",
            "eval_train_metric",
            "return_cvbooster",
        ],
        "return_description": "eval_results: dict\n        History of evaluation results of each metric.\n        The dictionary has the following format:\n        {'valid metric1-mean': [values], 'valid metric1-stdv': [values],\n        'valid metric2-mean': [values], 'valid metric2-stdv': [values],\n        ...}.\n        If ``return_cvbooster=True``, also returns trained boosters wrapped in a ``CVBooster`` object via ``cvbooster`` key.\n        If ``eval_train_metric=True``, also returns the train metric history.\n        In this case, the dictionary has the following format:\n        {'train metric1-mean': [values], 'valid metric1-mean': [values],\n        'train metric2-mean': [values], 'valid metric2-mean': [values],\n        ...}.",
    },
}


def generate_docstring(
    description: str,
    selected_params: list[str],
    return_description: str = "",
) -> str:
    """Generate a docstring with a provided description, selected parameters, and optional return description."""
    docstring = f"{description}\n\n    Parameters\n    ----------\n"
    for param in selected_params:
        docstring += f"    {param}: {ALL_PARAMS[param]}\n"
    if return_description:
        docstring += f"\n    Returns\n    -------\n    {return_description}\n"
    return docstring


def add_docstring(func_name: str) -> Callable:
    """Decorator to add a docstring to a function based on provided parameters and descriptions."""

    def decorator(func: Callable) -> Callable:
        func.__doc__ = generate_docstring(**PARAMS_MAPPER[func_name])
        return func

    return decorator
