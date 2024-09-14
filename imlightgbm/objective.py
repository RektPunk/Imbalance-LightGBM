import numpy as np
from lightgbm import Dataset


def binary_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    is_higher_better = False
    return "binary_focal", ..., is_higher_better


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    # TODO
    return ...


def multiclass_focal_eval(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    # TODO
    return


def multiclass_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
):
    # TODO
    return
