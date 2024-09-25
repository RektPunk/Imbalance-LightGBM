import numpy as np
from lightgbm import Dataset
from scipy.special import expit


def _safe_power(num_base: np.ndarray, num_pow: float):
    """Safe power."""
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


def _safe_log(array: np.ndarray, min_value: float = 1e-6) -> np.ndarray:
    """Safe log."""
    return np.log(np.clip(array, min_value, None))


def sklearn_binary_focal_objective(
    y_true: np.ndarray, y_pred: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective."""
    pred_prob = expit(y_pred)

    # gradient
    g1 = pred_prob * (1 - pred_prob)
    g2 = y_true + ((-1) ** y_true) * pred_prob
    g3 = pred_prob + y_true - 1
    g4 = 1 - y_true - ((-1) ** y_true) * pred_prob
    g5 = y_true + ((-1) ** y_true) * pred_prob
    grad = gamma * g3 * _safe_power(g2, gamma) * _safe_log(g4) + (
        (-1) ** y_true
    ) * _safe_power(g5, (gamma + 1))
    # hess
    h1 = _safe_power(g2, gamma) + gamma * ((-1) ** y_true) * g3 * _safe_power(
        g2, (gamma - 1)
    )
    h2 = ((-1) ** y_true) * g3 * _safe_power(g2, gamma) / g4
    hess = (
        (h1 * _safe_log(g4) - h2) * gamma + (gamma + 1) * _safe_power(g5, gamma)
    ) * g1
    return grad, hess


def sklearn_binary_weighted_objective(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary weighted objective."""
    pred_prob = expit(y_pred)
    grad = -(alpha**y_true) * (y_true - pred_prob)
    hess = (alpha**y_true) * pred_prob * (1.0 - pred_prob)
    return grad, hess


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective."""
    label = train_data.get_label()
    grad, hess = sklearn_binary_focal_objective(
        y_true=label,
        y_pred=pred,
        gamma=gamma,
    )
    return grad, hess


def binary_weighted_objective(pred: np.ndarray, train_data: Dataset, alpha: float):
    """Return grad, hess for binary weighted objective."""
    label = train_data.get_label()
    grad, hess = sklearn_binary_weighted_objective(
        y_true=label, y_pred=pred, alpha=alpha
    )
    return grad, hess


def multiclass_focal_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    # TODO
    return


def multiclass_weighted_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float, gamma: float
) -> tuple[str, float, bool]:
    # TODO
    return
