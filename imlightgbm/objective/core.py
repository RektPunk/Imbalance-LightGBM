import numpy as np
from lightgbm import Dataset
from scipy.special import expit, softmax


def _safe_power(num_base: np.ndarray, num_pow: float):
    """Safe power."""
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


def _safe_log(array: np.ndarray, min_value: float = 1e-6) -> np.ndarray:
    """Safe log."""
    return np.log(np.clip(array, min_value, None))


def sklearn_binary_focal_objective(
    y_true: np.ndarray, y_pred: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective for sklearn API."""
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
    """Return grad, hess for binary weighted objective for sklearn API."""
    pred_prob = expit(y_pred)
    grad = -(alpha**y_true) * (y_true - pred_prob)
    hess = (alpha**y_true) * pred_prob * (1.0 - pred_prob)
    return grad, hess


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective for engine."""
    label = train_data.get_label()
    grad, hess = sklearn_binary_focal_objective(
        y_true=label,
        y_pred=pred,
        gamma=gamma,
    )
    return grad, hess


def binary_weighted_objective(pred: np.ndarray, train_data: Dataset, alpha: float):
    """Return grad, hess for binary weighted objective for engine."""
    label = train_data.get_label()
    grad, hess = sklearn_binary_weighted_objective(
        y_true=label, y_pred=pred, alpha=alpha
    )
    return grad, hess


def sklearn_multiclass_focal_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective for sklearn API.."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true]

    # gradient
    g1 = pred_prob * (1 - pred_prob)
    g2 = y_true_onehot + ((-1) ** y_true_onehot) * pred_prob
    g3 = pred_prob + y_true_onehot - 1
    g4 = 1 - y_true_onehot - ((-1) ** y_true_onehot) * pred_prob
    g5 = y_true_onehot + ((-1) ** y_true_onehot) * pred_prob
    grad = gamma * g3 * _safe_power(g2, gamma) * _safe_log(g4) + (
        (-1) ** y_true_onehot
    ) * _safe_power(g5, (gamma + 1))
    # hess
    h1 = _safe_power(g2, gamma) + gamma * ((-1) ** y_true_onehot) * g3 * _safe_power(
        g2, (gamma - 1)
    )
    h2 = ((-1) ** y_true_onehot) * g3 * _safe_power(g2, gamma) / g4
    hess = (
        (h1 * _safe_log(g4) - h2) * gamma + (gamma + 1) * _safe_power(g5, gamma)
    ) * g1

    return grad, hess


def sklearn_multiclass_weighted_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass weighted objective for sklearn API."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true]
    grad = -(alpha**y_true_onehot) * (y_true_onehot - pred_prob)
    hess = (alpha**y_true_onehot) * pred_prob * (1.0 - pred_prob)
    return grad, hess


def multiclass_focal_objective(
    pred: np.ndarray,
    train_data: Dataset,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective for engine."""
    label = train_data.get_label().astype(int)
    grad, hess = sklearn_multiclass_focal_objective(
        y_true=label,
        y_pred=pred,
        gamma=gamma,
        num_class=num_class,
    )
    return grad, hess


def multiclass_weighted_objective(
    pred: np.ndarray,
    train_data: Dataset,
    alpha: float,
    num_class: int,
) -> tuple[str, float, bool]:
    """Return grad, hess for multclass weighted objective for engine."""
    label = train_data.get_label().astype(int)
    grad, hess = sklearn_multiclass_weighted_objective(
        y_true=label,
        y_pred=pred,
        alpha=alpha,
        num_class=num_class,
    )
    return grad, hess
