import numpy as np
from lightgbm import Dataset
from scipy.special import expit, softmax


def _safe_power(num_base: np.ndarray, num_pow: float) -> np.ndarray:
    """Safe power."""
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)


def _safe_log(array: np.ndarray, min_value: float = 1e-6) -> np.ndarray:
    """Safe log."""
    return np.log(np.clip(array, min_value, None))


def _weighted_grad_hess(
    y_true: np.ndarray, pred_prob: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted grad hess."""
    grad = -(alpha**y_true) * (y_true - pred_prob)
    hess = (alpha**y_true) * pred_prob * (1.0 - pred_prob)
    return grad, hess


def _focal_grad_hess(
    y_true: np.ndarray, pred_prob: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reurtn focal grad hess."""
    prob_product = pred_prob * (1 - pred_prob)
    true_diff_pred = y_true + ((-1) ** y_true) * pred_prob
    focal_grad_term = pred_prob + y_true - 1
    focal_log_term = 1 - y_true - ((-1) ** y_true) * pred_prob
    focal_grad_base = y_true + ((-1) ** y_true) * pred_prob
    grad = gamma * focal_grad_term * _safe_power(true_diff_pred, gamma) * _safe_log(
        focal_log_term
    ) + ((-1) ** y_true) * _safe_power(focal_grad_base, (gamma + 1))

    hess_term1 = _safe_power(true_diff_pred, gamma) + gamma * (
        (-1) ** y_true
    ) * focal_grad_term * _safe_power(true_diff_pred, (gamma - 1))
    hess_term2 = (
        ((-1) ** y_true)
        * focal_grad_term
        * _safe_power(true_diff_pred, gamma)
        / focal_log_term
    )
    hess = (
        (hess_term1 * _safe_log(focal_log_term) - hess_term2) * gamma
        + (gamma + 1) * _safe_power(focal_grad_base, gamma)
    ) * prob_product
    return grad, hess


def sklearn_binary_focal_objective(
    y_true: np.ndarray, y_pred: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective for sklearn API."""
    pred_prob = expit(y_pred)
    return _focal_grad_hess(y_true=y_true, pred_prob=pred_prob, gamma=gamma)


def sklearn_binary_weighted_objective(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary weighted objective for sklearn API."""
    pred_prob = expit(y_pred)
    return _weighted_grad_hess(y_true=y_true, pred_prob=pred_prob, alpha=alpha)


def binary_focal_objective(
    pred: np.ndarray, train_data: Dataset, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary focal objective for engine."""
    label = train_data.get_label()
    return sklearn_binary_focal_objective(y_true=label, y_pred=pred, gamma=gamma)


def binary_weighted_objective(
    pred: np.ndarray, train_data: Dataset, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for binary weighted objective for engine."""
    label = train_data.get_label()
    return sklearn_binary_weighted_objective(y_true=label, y_pred=pred, alpha=alpha)


def sklearn_multiclass_focal_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective for sklearn API.."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true.astype(int)]
    return _focal_grad_hess(y_true=y_true_onehot, pred_prob=pred_prob, gamma=gamma)


def sklearn_multiclass_weighted_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass weighted objective for sklearn API."""
    pred_prob = softmax(y_pred, axis=1)
    y_true_onehot = np.eye(num_class)[y_true.astype(int)]
    return _weighted_grad_hess(y_true=y_true_onehot, pred_prob=pred_prob, alpha=alpha)


def multiclass_focal_objective(
    pred: np.ndarray,
    train_data: Dataset,
    gamma: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass focal objective for engine."""
    label = train_data.get_label()
    return sklearn_multiclass_focal_objective(
        y_true=label,
        y_pred=pred,
        gamma=gamma,
        num_class=num_class,
    )


def multiclass_weighted_objective(
    pred: np.ndarray,
    train_data: Dataset,
    alpha: float,
    num_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grad, hess for multclass weighted objective for engine."""
    label = train_data.get_label()
    return sklearn_multiclass_weighted_objective(
        y_true=label,
        y_pred=pred,
        alpha=alpha,
        num_class=num_class,
    )
