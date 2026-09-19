import lightgbm as lgb
import numpy as np
import pytest
from scipy.special import expit, softmax

from imlightgbm.objective import (
    _safe_log,
    _safe_power,
    binary_focal_lgb_objective,
    binary_focal_objective,
    binary_weighted_lgb_objective,
    binary_weighted_objective,
    multiclass_focal_lgb_objective,
    multiclass_focal_objective,
    multiclass_weighted_lgb_objective,
    multiclass_weighted_objective,
)


class TestHelpers:
    def test_safe_power_positive(self):
        arr = np.array([1.0, 4.0, 9.0])
        res = _safe_power(arr, 0.5)
        np.testing.assert_allclose(res, [1.0, 2.0, 3.0])

    def test_safe_power_negative(self):
        arr = np.array([-4.0, -9.0])
        res = _safe_power(arr, 0.5)
        np.testing.assert_allclose(res, [-2.0, -3.0])

    def test_safe_power_zero(self):
        arr = np.array([0.0])
        res = _safe_power(arr, 2.0)
        np.testing.assert_allclose(res, [0.0])

    def test_safe_log_values(self):
        arr = np.array([1.0, np.e, np.e**2])
        res = _safe_log(arr)
        np.testing.assert_allclose(res, [0.0, 1.0, 2.0])

    def test_safe_log_clipping(self):
        arr = np.array([0.0, -5.0])
        res = _safe_log(arr, min_value=1e-6)
        expected = np.log(1e-6)
        np.testing.assert_allclose(res, [expected, expected])


class TestBinaryWeightedObjective:
    def test_hand_calculated_values(self):
        alpha = 0.5
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 0.0])

        grad, hess = binary_weighted_objective(y_true, y_pred, alpha=alpha)

        expected_grad = np.array([-0.25, 0.5])
        expected_hess = np.array([0.125, 0.25])

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(hess, expected_hess, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("y", [0, 1])
    @pytest.mark.parametrize("z", [-2.5, -1.0, 0.0, 1.0, 2.5])
    def test_numerical_derivative_comparison(self, alpha: float, y: int, z: float):
        y_true = np.array([y], dtype=np.float64)
        y_pred = np.array([z], dtype=np.float64)

        grad, hess = binary_weighted_objective(y_true, y_pred, alpha=alpha)

        def weighted_loss(val_z: float) -> float:
            p = expit(val_z)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            weight = alpha**y
            return -weight * (y * np.log(p) + (1 - y) * np.log(1 - p))

        eps = 1e-5
        l_plus = weighted_loss(z + eps)
        l_minus = weighted_loss(z - eps)
        l_zero = weighted_loss(z)

        num_grad = (l_plus - l_minus) / (2 * eps)
        num_hess = (l_plus - 2 * l_zero + l_minus) / (eps**2)

        np.testing.assert_allclose(grad[0], num_grad, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hess[0], num_hess, rtol=1e-4, atol=1e-4)


class TestBinaryFocalObjective:
    def test_hand_calculated_values(self):
        gamma = 2.0
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 0.0])

        grad, hess = binary_focal_objective(y_true, y_pred, gamma=gamma)

        expected_grad_y1 = 0.25 * np.log(0.5) - 0.125
        expected_grad_y0 = -0.25 * np.log(0.5) + 0.125

        np.testing.assert_allclose(grad[0], expected_grad_y1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad[1], expected_grad_y0, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("y", [0, 1])
    @pytest.mark.parametrize("z", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_numerical_derivative_comparison(self, gamma: float, y: int, z: float):
        y_true = np.array([y], dtype=np.float64)
        y_pred = np.array([z], dtype=np.float64)

        grad, hess = binary_focal_objective(y_true, y_pred, gamma=gamma)

        def focal_loss(val_z: float) -> float:
            p = expit(val_z)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            if y == 1:
                return -((1.0 - p) ** gamma) * np.log(p)
            else:
                return -(p**gamma) * np.log(1.0 - p)

        eps = 1e-5
        l_plus = focal_loss(z + eps)
        l_minus = focal_loss(z - eps)
        l_zero = focal_loss(z)

        num_grad = (l_plus - l_minus) / (2 * eps)
        num_hess = (l_plus - 2 * l_zero + l_minus) / (eps**2)

        np.testing.assert_allclose(grad[0], num_grad, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hess[0], num_hess, rtol=1e-4, atol=1e-4)


class TestMulticlassObjectives:
    def test_multiclass_weighted_shape_and_values(self):
        num_class = 3
        alpha = 0.5
        y_true = np.array([0, 2], dtype=np.int32)
        y_pred = np.array([[1.0, 0.0, -1.0], [0.5, 1.5, 2.0]])

        grad, hess = multiclass_weighted_objective(
            y_true=y_true,
            y_pred=y_pred,
            alpha=alpha,
            num_class=num_class,
        )

        probs = softmax(y_pred, axis=1)
        y_onehot = np.eye(num_class)[y_true]

        expected_grad = -(alpha**y_onehot) * (y_onehot - probs)
        expected_hess = (alpha**y_onehot) * probs * (1.0 - probs)

        assert grad.shape == (2, 3)
        assert hess.shape == (2, 3)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-7)
        np.testing.assert_allclose(hess, expected_hess, rtol=1e-7)

    def test_multiclass_focal_shape_and_values(self):
        num_class = 3
        gamma = 2.0
        y_true = np.array([1, 0], dtype=np.int32)
        y_pred = np.array([[0.0, 1.0, 0.0], [2.0, 0.5, -0.5]])

        grad, hess = multiclass_focal_objective(
            y_true=y_true,
            y_pred=y_pred,
            gamma=gamma,
            num_class=num_class,
        )

        assert grad.shape == (2, 3)
        assert hess.shape == (2, 3)
        probs = softmax(y_pred, axis=1)
        y_onehot = np.eye(num_class)[y_true]

        for i in range(2):
            for c in range(num_class):
                y_val = y_onehot[i, c]
                p_val = probs[i, c]

                if y_val == 1:
                    exp_g = gamma * p_val * ((1 - p_val) ** gamma) * np.log(p_val) - (
                        (1 - p_val) ** (gamma + 1)
                    )
                else:
                    exp_g = -gamma * (1 - p_val) * (p_val**gamma) * np.log(
                        1 - p_val
                    ) + (p_val ** (gamma + 1))

                np.testing.assert_allclose(grad[i, c], exp_g, rtol=1e-5, atol=1e-5)


class TestLgbDatasetWrappers:
    def test_binary_focal_lgb_objective(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1, 0])
        dataset = lgb.Dataset(X, label=y)
        y_pred = np.array([0.5, -0.5])

        grad_wrapper, hess_wrapper = binary_focal_lgb_objective(
            pred=y_pred, train_data=dataset, gamma=2.0
        )
        grad_direct, hess_direct = binary_focal_objective(
            y_true=y, y_pred=y_pred, gamma=2.0
        )

        np.testing.assert_allclose(grad_wrapper, grad_direct)
        np.testing.assert_allclose(hess_wrapper, hess_direct)

    def test_binary_weighted_lgb_objective(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1, 0])
        dataset = lgb.Dataset(X, label=y)
        y_pred = np.array([0.5, -0.5])

        grad_wrapper, hess_wrapper = binary_weighted_lgb_objective(
            pred=y_pred, train_data=dataset, alpha=0.5
        )
        grad_direct, hess_direct = binary_weighted_objective(
            y_true=y, y_pred=y_pred, alpha=0.5
        )

        np.testing.assert_allclose(grad_wrapper, grad_direct)
        np.testing.assert_allclose(hess_wrapper, hess_direct)

    def test_multiclass_focal_lgb_objective(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 2])
        dataset = lgb.Dataset(X, label=y)
        y_pred = np.array([[0.5, -0.5, 0.0], [1.0, 0.0, -1.0]])

        grad_wrapper, hess_wrapper = multiclass_focal_lgb_objective(
            pred=y_pred, train_data=dataset, gamma=2.0, num_class=3
        )
        grad_direct, hess_direct = multiclass_focal_objective(
            y_true=y, y_pred=y_pred, gamma=2.0, num_class=3
        )

        np.testing.assert_allclose(grad_wrapper, grad_direct)
        np.testing.assert_allclose(hess_wrapper, hess_direct)

    def test_multiclass_weighted_lgb_objective(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1, 2])
        dataset = lgb.Dataset(X, label=y)
        y_pred = np.array([[0.5, -0.5, 0.0], [1.0, 0.0, -1.0]])

        grad_wrapper, hess_wrapper = multiclass_weighted_lgb_objective(
            pred=y_pred, train_data=dataset, alpha=0.5, num_class=3
        )
        grad_direct, hess_direct = multiclass_weighted_objective(
            y_true=y, y_pred=y_pred, alpha=0.5, num_class=3
        )

        np.testing.assert_allclose(grad_wrapper, grad_direct)
        np.testing.assert_allclose(hess_wrapper, hess_direct)
