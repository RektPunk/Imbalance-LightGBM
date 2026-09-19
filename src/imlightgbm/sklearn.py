from collections.abc import Callable

import numpy as np
from lightgbm.sklearn import LGBMClassifier, LGBMModel
from scipy.sparse import spmatrix
from scipy.special import expit, softmax

from imlightgbm.objective import (
    binary_focal_objective,
    binary_weighted_objective,
    multiclass_focal_objective,
    multiclass_weighted_objective,
)
from imlightgbm.parameters import select_alpha, select_gamma


class ImbalancedLGBMClassifier(LGBMClassifier):
    """Imbalanced LightGBM classifier."""

    def __init__(
        self,
        *,
        objective: str,
        alpha: float | None = None,
        gamma: float | None = None,
        num_class: int | None = None,
        **kwargs,
    ) -> None:
        """Construct a gradient boosting model."""
        self.objective = objective
        self.alpha = alpha
        self.gamma = gamma
        self.num_class = num_class

        if (
            objective in {"multiclass_focal", "multiclass_weighted"}
            and num_class is None
        ):
            raise ValueError("num_class must be provided")

        super().__init__(
            objective=objective,
            **kwargs,
        )

    def predict(self, *args, **kwargs) -> np.ndarray | spmatrix | list[spmatrix]:
        """Predict class labels."""
        _predict = LGBMModel.predict(self, *args, **kwargs)
        if (
            kwargs.get("raw_score", False)
            or kwargs.get("pred_leaf", False)
            or kwargs.get("pred_contrib", False)
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if _predict.ndim == 1:
            class_indices = (expit(_predict) >= 0.5).astype(int)
            return self._le.inverse_transform(class_indices)

        return self._le.inverse_transform(np.argmax(_predict, axis=1))

    def predict_proba(self, *args, **kwargs) -> np.ndarray | spmatrix | list[spmatrix]:
        """Predict class probabilities."""
        _predict = LGBMModel.predict(self, *args, raw_score=True, **kwargs)
        if (
            kwargs.get("pred_leaf", False)
            or kwargs.get("pred_contrib", False)
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if _predict.ndim == 1:
            p1 = expit(_predict)
            return np.column_stack([1.0 - p1, p1])

        return softmax(_predict, axis=1)

    def _objective_select(
        self,
        objective: str,
        alpha: float,
        gamma: float,
        num_class: int,
    ) -> Callable:
        """Select objective function."""
        _objective_mapper: dict[str, Callable] = {
            "binary_focal": lambda y_true, y_pred: binary_focal_objective(
                y_true=y_true, y_pred=y_pred, gamma=gamma
            ),
            "binary_weighted": lambda y_true, y_pred: binary_weighted_objective(
                y_true=y_true, y_pred=y_pred, alpha=alpha
            ),
            "multiclass_focal": lambda y_true, y_pred: multiclass_focal_objective(
                y_true=y_true,
                y_pred=y_pred,
                gamma=gamma,
                num_class=num_class,
            ),
            "multiclass_weighted": lambda y_true, y_pred: multiclass_weighted_objective(
                y_true=y_true,
                y_pred=y_pred,
                alpha=alpha,
                num_class=num_class,
            ),
        }
        return _objective_mapper[objective]

    def _process_params(self, stage: str) -> dict:
        _alpha = select_alpha(self.objective, self.alpha)
        _gamma = select_gamma(self.objective, self.gamma)
        num_class = self.num_class if isinstance(self.num_class, int) else -1

        if stage == "fit":
            self._objective = self._objective_select(
                self.objective,
                _alpha,
                _gamma,
                num_class,
            )

        params = super()._process_params(stage)
        params.pop("alpha", None)
        params.pop("gamma", None)
        return params
