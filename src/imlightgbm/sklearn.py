from collections.abc import Callable

import numpy as np
from lightgbm.sklearn import LGBMClassifier
from scipy.sparse import spmatrix
from scipy.special import expit

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

        self.num_class = num_class
        self.alpha = select_alpha(objective, alpha)
        self.gamma = select_gamma(objective, gamma)

        if (
            objective in {"multiclass_focal", "multiclass_weighted"}
            and num_class is None
        ):
            raise ValueError("num_class must be provided")

        super().__init__(
            objective=self.__objective_select(
                objective,
                self.alpha,
                self.gamma,
                self.num_class if isinstance(self.num_class, int) else -1,
            ),
            **kwargs,
        )

    def predict(self, *args, **kwargs) -> np.ndarray | spmatrix | list[spmatrix]:
        """"""
        _predict = super().predict(*args, **kwargs)
        if (
            kwargs.get("raw_score", False)
            or kwargs.get("pred_leaf", False)
            or kwargs.get("pred_contrib", False)
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if _predict.ndim == 1:
            return expit(_predict)

        return self._le.inverse_transform(np.argmax(_predict, axis=1))

    def __objective_select(
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
        params = super()._process_params(stage)
        params.pop("alpha", None)
        params.pop("gamma", None)
        return params
