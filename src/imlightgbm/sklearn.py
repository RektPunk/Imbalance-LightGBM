from collections.abc import Callable
from typing import Any

import numpy as np
from lightgbm.sklearn import LGBMClassifier, _LGBM_ScikitMatrixLike
from scipy.sparse import spmatrix
from scipy.special import expit

from imlightgbm.base import ALPHA_DEFAULT, GAMMA_DEFAULT, Objective
from imlightgbm.objective.core import (
    sklearn_binary_focal_objective,
    sklearn_binary_weighted_objective,
    sklearn_multiclass_focal_objective,
    sklearn_multiclass_weighted_objective,
)
from imlightgbm.utils import validate_positive_number

_SklearnObjLike = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


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
        _objective_enum: Objective = Objective[objective]
        self.__alpha_select(objective=_objective_enum, alpha=alpha)
        self.__gamma_select(objective=_objective_enum, gamma=gamma)
        _objective = self.__objective_select(objective_enum=_objective_enum)
        super().__init__(
            objective=_objective,
            **kwargs,
        )

    def predict(
        self,
        X: _LGBM_ScikitMatrixLike,
        **kwargs,
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        _predict = super().predict(X=X, **kwargs)
        if (
            kwargs.get("raw_score", False)
            or kwargs.get("pred_leaf", False)
            or kwargs.get("pred_contrib", False)
            or isinstance(_predict, spmatrix | list)
        ):
            return _predict

        if self._LGBMClassifier__is_multiclass:
            class_index = np.argmax(_predict, axis=1)
            return self._le.inverse_transform(class_index)
        else:
            return expit(_predict)

    predict.__doc__ = LGBMClassifier.predict.__doc__

    def __objective_select(self, objective_enum: Objective) -> _SklearnObjLike:
        """Select objective function."""
        if objective_enum in {
            Objective.multiclass_focal,
            Objective.multiclass_weighted,
        } and not isinstance(self.num_class, int):
            raise ValueError("num_class must be provided")

        _objective_mapper: dict[Objective, _SklearnObjLike] = {
            Objective.binary_focal: lambda y_true, y_pred: (
                sklearn_binary_focal_objective(
                    y_true=y_true, y_pred=y_pred, gamma=self.gamma
                )
            ),
            Objective.binary_weighted: lambda y_true, y_pred: (
                sklearn_binary_weighted_objective(
                    y_true=y_true, y_pred=y_pred, alpha=self.alpha
                )
            ),
            Objective.multiclass_focal: lambda y_true, y_pred: (
                sklearn_multiclass_focal_objective(
                    y_true=y_true,
                    y_pred=y_pred,
                    gamma=self.gamma,
                    num_class=self.num_class,
                )
            ),
            Objective.multiclass_weighted: lambda y_true, y_pred: (
                sklearn_multiclass_weighted_objective(
                    y_true=y_true,
                    y_pred=y_pred,
                    alpha=self.alpha,
                    num_class=self.num_class,
                )
            ),
        }
        return _objective_mapper[objective_enum]

    def __param_select(
        self,
        objective: Objective,
        param: float | None,
        valid_objectives: set[Objective],
        default_value: float,
        param_name: str,
    ) -> None:
        """General method to select appropriate parameter (alpha or gamma)."""
        if objective not in valid_objectives:
            setattr(self, param_name, None)
            return
        if param:
            validate_positive_number(param)
            setattr(self, param_name, param)
            return
        setattr(self, param_name, default_value)

    def __alpha_select(self, objective: Objective, alpha: float | None) -> None:
        """Select appropriate alpha."""
        self.__param_select(
            objective=objective,
            param=alpha,
            valid_objectives={Objective.binary_weighted, Objective.multiclass_weighted},
            default_value=ALPHA_DEFAULT,
            param_name="alpha",
        )

    def __gamma_select(self, objective: Objective, gamma: float | None) -> None:
        """Select appropriate gamma."""
        self.__param_select(
            objective=objective,
            param=gamma,
            valid_objectives={Objective.binary_focal, Objective.multiclass_focal},
            default_value=GAMMA_DEFAULT,
            param_name="gamma",
        )
