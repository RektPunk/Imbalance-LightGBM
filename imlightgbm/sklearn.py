from typing import Any, Callable, Literal

import numpy as np
from lightgbm.sklearn import LGBMClassifier, _LGBM_ScikitMatrixLike

from imlightgbm.docstring import add_docstring
from imlightgbm.objective import (
    _sigmoid,
    sklearn_binary_focal_objective,
    sklearn_binary_weighted_objective,
)

_Objective = Literal["binary_focal", "binary_weighted"]


class ImbalancedLGBMClassifier(LGBMClassifier):
    """Inbalanced LightGBM classifier."""

    @add_docstring("classifier")
    def __init__(
        self,
        *,
        objective: _Objective,
        alpha: float = 0.25,
        gamma: float = 2.0,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        class_weight: dict | str | None = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        n_jobs: int | None = None,
        importance_type: str = "split",
    ) -> None:
        """Construct a gradient boosting model.

        Parameters
        ----------
        objective : str
            Specify the learning objective. Options are 'binary_focal' and 'binary_weighted'.
        alpha: float
        gamma: float
        Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more other parameters.
        """
        self.alpha = alpha
        self.gamma = gamma
        _OBJECTIVE_MAPPER: dict[
            str, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
        ] = {
            "binary_focal": lambda y_true, y_pred: sklearn_binary_focal_objective(
                y_true, y_pred, gamma=gamma
            ),
            "binary_weighted": lambda y_true, y_pred: sklearn_binary_weighted_objective(
                y_true, y_pred, alpha=alpha
            ),
        }
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=_OBJECTIVE_MAPPER[objective],
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
        )

    def predict(
        self,
        X: _LGBM_ScikitMatrixLike,
        *,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ):
        """Docstring is inherited from the LGBMClassifier."""
        result = super().predict(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )
        if raw_score or pred_leaf or pred_contrib:
            return result

        if self._LGBMClassifier__is_multiclass:  # TODO: multiclass
            class_index = np.argmax(result, axis=1)
            return self._LGBMClassifier_le.inverse_transform(class_index)
        else:
            return _sigmoid(result)

    predict.__doc__ = LGBMClassifier.predict.__doc__
