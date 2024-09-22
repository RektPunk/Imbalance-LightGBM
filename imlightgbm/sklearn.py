from typing import Callable, Literal

import numpy as np
from lightgbm import LGBMClassifier

from imlightgbm.objective import (
    sklearn_binary_focal_objective,
    sklearn_binary_weighted_objective,
)

_Objective = Literal["binary_focal", "binary_weighted"]


class ImbalancedLGBMClassifier(LGBMClassifier):
    def __init__(
        self,
        objective: _Objective,
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
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
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
