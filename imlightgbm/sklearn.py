from typing import Any, Callable

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
        num_class: int | None = None,
    ) -> None:
        """Construct a gradient boosting model.
        Parameters
        ----------
        objective : str
            Specify the learning objective.
            Options are 'binary_focal', 'binary_weighted', 'multiclass focal', and 'multiclas_weighted'.
        alpha: float
            Parameter used with 'binary_weighted' and 'multiclass_weighted' objective.
        gamma: float
            Parameter used with 'binary_focal' and 'multiclass_focal' objective.
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.
            **kwargs is not supported in sklearn, it may cause unexpected issues.
        """

        self.num_class = num_class
        _objective_enum: Objective = Objective.get(objective)
        self.__alpha_select(objective=_objective_enum, alpha=alpha)
        self.__gamma_select(objective=_objective_enum, gamma=gamma)
        _objective = self.__objective_select(objective_enum=_objective_enum)
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=_objective,
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
    ) -> np.ndarray | spmatrix | list[spmatrix]:
        _predict = super().predict(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )
        if (
            raw_score
            or pred_leaf
            or pred_contrib
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
            Objective.binary_focal: lambda y_true,
            y_pred: sklearn_binary_focal_objective(
                y_true=y_true, y_pred=y_pred, gamma=self.gamma
            ),
            Objective.binary_weighted: lambda y_true,
            y_pred: sklearn_binary_weighted_objective(
                y_true=y_true, y_pred=y_pred, alpha=self.alpha
            ),
            Objective.multiclass_focal: lambda y_true,
            y_pred: sklearn_multiclass_focal_objective(
                y_true=y_true, y_pred=y_pred, gamma=self.gamma, num_class=self.num_class
            ),
            Objective.multiclass_weighted: lambda y_true,
            y_pred: sklearn_multiclass_weighted_objective(
                y_true=y_true, y_pred=y_pred, alpha=self.alpha, num_class=self.num_class
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
