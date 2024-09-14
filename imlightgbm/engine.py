from functools import partial
from typing import Any, Callable, Literal

import lightgbm as lgb
from sklearn.utils.multiclass import type_of_target

from imlightgbm.objective import (
    binary_focal_eval,
    binary_focal_objective,
    multiclass_focal_eval,
    multiclass_focal_objective,
)
from imlightgbm.utils import logger, modify_docstring


def train(
    params: dict[str, Any],
    train_set: lgb.Dataset,
    valid_sets: list[lgb.Dataset] = None,
    valid_names: list[str] = None,
    num_boost_round: int = 100,
    init_model: str | lgb.Path | lgb.Booster | None = None,
    feature_name: list[str] | Literal["auto"] = "auto",
    categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
    keep_training_booster: bool = False,
    callbacks: list[Callable] | None = None,
) -> lgb.Booster:
    if "objective" in params:
        logger.warning("'objective' exists in params will not used.")
        params.pop("objective")

    params.setdefault("alpha", 0.05)
    params.setdefault("gamma", 0.01)

    inferred_task = type_of_target(train_set.get_label())
    if inferred_task not in {"binary", "multiclass"}:
        raise ValueError(
            f"Invalid target type: {inferred_task}. Supported types are 'binary' or 'multiclass'."
        )

    eval_mapper = {
        "binary": partial(
            binary_focal_eval, alpha=params["alpha"], gamma=params["gamma"]
        ),
        "multiclass": partial(
            multiclass_focal_eval, alpha=params["alpha"], gamma=params["gamma"]
        ),
    }
    objective_mapper = {
        "binary": partial(
            binary_focal_objective, alpha=params["alpha"], gamma=params["gamma"]
        ),
        "multiclass": partial(
            multiclass_focal_objective, alpha=params["alpha"], gamma=params["gamma"]
        ),
    }

    fobj = objective_mapper.get(inferred_task)
    feval = eval_mapper.get(inferred_task)
    params.update({"objective": fobj})

    return lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        num_boost_round=num_boost_round,
        init_model=init_model,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
    )


train.__doc__ = modify_docstring(lgb.train.__doc__)
