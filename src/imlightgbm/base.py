from enum import StrEnum
from typing import Any

ALPHA_DEFAULT: float = 0.25
GAMMA_DEFAULT: float = 2.0


class SupportedTask(StrEnum):
    binary = "binary"
    multiclass = "multiclass"


class Objective(StrEnum):
    binary_focal = "binary_focal"
    binary_weighted = "binary_weighted"
    multiclass_focal = "multiclass_focal"
    multiclass_weighted = "multiclass_weighted"


class Metric(StrEnum):
    auc = "auc"
    binary_logloss = "binary_logloss"
    binary_error = "binary_error"
    auc_mu = "auc_mu"
    multi_logloss = "multi_logloss"
    multi_error = "multi_error"


def validate_positive_number(param: Any) -> None:
    """Validate positive number."""
    if not isinstance(param, int | float):
        raise ValueError(
            f"Expected a numeric type for parameter, but got {type(param).__name__}."
        )
    if param < 0:
        raise ValueError(f"Expected a positive number for parameter, but got {param}.")
