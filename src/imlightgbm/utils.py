from typing import Any


def validate_positive_number(param: Any) -> None:
    """Validate positive number."""
    if not isinstance(param, int | float):
        raise ValueError(
            f"Expected a numeric type for parameter, but got {type(param).__name__}."
        )
    if param < 0:
        raise ValueError(f"Expected a positive number for parameter, but got {param}.")
