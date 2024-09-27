from enum import Enum

ALPHA_DEFAULT: float = 0.25
GAMMA_DEFAULT: float = 2.0


class BaseEnum(str, Enum):
    @classmethod
    def get(cls, text: str) -> Enum:
        cls.__check_valid(text)
        return cls[text]

    @classmethod
    def __check_valid(cls, text: str) -> None:
        if text not in cls._member_map_.keys():
            valid_members = ", ".join(list(cls._member_map_.keys()))
            raise ValueError(
                f"Invalid value: '{text}'. Expected one of: {valid_members}."
            )


class SupportedTask(BaseEnum):
    binary: str = "binary"
    multiclass: str = "multiclass"


class Objective(BaseEnum):
    binary_focal: str = "binary_focal"
    binary_weighted: str = "binary_weighted"
    multiclass_focal: str = "multiclass_focal"
    multiclass_weighted: str = "multiclass_weighted"


class Metric(BaseEnum):
    auc: str = "auc"
    binary_logloss: str = "binary_logloss"
    binary_error: str = "binary_error"
    auc_mu: str = "auc_mu"
    multi_logloss: str = "multi_logloss"
    multi_error: str = "multi_error"
