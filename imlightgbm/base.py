from enum import Enum


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


class Metric(BaseEnum):
    auc: str = "auc"
    binary_logloss: str = "binary_logloss"
    binary_error: str = "binary_error"
    auc_mu: str = "auc_mu"
    multi_logloss: str = "multi_logloss"
    multi_error: str = "multi_error"


class Objective(BaseEnum):
    focal: str = "focal"
    weighted: str = "weighted"
