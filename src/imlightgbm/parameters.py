ALPHA_DEFAULT: float = 0.25
GAMMA_DEFAULT: float = 2.0


def select_alpha(objective: str, alpha: float | None) -> float:
    if objective not in {"binary_weighted", "multiclass_weighted"} or alpha is None:
        return ALPHA_DEFAULT

    if alpha <= 0.0:
        raise ValueError(f"Expected a positive number for alpha, but got {alpha}.")

    return alpha


def select_gamma(objective: str, gamma: float | None) -> float:
    if objective not in {"binary_focal", "multiclass_focal"} or gamma is None:
        return GAMMA_DEFAULT

    if gamma <= 0.0:
        raise ValueError(f"Expected a positive number for gamma, but got {gamma}.")

    return gamma
