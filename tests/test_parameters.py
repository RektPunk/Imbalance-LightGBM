import pytest

from imlightgbm.parameters import (
    ALPHA_DEFAULT,
    GAMMA_DEFAULT,
    select_alpha,
    select_gamma,
)


class TestSelectAlpha:
    def test_alpha_default_when_none(self):
        assert select_alpha("binary_weighted", None) == ALPHA_DEFAULT
        assert select_alpha("multiclass_weighted", None) == ALPHA_DEFAULT

    def test_alpha_default_for_other_objectives(self):
        assert select_alpha("binary_focal", 0.5) == ALPHA_DEFAULT
        assert select_alpha("multiclass_focal", 0.8) == ALPHA_DEFAULT
        assert select_alpha("unknown_objective", 0.5) == ALPHA_DEFAULT

    def test_alpha_valid_custom_values(self):
        assert select_alpha("binary_weighted", 0.75) == 0.75
        assert select_alpha("multiclass_weighted", 1.5) == 1.5

    def test_alpha_non_positive_raises_value_error(self):
        with pytest.raises(ValueError, match="Expected a positive number for alpha"):
            select_alpha("binary_weighted", 0.0)

        with pytest.raises(ValueError, match="Expected a positive number for alpha"):
            select_alpha("binary_weighted", -0.1)

        with pytest.raises(ValueError, match="Expected a positive number for alpha"):
            select_alpha("multiclass_weighted", -1.0)


class TestSelectGamma:
    def test_gamma_default_when_none(self):
        assert select_gamma("binary_focal", None) == GAMMA_DEFAULT
        assert select_gamma("multiclass_focal", None) == GAMMA_DEFAULT

    def test_gamma_default_for_other_objectives(self):
        assert select_gamma("binary_weighted", 3.0) == GAMMA_DEFAULT
        assert select_gamma("multiclass_weighted", 4.0) == GAMMA_DEFAULT
        assert select_gamma("unknown_objective", 3.0) == GAMMA_DEFAULT

    def test_gamma_valid_custom_values(self):
        assert select_gamma("binary_focal", 1.5) == 1.5
        assert select_gamma("multiclass_focal", 3.0) == 3.0

    def test_gamma_non_positive_raises_value_error(self):
        with pytest.raises(ValueError, match="Expected a positive number for gamma"):
            select_gamma("binary_focal", 0.0)

        with pytest.raises(ValueError, match="Expected a positive number for gamma"):
            select_gamma("binary_focal", -0.5)

        with pytest.raises(ValueError, match="Expected a positive number for gamma"):
            select_gamma("multiclass_focal", -2.0)
