from typing import cast

import numpy as np
import pytest
from sklearn.base import clone

from imlightgbm.sklearn import ImbalancedLGBMClassifier


class TestImbalancedLGBMClassifierInit:
    def test_multiclass_requires_num_class(self):
        with pytest.raises(ValueError, match="num_class must be provided"):
            ImbalancedLGBMClassifier(objective="multiclass_focal", num_class=None)

        with pytest.raises(ValueError, match="num_class must be provided"):
            ImbalancedLGBMClassifier(objective="multiclass_weighted", num_class=None)

    def test_binary_does_not_require_num_class(self):
        clf_focal = ImbalancedLGBMClassifier(objective="binary_focal")
        assert clf_focal.num_class is None
        assert clf_focal.gamma is None
        assert clf_focal.alpha is None

        clf_weighted = ImbalancedLGBMClassifier(objective="binary_weighted")
        assert clf_weighted.num_class is None
        assert clf_weighted.alpha is None
        assert clf_weighted.gamma is None

    def test_custom_alpha_gamma_assigned(self):
        clf = ImbalancedLGBMClassifier(
            objective="binary_weighted",
            alpha=0.75,
        )
        assert clf.alpha == 0.75

        clf_focal = ImbalancedLGBMClassifier(
            objective="binary_focal",
            gamma=3.5,
        )
        assert clf_focal.gamma == 3.5

    def test_sklearn_clone_compatibility(self):
        clf = ImbalancedLGBMClassifier(
            objective="binary_focal",
            gamma=1.5,
            n_estimators=10,
        )
        cloned_clf = clone(clf)
        assert cloned_clf.objective == "binary_focal"
        assert cloned_clf.gamma == 1.5  # type: ignore
        assert cloned_clf.n_estimators == 10  # type: ignore
        assert cloned_clf.alpha is None  # type: ignore

    def test_unsupported_objective_raises_key_error_on_fit(self):
        clf = ImbalancedLGBMClassifier(objective="invalid_objective")
        with pytest.raises(KeyError):
            clf.fit(np.random.randn(10, 2), np.random.randint(0, 2, size=10))

    def test_process_params_removes_alpha_and_gamma(self):
        clf = ImbalancedLGBMClassifier(
            objective="binary_focal",
            gamma=1.5,
            alpha=0.5,
        )
        processed = clf._process_params("fit")
        assert "alpha" not in processed
        assert "gamma" not in processed


class TestImbalancedLGBMClassifierFitPredict:
    @pytest.fixture
    def binary_data(self):
        np.random.seed(42)
        X = np.random.randn(80, 4)
        y = np.random.randint(0, 2, size=80)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        np.random.seed(42)
        X = np.random.randn(90, 4)
        y = np.random.randint(0, 3, size=90)
        return X, y

    @pytest.mark.parametrize("objective", ["binary_focal", "binary_weighted"])
    def test_binary_fit_and_predict(self, binary_data, objective):
        X, y = binary_data
        clf = ImbalancedLGBMClassifier(
            objective=objective,
            n_estimators=5,
            min_child_samples=5,
            verbose=-1,
        )
        clf.fit(X, y)

        preds = cast(np.ndarray, clf.predict(X))
        assert preds.shape == (80,)
        assert set(np.unique(preds)).issubset({0, 1})

    @pytest.mark.parametrize("objective", ["binary_focal", "binary_weighted"])
    def test_binary_predict_proba(self, binary_data, objective):
        X, y = binary_data
        clf = ImbalancedLGBMClassifier(
            objective=objective,
            n_estimators=5,
            min_child_samples=5,
            verbose=-1,
        )
        clf.fit(X, y)

        proba = cast(np.ndarray, clf.predict_proba(X))
        assert proba.shape == (80, 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(80), rtol=1e-5)

    @pytest.mark.parametrize("objective", ["multiclass_focal", "multiclass_weighted"])
    def test_multiclass_fit_and_predict(self, multiclass_data, objective):
        X, y = multiclass_data
        clf = ImbalancedLGBMClassifier(
            objective=objective,
            num_class=3,
            n_estimators=5,
            min_child_samples=5,
            verbose=-1,
        )
        clf.fit(X, y)

        preds = cast(np.ndarray, clf.predict(X))
        assert preds.shape == (90,)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    @pytest.mark.parametrize("objective", ["multiclass_focal", "multiclass_weighted"])
    def test_multiclass_predict_proba(self, multiclass_data, objective):
        X, y = multiclass_data
        clf = ImbalancedLGBMClassifier(
            objective=objective,
            num_class=3,
            n_estimators=5,
            min_child_samples=5,
            verbose=-1,
        )
        clf.fit(X, y)

        proba = cast(np.ndarray, clf.predict_proba(X))
        assert proba.shape == (90, 3)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(90), rtol=1e-5)

    def test_predict_kwargs_passthrough(self, binary_data):
        X, y = binary_data
        clf = ImbalancedLGBMClassifier(
            objective="binary_focal",
            n_estimators=5,
            min_child_samples=5,
            verbose=-1,
        )
        clf.fit(X, y)

        raw = cast(np.ndarray, clf.predict(X, raw_score=True))
        leaf = cast(np.ndarray, clf.predict(X, pred_leaf=True))
        contrib = cast(np.ndarray, clf.predict(X, pred_contrib=True))

        assert raw.shape == (80,)
        assert leaf.shape == (80, 5)
        assert contrib.shape == (80, X.shape[1] + 1)
