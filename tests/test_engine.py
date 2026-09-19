from functools import partial
from typing import cast

import lightgbm as lgb
import numpy as np
import pytest
from scipy.special import expit, softmax

from imlightgbm.engine import (
    ImbalancedBooster,
    cv,
    select_metric,
    select_objective,
    set_params,
    train,
)


class TestSelectMetric:
    def test_explicit_metric_returned_as_is(self):
        assert select_metric("binary_focal", "auc") == "auc"
        assert select_metric("multiclass_focal", "multi_error") == "multi_error"
        assert select_metric("binary_focal", ["binary_logloss", "auc"]) == [
            "binary_logloss",
            "auc",
        ]

    def test_default_metric_for_binary(self):
        assert select_metric("binary_focal", None) == "binary_logloss"
        assert select_metric("binary_weighted", None) == "binary_logloss"

    def test_default_metric_for_multiclass_and_others(self):
        assert select_metric("multiclass_focal", None) == "multi_logloss"
        assert select_metric("multiclass_weighted", None) == "multi_logloss"
        assert select_metric("custom_objective", None) == "multi_logloss"


class TestSelectObjective:
    def test_select_objective_returns_partial_with_correct_kwargs(self):
        fobj_bf = select_objective("binary_focal", alpha=0.5, gamma=1.5, num_class=2)
        assert isinstance(fobj_bf, partial)
        assert fobj_bf.keywords == {"gamma": 1.5}

        fobj_bw = select_objective("binary_weighted", alpha=0.7, gamma=1.5, num_class=2)
        assert isinstance(fobj_bw, partial)
        assert fobj_bw.keywords == {"alpha": 0.7}

        fobj_mf = select_objective(
            "multiclass_focal", alpha=0.5, gamma=2.5, num_class=4
        )
        assert isinstance(fobj_mf, partial)
        assert fobj_mf.keywords == {"gamma": 2.5, "num_class": 4}

        fobj_mw = select_objective(
            "multiclass_weighted", alpha=0.3, gamma=1.5, num_class=5
        )
        assert isinstance(fobj_mw, partial)
        assert fobj_mw.keywords == {"alpha": 0.3, "num_class": 5}

    def test_unknown_objective_raises_key_error(self):
        with pytest.raises(KeyError):
            select_objective("unsupported", alpha=0.25, gamma=2.0, num_class=2)


class TestSetParams:
    def test_missing_objective_raises_value_error(self):
        with pytest.raises(ValueError, match="objective must be included in params."):
            set_params({"learning_rate": 0.1})

    def test_callable_metric_raises_value_error(self):
        with pytest.raises(ValueError, match="custom metric are not supported."):
            set_params({"objective": "binary_focal", "metric": lambda x: x})

        with pytest.raises(ValueError, match="custom metric are not supported."):
            set_params(
                {"objective": "binary_focal", "metric": ["binary_logloss", lambda x: x]}
            )

    def test_invalid_metric_type_raises_type_error(self):
        with pytest.raises(
            TypeError, match="metric must be a string or collection of strings"
        ):
            set_params({"objective": "binary_focal", "metric": 123})

    def test_metric_collection_support(self):
        res_list = set_params(
            {"objective": "binary_focal", "metric": ["binary_logloss", "auc"]}
        )
        assert res_list["metric"] == ["binary_logloss", "auc"]

        res_tuple = set_params(
            {"objective": "binary_focal", "metric": ("binary_logloss", "auc")}
        )
        assert res_tuple["metric"] == ("binary_logloss", "auc")

    def test_input_params_not_mutated(self):
        orig_params = {
            "objective": "binary_focal",
            "alpha": 0.5,
            "gamma": 1.5,
            "metric": "auc",
            "verbose": -1,
        }
        params_copy = dict(orig_params)

        result = set_params(orig_params)

        assert orig_params == params_copy
        assert "alpha" not in result
        assert "gamma" not in result
        assert callable(result["objective"])
        assert result["metric"] == "auc"

    def test_set_params_applies_defaults_when_omitted(self):
        params = {"objective": "binary_weighted"}
        result = set_params(params)

        assert callable(result["objective"])
        assert result["metric"] == "binary_logloss"


class TestImbalancedBooster:
    @pytest.fixture
    def dummy_booster(self):
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randint(0, 2, size=60)
        ds = lgb.Dataset(X, y)
        bst = train(
            {
                "objective": "binary_focal",
                "verbose": -1,
                "min_data_in_leaf": 5,
                "min_data_in_bin": 5,
            },
            ds,
            num_boost_round=3,
        )
        return bst, X

    def test_predict_transforms_1d_with_expit(self, dummy_booster):
        bst, X = dummy_booster
        preds = bst.predict(X)
        raw_preds = bst.predict(X, raw_score=True)

        assert preds.ndim == 1
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
        np.testing.assert_allclose(preds, expit(raw_preds))

    def test_predict_bypasses_transformation_on_flags(self, dummy_booster):
        bst, X = dummy_booster

        raw = bst.predict(X, raw_score=True)
        leaf = bst.predict(X, pred_leaf=True)
        contrib = bst.predict(X, pred_contrib=True)

        assert raw is not None
        assert leaf.dtype == np.int32
        assert contrib.shape[1] == X.shape[1] + 1

    def test_predict_transforms_2d_with_softmax(self):
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randint(0, 3, size=60)
        ds = lgb.Dataset(X, y)
        bst = train(
            {
                "objective": "multiclass_focal",
                "num_class": 3,
                "verbose": -1,
                "min_data_in_leaf": 5,
                "min_data_in_bin": 5,
            },
            ds,
            num_boost_round=3,
        )

        preds = cast(np.ndarray, bst.predict(X))
        raw_preds = cast(np.ndarray, bst.predict(X, raw_score=True))

        assert preds.shape == (60, 3)
        np.testing.assert_allclose(np.sum(preds, axis=1), np.ones(60), rtol=1e-5)
        np.testing.assert_allclose(preds, softmax(raw_preds, axis=1), rtol=1e-5)


class TestEngineTrainAndCv:
    @pytest.mark.parametrize(
        ("objective", "kwargs"),
        [
            ("binary_focal", {"gamma": 1.5}),
            ("binary_weighted", {"alpha": 0.5}),
        ],
    )
    def test_train_binary_objectives(self, objective, kwargs):
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randint(0, 2, size=60)
        ds = lgb.Dataset(X, y)

        params = {
            "objective": objective,
            "verbose": -1,
            "min_data_in_leaf": 5,
            "min_data_in_bin": 5,
            **kwargs,
        }
        bst = train(params, ds, num_boost_round=3)

        assert isinstance(bst, ImbalancedBooster)
        preds = cast(np.ndarray, bst.predict(X))
        assert preds.shape == (60,)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    @pytest.mark.parametrize(
        ("objective", "kwargs"),
        [
            ("multiclass_focal", {"gamma": 2.0, "num_class": 3}),
            ("multiclass_weighted", {"alpha": 0.4, "num_class": 3}),
        ],
    )
    def test_train_multiclass_objectives(self, objective, kwargs):
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randint(0, 3, size=60)
        ds = lgb.Dataset(X, y)

        params = {
            "objective": objective,
            "verbose": -1,
            "min_data_in_leaf": 5,
            "min_data_in_bin": 5,
            **kwargs,
        }
        bst = train(params, ds, num_boost_round=3)

        assert isinstance(bst, ImbalancedBooster)
        preds = cast(np.ndarray, bst.predict(X))
        assert preds.shape == (60, 3)
        np.testing.assert_allclose(np.sum(preds, axis=1), np.ones(60), rtol=1e-5)

    def test_train_with_multiple_metrics(self):
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randint(0, 2, size=60)
        ds = lgb.Dataset(X, y)

        params = {
            "objective": "binary_focal",
            "metric": ["binary_logloss", "auc"],
            "verbose": -1,
            "min_data_in_leaf": 5,
            "min_data_in_bin": 5,
        }
        bst = train(params, ds, num_boost_round=3)
        assert isinstance(bst, ImbalancedBooster)

    def test_cv_execution(self):
        np.random.seed(42)
        X = np.random.randn(60, 4)
        y = np.random.randint(0, 2, size=60)
        ds = lgb.Dataset(X, y)

        params = {
            "objective": "binary_focal",
            "verbose": -1,
            "min_data_in_leaf": 5,
            "min_data_in_bin": 5,
        }
        cv_res = cv(params, ds, num_boost_round=3, nfold=2, stratified=False)

        assert isinstance(cv_res, dict)
        assert any("logloss" in k for k in cv_res)
