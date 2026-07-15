import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.explainability import (
    _normalize_shap_output,
    get_top_contributors,
    _get_shap_values,
    explain_anomalies,
)


# ── TestNormalizeShapOutput ───────────────────────────────────────


class TestNormalizeShapOutput:
    def test_list_input(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _normalize_shap_output([arr], (2, 2))
        np.testing.assert_array_equal(result, arr)

    def test_3d_input(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        result = _normalize_shap_output(arr, (1, 2))
        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_1d_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _normalize_shap_output(arr, (3, 1))
        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_2d_passthrough(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _normalize_shap_output(arr, (2, 2))
        np.testing.assert_array_equal(result, arr)

    def test_list_of_3d(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        result = _normalize_shap_output([arr], (3, 2))
        assert result.ndim == 2
        assert result.shape == (3, 2)

    def test_list_of_1d(self):
        arr = np.array([1.0, 2.0])
        result = _normalize_shap_output([arr], (2, 1))
        assert result.ndim == 2
        assert result.shape == (2, 1)

    def test_3d_multi_sample(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = _normalize_shap_output(arr, (2, 2))
        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_0d_scalar_returns_as_is(self):
        arr = np.array(5.0)
        result = _normalize_shap_output(arr, (1, 1))
        assert result.ndim == 0
        assert float(result) == pytest.approx(5.0)


# ── TestGetTopContributors ────────────────────────────────────────


class TestGetTopContributors:
    def test_single_feature(self):
        shap_vals = np.array([0.5])
        result = get_top_contributors(shap_vals, ["temperature"])
        assert len(result) == 1
        assert result[0]["feature"] == "temperature"
        assert result[0]["shap_value"] == pytest.approx(0.5)

    def test_multi_feature(self):
        shap_vals = np.array([0.1, 0.9, 0.5])
        result = get_top_contributors(shap_vals, ["a", "b", "c"])
        assert len(result) == 3
        assert result[0]["feature"] == "b"
        assert result[0]["shap_value"] == pytest.approx(0.9)
        assert result[1]["feature"] == "c"
        assert result[1]["shap_value"] == pytest.approx(0.5)
        assert result[2]["feature"] == "a"
        assert result[2]["shap_value"] == pytest.approx(0.1)

    def test_top_n_capped(self):
        shap_vals = np.array([0.1, 0.2, 0.3])
        result = get_top_contributors(shap_vals, ["a", "b", "c"], top_n=2)
        assert len(result) == 2

    def test_top_n_exceeds_features(self):
        shap_vals = np.array([0.5, 0.3])
        result = get_top_contributors(shap_vals, ["x", "y"], top_n=10)
        assert len(result) == 2

    def test_negative_shap_values(self):
        shap_vals = np.array([-0.9, 0.1, 0.5])
        result = get_top_contributors(shap_vals, ["a", "b", "c"])
        assert result[0]["feature"] == "a"
        assert result[0]["shap_value"] == pytest.approx(-0.9)

    def test_all_zero_values(self):
        shap_vals = np.array([0.0, 0.0, 0.0])
        result = get_top_contributors(shap_vals, ["a", "b", "c"])
        assert len(result) == 3

    def test_single_element_in_multi_feature(self):
        shap_vals = np.array([42.0])
        result = get_top_contributors(shap_vals, ["only_col"], top_n=5)
        assert len(result) == 1
        assert result[0]["shap_value"] == pytest.approx(42.0)


# ── TestGetShapValues ─────────────────────────────────────────────


class TestGetShapValues:
    @patch("src.explainability._shap_kernel")
    def test_single_feature_uses_kernel(self, mock_kernel):
        mock_kernel.return_value = np.array([[0.5]])
        if_model = MagicMock()
        X = np.random.randn(10, 1).astype(np.float32)

        result = _get_shap_values(if_model, X)

        mock_kernel.assert_called_once()
        assert result.shape == (1, 1)

    @patch("src.explainability._shap_tree")
    def test_multi_feature_uses_tree(self, mock_tree):
        mock_tree.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        if_model = MagicMock()
        X = np.random.randn(2, 3).astype(np.float32)

        result = _get_shap_values(if_model, X)

        mock_tree.assert_called_once()
        assert result.shape == (2, 3)

    @patch("src.explainability._shap_kernel")
    @patch("src.explainability._shap_tree", side_effect=Exception("tree failed"))
    def test_tree_fallback_to_kernel(self, mock_tree, mock_kernel):
        mock_kernel.return_value = np.array([[0.1, 0.2]])
        if_model = MagicMock()
        X = np.random.randn(1, 3).astype(np.float32)

        result = _get_shap_values(if_model, X)

        mock_tree.assert_called_once()
        mock_kernel.assert_called_once()
        assert result.shape == (1, 2)


# ── TestExplainAnomalies ─────────────────────────────────────────


class TestExplainAnomalies:
    def test_no_anomalies(self):
        if_model = MagicMock()
        X = np.random.randn(10, 2).astype(np.float32)
        ensemble_preds = np.zeros(10, dtype=int)
        if_preds = np.zeros(10, dtype=int)

        result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        assert len(result) == 10
        assert all(r is None for r in result)

    def test_all_explainable(self):
        if_model = MagicMock()
        X = np.random.randn(5, 2).astype(np.float32)
        ensemble_preds = np.array([1, 1, 1, 0, 0])
        if_preds = np.array([1, 1, 1, 0, 0])

        mock_shap = np.array([[0.5, 0.3], [0.2, 0.8], [0.1, 0.9]])
        with patch("src.explainability._get_shap_values", return_value=mock_shap):
            result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        assert result[0] is not None
        assert result[0]["is_anomaly"] is True
        assert len(result[0]["top_features"]) == 2
        assert result[0]["message"] is None
        assert result[1] is not None
        assert result[2] is not None
        assert result[3] is None
        assert result[4] is None

    def test_all_non_explainable(self):
        if_model = MagicMock()
        X = np.random.randn(5, 2).astype(np.float32)
        ensemble_preds = np.array([1, 1, 0, 0, 0])
        if_preds = np.array([0, 0, 0, 0, 0])

        with patch("src.explainability._get_shap_values") as mock_shap:
            result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        mock_shap.assert_not_called()
        assert result[0] is not None
        assert result[0]["top_features"] == []
        assert "IF did not flag" in result[0]["message"]
        assert result[1] is not None
        assert result[2] is None

    def test_mixed_explainable_and_non_explainable(self):
        if_model = MagicMock()
        X = np.random.randn(5, 2).astype(np.float32)
        ensemble_preds = np.array([1, 1, 1, 0, 0])
        if_preds = np.array([1, 0, 1, 0, 0])

        mock_shap = np.array([[0.5, 0.3], [0.1, 0.9]])
        with patch("src.explainability._get_shap_values", return_value=mock_shap):
            result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        assert result[0] is not None
        assert result[0]["top_features"] != []
        assert result[1] is not None
        assert result[1]["top_features"] == []
        assert "IF did not flag" in result[1]["message"]
        assert result[2] is not None
        assert result[2]["top_features"] != []
        assert result[3] is None

    def test_subsampling(self):
        if_model = MagicMock()
        X = np.random.randn(200, 2).astype(np.float32)
        ensemble_preds = np.ones(200, dtype=int)
        if_preds = np.ones(200, dtype=int)

        mock_shap = np.zeros((100, 2))
        with patch("src.explainability._get_shap_values", return_value=mock_shap) as mock_get, \
             patch("src.explainability.SHAP_SAMPLE_SIZE", 100):
            result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        called_X = mock_get.call_args[0][1]
        assert called_X.shape[0] == 100
        assert called_X.shape[1] == 2

    def test_subsampling_not_triggered(self):
        if_model = MagicMock()
        X = np.random.randn(5, 2).astype(np.float32)
        ensemble_preds = np.array([1, 1, 1, 0, 0])
        if_preds = np.array([1, 1, 1, 0, 0])

        mock_shap = np.zeros((3, 2))
        with patch("src.explainability._get_shap_values", return_value=mock_shap) as mock_get, \
             patch("src.explainability.SHAP_SAMPLE_SIZE", 100):
            result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        called_X = mock_get.call_args[0][1]
        assert called_X.shape[0] == 3

    def test_single_feature(self):
        if_model = MagicMock()
        X = np.random.randn(5, 1).astype(np.float32)
        ensemble_preds = np.array([1, 1, 0, 0, 0])
        if_preds = np.array([1, 1, 0, 0, 0])

        mock_shap = np.array([[0.5], [0.3]])
        with patch("src.explainability._get_shap_values", return_value=mock_shap):
            result = explain_anomalies(if_model, X, ["value"], ensemble_preds, if_preds)

        assert result[0] is not None
        assert result[0]["top_features"][0]["feature"] == "value"
        assert result[1] is not None
        assert result[2] is None

    def test_non_anomalous_points_unchanged(self):
        if_model = MagicMock()
        X = np.random.randn(5, 2).astype(np.float32)
        ensemble_preds = np.array([0, 0, 0, 0, 0])
        if_preds = np.array([0, 0, 0, 0, 0])

        result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        assert all(r is None for r in result)

    def test_explanations_row_indices_correct(self):
        if_model = MagicMock()
        X = np.random.randn(10, 2).astype(np.float32)
        ensemble_preds = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        if_preds = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        mock_shap = np.array([[0.5, 0.3]])
        with patch("src.explainability._get_shap_values", return_value=mock_shap):
            result = explain_anomalies(if_model, X, ["a", "b"], ensemble_preds, if_preds)

        assert result[2] is not None
        assert result[2]["row_index"] == 2
        assert result[5] is not None
        assert result[5]["row_index"] == 5
        assert result[5]["message"] is not None
