import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.models.dbscan import DBSCANModel


class TestDBSCANModel:
    def test_init(self):
        model = DBSCANModel()
        assert model.model is None
        assert model.nn_model is None
        assert "eps" in model.params
        assert "min_samples" in model.params
        assert "metric" in model.params

    def test_predict_before_train_raises(self):
        model = DBSCANModel()
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(np.random.randn(10, 3))

    def test_score_before_train_raises(self):
        model = DBSCANModel()
        with pytest.raises(ValueError, match="Model not trained"):
            model.score(np.random.randn(10, 3))

    @patch("src.models.dbscan.mlflow")
    def test_train(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = DBSCANModel()
        X = np.random.randn(100, 3)
        metrics = model.train(X)

        assert model.model is not None
        assert "n_anomalies" in metrics
        assert "anomaly_rate" in metrics
        assert "n_clusters" in metrics
        assert "mean_score" in metrics
        assert "run_id" in metrics

    @patch("src.models.dbscan.mlflow")
    def test_predict_after_train(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = DBSCANModel()
        X = np.random.randn(100, 3)
        model.train(X)

        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    @patch("src.models.dbscan.mlflow")
    def test_score_after_train(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = DBSCANModel()
        X = np.random.randn(100, 3)
        model.train(X)

        scores = model.score(X)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)

    @patch("src.models.dbscan.mlflow")
    def test_anomaly_rate_in_range(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = DBSCANModel()
        X = np.random.randn(100, 3)
        metrics = model.train(X)

        assert 0.0 <= metrics["anomaly_rate"] <= 1.0

    @patch("src.models.dbscan.mlflow")
    def test_n_clusters_non_negative(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = DBSCANModel()
        X = np.random.randn(100, 3)
        metrics = model.train(X)

        assert metrics["n_clusters"] >= 0
