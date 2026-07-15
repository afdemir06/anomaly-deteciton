import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderModel, map_sequences_to_points


class TestLSTMAutoencoder:
    def test_init(self):
        model = LSTMAutoencoder(n_features=3, hidden_dim=64, seq_length=10)
        assert model.seq_length == 10
        assert model.hidden_dim == 64

    def test_forward_shape(self):
        import torch
        model = LSTMAutoencoder(n_features=3, hidden_dim=64, seq_length=10)
        x = torch.randn(5, 10, 3)
        output = model(x)
        assert output.shape == (5, 10, 3)


class TestLSTMAutoencoderModel:
    def test_init(self):
        model = LSTMAutoencoderModel()
        assert model.model is None
        assert model.threshold is None
        assert "seq_length" in model.params
        assert "hidden_dim" in model.params
        assert "num_epochs" in model.params
        assert "learning_rate" in model.params
        assert "threshold_percentile" in model.params

    def test_predict_before_train_raises(self):
        model = LSTMAutoencoderModel()
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(np.random.randn(10, 5, 3).astype(np.float32))

    def test_score_before_train_raises(self):
        model = LSTMAutoencoderModel()
        with pytest.raises(ValueError, match="Model not trained"):
            model.score(np.random.randn(10, 5, 3).astype(np.float32))

    @patch("src.models.lstm_autoencoder.mlflow")
    def test_train(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = LSTMAutoencoderModel()
        model.params["num_epochs"] = 2
        X_seq = np.random.randn(50, 10, 3).astype(np.float32)
        metrics = model.train(X_seq)

        assert model.model is not None
        assert model.threshold is not None
        assert "n_anomalies" in metrics
        assert "anomaly_rate" in metrics
        assert "mean_reconstruction_error" in metrics
        assert "threshold" in metrics
        assert "final_loss" in metrics
        assert "run_id" in metrics

    @patch("src.models.lstm_autoencoder.mlflow")
    def test_predict_after_train(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = LSTMAutoencoderModel()
        model.params["num_epochs"] = 2
        X_seq = np.random.randn(50, 10, 3).astype(np.float32)
        model.train(X_seq)

        preds = model.predict(X_seq)
        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0, 1})

    @patch("src.models.lstm_autoencoder.mlflow")
    def test_score_after_train(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = LSTMAutoencoderModel()
        model.params["num_epochs"] = 2
        X_seq = np.random.randn(50, 10, 3).astype(np.float32)
        model.train(X_seq)

        scores = model.score(X_seq)
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    @patch("src.models.lstm_autoencoder.mlflow")
    def test_anomaly_rate_in_range(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        model = LSTMAutoencoderModel()
        model.params["num_epochs"] = 2
        X_seq = np.random.randn(50, 10, 3).astype(np.float32)
        metrics = model.train(X_seq)

        assert 0.0 <= metrics["anomaly_rate"] <= 1.0


class TestMapSequencesToPoints:
    def test_no_anomalies(self):
        seq_preds = np.array([0, 0, 0, 0])
        result = map_sequences_to_points(seq_preds, n_points=10, seq_length=3)
        assert result.shape == (10,)
        assert np.sum(result) == 0

    def test_single_anomaly(self):
        seq_preds = np.array([0, 1, 0, 0])
        result = map_sequences_to_points(seq_preds, n_points=10, seq_length=3)
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 1

    def test_overlapping_anomalies(self):
        seq_preds = np.array([1, 1, 0, 0])
        result = map_sequences_to_points(seq_preds, n_points=10, seq_length=3)
        assert result[0] == 1
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 1

    def test_all_anomalies(self):
        seq_preds = np.array([1, 1, 1, 1])
        result = map_sequences_to_points(seq_preds, n_points=6, seq_length=3)
        assert np.all(result == 1)
