import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.pipeline import AnomalyDetectionPipeline


def _make_preprocess_data(n_rows=100, n_features=1):
    """Return a mock preprocess() output dict."""
    np.random.seed(42)
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
         **{c: np.random.randn(n_rows).astype(float) for c in cols}},
    )
    seq_length = 30
    X_scaled = np.random.randn(n_rows, n_features).astype(np.float32)
    n_seq = max(n_rows - seq_length + 1, 1)
    X_seq = np.random.randn(n_seq, seq_length, n_features).astype(np.float32)
    return {
        "df": df,
        "datetime_col": "timestamp",
        "feature_cols": cols,
        "X_scaled": X_scaled,
        "X_seq": X_seq,
        "seq_length": seq_length,
        "scaler": MagicMock(),
    }


def _make_mock_models(pipeline):
    """Patch train/predict on all three models to return valid arrays."""
    data = _make_preprocess_data()
    n = data["X_scaled"].shape[0]
    n_seq = data["X_seq"].shape[0]

    pipeline.if_model.train = MagicMock(return_value={"n_anomalies": 5, "anomaly_rate": 0.05, "mean_score": -0.5})
    pipeline.if_model.predict = MagicMock(return_value=np.array([0] * 95 + [1] * 5))

    pipeline.dbscan_model.train = MagicMock(return_value={"n_anomalies": 3, "anomaly_rate": 0.03, "mean_score": 1.2, "n_clusters": 5})
    pipeline.dbscan_model.predict = MagicMock(return_value=np.array([0] * 97 + [1] * 3))

    pipeline.lstm_model.train = MagicMock(return_value={"n_anomalies": 4, "anomaly_rate": 0.04, "mean_reconstruction_error": 0.01, "threshold": 0.02, "final_loss": 0.005})
    pipeline.lstm_model.predict = MagicMock(return_value=np.array([0] * (n_seq - 4) + [1] * 4))

    return data


# ── TestEnsembleVote ──────────────────────────────────────────────


class TestEnsembleVote:
    def test_majority_vote(self):
        pipeline = AnomalyDetectionPipeline()
        if_preds = np.array([1, 1, 0, 0])
        dbscan_preds = np.array([1, 0, 1, 0])
        lstm_preds = np.array([0, 1, 0, 0])

        result = pipeline._ensemble_vote(if_preds, dbscan_preds, lstm_preds)
        assert result[0] == 1
        assert result[1] == 1
        assert result[2] == 0
        assert result[3] == 0

    def test_unanimous_vote(self):
        pipeline = AnomalyDetectionPipeline()
        if_preds = np.array([1, 1, 1])
        dbscan_preds = np.array([1, 1, 1])
        lstm_preds = np.array([1, 1, 1])

        result = pipeline._ensemble_vote(if_preds, dbscan_preds, lstm_preds)
        assert np.all(result == 1)

    def test_no_anomalies(self):
        pipeline = AnomalyDetectionPipeline()
        if_preds = np.array([0, 0, 0])
        dbscan_preds = np.array([0, 0, 0])
        lstm_preds = np.array([0, 0, 0])

        result = pipeline._ensemble_vote(if_preds, dbscan_preds, lstm_preds)
        assert np.all(result == 0)

    def test_single_vote_not_enough(self):
        pipeline = AnomalyDetectionPipeline()
        if_preds = np.array([1, 0, 0])
        dbscan_preds = np.array([0, 0, 0])
        lstm_preds = np.array([0, 0, 0])

        result = pipeline._ensemble_vote(if_preds, dbscan_preds, lstm_preds)
        assert np.all(result == 0)

    def test_two_vote_exact_threshold(self):
        pipeline = AnomalyDetectionPipeline()
        if_preds = np.array([1, 0, 0])
        dbscan_preds = np.array([1, 1, 0])
        lstm_preds = np.array([0, 1, 0])

        result = pipeline._ensemble_vote(if_preds, dbscan_preds, lstm_preds)
        assert result[0] == 1
        assert result[1] == 1
        assert result[2] == 0

    def test_empty_arrays(self):
        pipeline = AnomalyDetectionPipeline()
        empty = np.array([], dtype=int)
        result = pipeline._ensemble_vote(empty, empty, empty)
        assert len(result) == 0


# ── TestPipelineInit ──────────────────────────────────────────────


class TestPipelineInit:
    def test_init(self):
        pipeline = AnomalyDetectionPipeline()
        assert pipeline.if_model is not None
        assert pipeline.dbscan_model is not None
        assert pipeline.lstm_model is not None
        assert pipeline.metadata == {}


# ── TestPipelineTrain ─────────────────────────────────────────────


class TestPipelineTrain:
    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_happy_path(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        result = pipeline.train("/fake/path.csv")

        assert "run_id" in result
        assert "if_metrics" in result
        assert "dbscan_metrics" in result
        assert "lstm_metrics" in result
        assert "ensemble_metrics" in result
        assert isinstance(result["run_id"], str)
        assert len(result["run_id"]) > 0

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_sets_mlflow(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        pipeline.train("/fake/path.csv")

        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_calls_preprocess(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        pipeline.train("/fake/path.csv")

        mock_preprocess.assert_called_once_with("/fake/path.csv", label_column="Class")

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_trains_all_models(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        pipeline.train("/fake/path.csv")

        pipeline.if_model.train.assert_called_once()
        pipeline.dbscan_model.train.assert_called_once()
        pipeline.lstm_model.train.assert_called_once()

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_generates_predictions(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        pipeline.train("/fake/path.csv")

        pipeline.if_model.predict.assert_called_once()
        pipeline.dbscan_model.predict.assert_called_once()
        pipeline.lstm_model.predict.assert_called_once()
        mock_map.assert_called_once()

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_saves_run_to_db(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        result = pipeline.train("/fake/path.csv")

        mock_save_run.assert_called_once()
        call_kwargs = mock_save_run.call_args
        assert call_kwargs[1]["run_id"] == result["run_id"]
        assert call_kwargs[1]["file_name"] == "path.csv"
        assert call_kwargs[1]["n_rows"] == data["X_scaled"].shape[0]
        assert call_kwargs[1]["n_features"] == data["X_scaled"].shape[1]
        assert call_kwargs[1]["feature_cols"] == data["feature_cols"]
        assert call_kwargs[1]["datetime_col"] == data["datetime_col"]
        assert call_kwargs[1]["seq_length"] == data["seq_length"]

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_saves_results_and_models(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        result = pipeline.train("/fake/path.csv")

        mock_save_results.assert_called_once()
        mock_save.assert_called_once()
        saved_results = mock_save_results.call_args[0][1]
        assert len(saved_results) == data["X_scaled"].shape[0]

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_shap_explanations_attached(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=10)
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(10, dtype=int)

        explanation = {"row_index": 5, "is_anomaly": True, "top_features": [], "message": None}
        explanations_list = [None] * 10
        explanations_list[5] = explanation
        mock_explain.return_value = explanations_list

        pipeline.if_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": -0.5})
        pipeline.if_model.predict = MagicMock(return_value=np.zeros(10, dtype=int))
        pipeline.dbscan_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": 1.0, "n_clusters": 2})
        pipeline.dbscan_model.predict = MagicMock(return_value=np.zeros(10, dtype=int))
        pipeline.lstm_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_reconstruction_error": 0.01, "threshold": 0.02, "final_loss": 0.005})
        pipeline.lstm_model.predict = MagicMock(return_value=np.zeros(7, dtype=int))

        result = pipeline.train("/fake/path.csv")

        mock_explain.assert_called_once()

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_sets_metadata(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        result = pipeline.train("/fake/path.csv")

        assert pipeline.metadata["run_id"] == result["run_id"]
        assert pipeline.metadata["feature_cols"] == data["feature_cols"]
        assert pipeline.metadata["datetime_col"] == data["datetime_col"]
        assert pipeline.metadata["seq_length"] == data["seq_length"]

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_timestamps_converted(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=5)
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(5, dtype=int)

        pipeline.if_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": -0.5})
        pipeline.if_model.predict = MagicMock(return_value=np.zeros(5, dtype=int))
        pipeline.dbscan_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": 1.0, "n_clusters": 2})
        pipeline.dbscan_model.predict = MagicMock(return_value=np.zeros(5, dtype=int))
        pipeline.lstm_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_reconstruction_error": 0.01, "threshold": 0.02, "final_loss": 0.005})
        pipeline.lstm_model.predict = MagicMock(return_value=np.zeros(3, dtype=int))

        pipeline.train("/fake/path.csv")

        saved_results = mock_save_results.call_args[0][1]
        for r in saved_results:
            assert r["timestamp"] is not None

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_ensemble_metrics(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        result = pipeline.train("/fake/path.csv")

        em = result["ensemble_metrics"]
        assert "n_anomalies" in em
        assert "anomaly_rate" in em
        assert isinstance(em["n_anomalies"], int)
        assert isinstance(em["anomaly_rate"], float)

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_results_structure(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=5)
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(5, dtype=int)

        pipeline.if_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": -0.5})
        pipeline.if_model.predict = MagicMock(return_value=np.array([1, 0, 0, 0, 0]))
        pipeline.dbscan_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": 1.0, "n_clusters": 2})
        pipeline.dbscan_model.predict = MagicMock(return_value=np.zeros(5, dtype=int))
        pipeline.lstm_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_reconstruction_error": 0.01, "threshold": 0.02, "final_loss": 0.005})
        pipeline.lstm_model.predict = MagicMock(return_value=np.zeros(3, dtype=int))

        pipeline.train("/fake/path.csv")

        saved_results = mock_save_results.call_args[0][1]
        assert len(saved_results) == 5
        for r in saved_results:
            assert "row_index" in r
            assert "timestamp" in r
            assert "feature_values" in r
            assert "is_anomaly" in r
            assert "anomaly_score" in r
            assert "if_vote" in r
            assert "dbscan_vote" in r
            assert "lstm_vote" in r

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_init_db_called(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data()
        mock_preprocess.return_value = data
        mock_map.return_value = np.zeros(data["X_scaled"].shape[0], dtype=int)
        _make_mock_models(pipeline)

        pipeline.train("/fake/path.csv")

        mock_init_db.assert_called_once()

    @patch("src.pipeline.save_models")
    @patch("src.pipeline.explain_anomalies", return_value=[None] * 100)
    @patch("src.pipeline.map_sequences_to_points")
    @patch("src.pipeline.mlflow")
    @patch("src.db.save_results")
    @patch("src.db.save_run")
    @patch("src.db.init_db")
    @patch("src.pipeline.preprocess")
    def test_train_multi_feature(
        self, mock_preprocess, mock_init_db, mock_save_run,
        mock_save_results, mock_mlflow, mock_map, mock_explain, mock_save,
    ):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=50, n_features=3)
        mock_preprocess.return_value = data
        n = data["X_scaled"].shape[0]
        n_seq = data["X_seq"].shape[0]
        mock_map.return_value = np.zeros(n, dtype=int)

        pipeline.if_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": -0.5})
        pipeline.if_model.predict = MagicMock(return_value=np.zeros(n, dtype=int))
        pipeline.dbscan_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_score": 1.0, "n_clusters": 2})
        pipeline.dbscan_model.predict = MagicMock(return_value=np.zeros(n, dtype=int))
        pipeline.lstm_model.train = MagicMock(return_value={"n_anomalies": 0, "anomaly_rate": 0.0, "mean_reconstruction_error": 0.01, "threshold": 0.02, "final_loss": 0.005})
        pipeline.lstm_model.predict = MagicMock(return_value=np.zeros(n_seq, dtype=int))

        result = pipeline.train("/fake/path.csv")

        assert result["ensemble_metrics"]["n_anomalies"] >= 0


# ── TestPipelineDetect ────────────────────────────────────────────


class TestPipelineDetect:
    def test_detect_no_run_raises(self):
        pipeline = AnomalyDetectionPipeline()
        with patch("src.db.get_latest_run_id", return_value=None):
            with pytest.raises(FileNotFoundError, match="No trained model found"):
                pipeline.detect(None)

    def test_detect_empty_results_raises(self):
        pipeline = AnomalyDetectionPipeline()
        with patch("src.db.get_latest_run_id", return_value="run-123"), \
             patch("src.db.get_results", return_value=[]):
            with pytest.raises(ValueError, match="No results found"):
                pipeline.detect("run-123")

    def test_detect_with_results(self):
        pipeline = AnomalyDetectionPipeline()
        mock_results = [
            {"if_vote": 1, "dbscan_vote": 1, "lstm_vote": 0},
            {"if_vote": 0, "dbscan_vote": 0, "lstm_vote": 0},
            {"if_vote": 1, "dbscan_vote": 0, "lstm_vote": 1},
        ]
        with patch("src.db.get_latest_run_id", return_value="run-123"), \
             patch("src.db.get_results", return_value=mock_results):
            result = pipeline.detect("run-123")

            assert result["run_id"] == "run-123"
            assert result["predictions"].tolist() == [1, 0, 1]
            assert result["if_votes"].tolist() == [1, 0, 1]
            assert result["dbscan_votes"].tolist() == [1, 0, 0]
            assert result["lstm_votes"].tolist() == [0, 0, 1]

    def test_detect_no_run_id_uses_latest(self):
        pipeline = AnomalyDetectionPipeline()
        mock_results = [
            {"if_vote": 1, "dbscan_vote": 0, "lstm_vote": 0},
        ]
        with patch("src.db.get_latest_run_id", return_value="latest-run"), \
             patch("src.db.get_results", return_value=mock_results):
            result = pipeline.detect()

            assert result["run_id"] == "latest-run"


# ── TestPipelineDetectFromFile ────────────────────────────────────


class TestPipelineDetectFromFile:
    def test_detect_from_file_happy_path(self):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=50)

        mock_loaded = {
            "if_model": MagicMock(),
            "dbscan_model": MagicMock(),
            "lstm_model": MagicMock(),
            "metadata": {"run_id": "run-abc", "feature_cols": ["f0"], "datetime_col": "timestamp", "seq_length": 30},
        }
        mock_loaded["if_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["dbscan_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["lstm_model"].predict.return_value = np.zeros(50 - 30 + 1, dtype=int)

        with patch("src.pipeline.load_models", return_value=mock_loaded), \
             patch("src.pipeline.preprocess", return_value=data), \
             patch("src.pipeline.map_sequences_to_points", return_value=np.zeros(50, dtype=int)), \
             patch("src.db.get_latest_run_id", return_value="run-abc"):
            result = pipeline.detect_from_file("/fake/file.csv", "run-abc")

            assert result["run_id"] == "run-abc"
            assert len(result["predictions"]) == 50

    def test_detect_from_file_no_run_id_no_latest(self):
        pipeline = AnomalyDetectionPipeline()
        with patch("src.db.get_latest_run_id", return_value=None):
            with pytest.raises(FileNotFoundError, match="No trained model found"):
                pipeline.detect_from_file("/fake/file.csv")

    def test_detect_from_file_no_run_id_with_latest(self):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=50)

        mock_loaded = {
            "if_model": MagicMock(),
            "dbscan_model": MagicMock(),
            "lstm_model": MagicMock(),
            "metadata": {"run_id": "latest-run", "feature_cols": ["f0"], "datetime_col": "timestamp", "seq_length": 30},
        }
        mock_loaded["if_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["dbscan_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["lstm_model"].predict.return_value = np.zeros(50 - 30 + 1, dtype=int)

        with patch("src.pipeline.load_models", return_value=mock_loaded), \
             patch("src.pipeline.preprocess", return_value=data), \
             patch("src.pipeline.map_sequences_to_points", return_value=np.zeros(50, dtype=int)), \
             patch("src.db.get_latest_run_id", return_value="latest-run"):
            result = pipeline.detect_from_file("/fake/file.csv")

            assert result["run_id"] == "latest-run"

    def test_detect_from_file_loads_models(self):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=50)

        mock_loaded = {
            "if_model": MagicMock(),
            "dbscan_model": MagicMock(),
            "lstm_model": MagicMock(),
            "metadata": {"run_id": "run-xyz", "feature_cols": ["f0"], "datetime_col": "timestamp", "seq_length": 30},
        }
        mock_loaded["if_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["dbscan_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["lstm_model"].predict.return_value = np.zeros(50 - 30 + 1, dtype=int)

        with patch("src.pipeline.load_models", return_value=mock_loaded) as mock_load, \
             patch("src.pipeline.preprocess", return_value=data), \
             patch("src.pipeline.map_sequences_to_points", return_value=np.zeros(50, dtype=int)), \
             patch("src.db.get_latest_run_id", return_value="run-xyz"):
            pipeline.detect_from_file("/fake/file.csv", "run-xyz")

            mock_load.assert_called_once_with("run-xyz")

    def test_detect_from_file_ensemble_vote(self):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=50)

        mock_loaded = {
            "if_model": MagicMock(),
            "dbscan_model": MagicMock(),
            "lstm_model": MagicMock(),
            "metadata": {"run_id": "run-99", "feature_cols": ["f0"], "datetime_col": "timestamp", "seq_length": 30},
        }
        mock_loaded["if_model"].predict.return_value = np.array([1, 1, 0] + [0] * 47)
        mock_loaded["dbscan_model"].predict.return_value = np.array([1, 0, 0] + [0] * 47)
        mock_loaded["lstm_model"].predict.return_value = np.zeros(50 - 30 + 1, dtype=int)

        with patch("src.pipeline.load_models", return_value=mock_loaded), \
             patch("src.pipeline.preprocess", return_value=data), \
             patch("src.pipeline.map_sequences_to_points", return_value=np.zeros(50, dtype=int)), \
             patch("src.db.get_latest_run_id", return_value="run-99"):
            result = pipeline.detect_from_file("/fake/file.csv", "run-99")

            assert result["if_votes"][0] == 1
            assert result["dbscan_votes"][0] == 1

    def test_detect_from_file_updates_model_attrs(self):
        pipeline = AnomalyDetectionPipeline()
        data = _make_preprocess_data(n_rows=50)

        mock_loaded = {
            "if_model": MagicMock(),
            "dbscan_model": MagicMock(),
            "lstm_model": MagicMock(),
            "metadata": {"run_id": "run-new", "feature_cols": ["f0"], "datetime_col": "timestamp", "seq_length": 30},
        }
        mock_loaded["if_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["dbscan_model"].predict.return_value = np.zeros(50, dtype=int)
        mock_loaded["lstm_model"].predict.return_value = np.zeros(50 - 30 + 1, dtype=int)

        with patch("src.pipeline.load_models", return_value=mock_loaded), \
             patch("src.pipeline.preprocess", return_value=data), \
             patch("src.pipeline.map_sequences_to_points", return_value=np.zeros(50, dtype=int)), \
             patch("src.db.get_latest_run_id", return_value="run-new"):
            pipeline.detect_from_file("/fake/file.csv", "run-new")

            assert pipeline.metadata == mock_loaded["metadata"]
