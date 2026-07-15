import numpy as np
import pytest
from unittest.mock import patch

from src.models.isolation_forest import IsolationForestModel
from src.models.dbscan import DBSCANModel
from src.models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderModel
from src.utils import save_models, load_models


class TestSaveAndLoadModels:
    def test_save_and_load_roundtrip(self, tmp_path):
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        run_id = "test-roundtrip"
        model_dir = tmp_path / "models" / run_id
        model_dir.mkdir(parents=True)

        with patch("src.utils.MODELS_DIR", tmp_path / "models"):
            if_model = IsolationForestModel()
            X = np.random.randn(100, 3)
            if_model.model = IsolationForest(n_estimators=10).fit(X)

            dbscan_model = DBSCANModel()
            dbscan_model.model = DBSCAN(eps=0.5).fit(X)
            core_indices = dbscan_model.model.core_sample_indices_
            if len(core_indices) > 0:
                dbscan_model.nn_model = NearestNeighbors(n_neighbors=1).fit(X[core_indices])

            lstm_model = LSTMAutoencoderModel()
            hidden_dim = lstm_model.params["hidden_dim"]
            lstm_model.model = LSTMAutoencoder(
                n_features=3, hidden_dim=hidden_dim, seq_length=5,
            )
            lstm_model.model.eval()
            lstm_model.threshold = 0.5

            metadata = {
                "run_id": run_id,
                "feature_cols": ["a", "b", "c"],
                "datetime_col": "ts",
                "seq_length": 5,
            }

            save_models(run_id, if_model, dbscan_model, lstm_model, metadata)

            assert (model_dir / "isolation_forest.joblib").exists()
            assert (model_dir / "dbscan.joblib").exists()
            assert (model_dir / "dbscan_nn.joblib").exists()
            assert (model_dir / "lstm_autoencoder.pt").exists()
            assert (model_dir / "lstm_threshold.joblib").exists()
            assert (model_dir / "lstm_config.joblib").exists()
            assert (model_dir / "metadata.json").exists()

            loaded = load_models(run_id)

            assert loaded["if_model"].model is not None
            assert loaded["dbscan_model"].model is not None
            assert loaded["lstm_model"].model is not None
            assert loaded["metadata"]["feature_cols"] == ["a", "b", "c"]
            assert loaded["metadata"]["datetime_col"] == "ts"
            assert loaded["metadata"]["seq_length"] == 5

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="No trained model found"):
            load_models("nonexistent-run-id")
