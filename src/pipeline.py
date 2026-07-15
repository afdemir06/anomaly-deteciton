import logging
import os
from uuid import uuid4
import mlflow

import numpy as np
import pandas as pd

from src.config import settings
from src.models.isolation_forest import IsolationForestModel
from src.models.dbscan import DBSCANModel
from src.models.lstm_autoencoder import LSTMAutoencoderModel, map_sequences_to_points
from src.preprocessing import preprocess
from src.explainability import explain_anomalies
from src.utils import save_models, load_models

logger = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    """Orchestrates training, prediction, and ensemble voting."""

    def __init__(self):
        self.if_model = IsolationForestModel()
        self.dbscan_model = DBSCANModel()
        self.lstm_model = LSTMAutoencoderModel()
        self.metadata: dict = {}

    # -- Train ---------------------------------------------------------------

    def train(self, file_path: str) -> dict:
        """Full train pipeline: preprocess -> train 3 models -> ensemble -> save."""
        from src import db

        db.init_db()

        logger.info("Preprocessing %s ...", file_path)
        data = preprocess(file_path, label_column=settings.LABEL_COLUMN)
        file_name = os.path.basename(file_path)

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        logger.info("Training Isolation Forest...")
        if_metrics = self.if_model.train(data["X_scaled"])

        logger.info("Training DBSCAN...")
        dbscan_metrics = self.dbscan_model.train(
            data["X_scaled"], subsample_size=settings.DBSCAN_SUBSAMPLE_SIZE
        )

        logger.info("Training LSTM Autoencoder...")
        lstm_metrics = self.lstm_model.train(data["X_seq"])

        logger.info("Generating predictions...")
        if_preds = self.if_model.predict(data["X_scaled"])
        dbscan_preds = self.dbscan_model.predict(data["X_scaled"])
        lstm_seq_preds = self.lstm_model.predict(data["X_seq"])
        lstm_preds = map_sequences_to_points(
            lstm_seq_preds, data["X_scaled"].shape[0], data["seq_length"]
        )

        logger.info("Computing ensemble vote...")
        ensemble_preds = self._ensemble_vote(if_preds, dbscan_preds, lstm_preds)

        n_anomalies = int(np.sum(ensemble_preds))
        ensemble_metrics = {
            "n_anomalies": n_anomalies,
            "anomaly_rate": float(np.mean(ensemble_preds)),
        }

        run_id = str(uuid4())

        logger.info("Saving run to database (run_id=%s)...", run_id)
        db.save_run(
            run_id=run_id,
            file_name=file_name,
            n_rows=data["X_scaled"].shape[0],
            n_features=data["X_scaled"].shape[1],
            feature_cols=data["feature_cols"],
            datetime_col=data["datetime_col"],
            seq_length=data["seq_length"],
            if_metrics=if_metrics,
            dbscan_metrics=dbscan_metrics,
            lstm_metrics=lstm_metrics,
            ensemble_metrics=ensemble_metrics,
        )

        self.metadata = {
            "run_id": run_id,
            "feature_cols": data["feature_cols"],
            "datetime_col": data["datetime_col"],
            "seq_length": data["seq_length"],
        }

        logger.info("Building results list for %d rows...", len(ensemble_preds))
        results = []
        timestamps = (
            data["df"][data["datetime_col"]].tolist()
            if data["datetime_col"] in data["df"].columns
            else [None] * len(ensemble_preds)
        )
        has_label = (
            settings.LABEL_COLUMN
            and settings.LABEL_COLUMN in data["df"].columns
        )
        for i in range(len(ensemble_preds)):
            ts = timestamps[i]
            if ts is not None and isinstance(ts, (pd.Timestamp, pd.DatetimeTZDtype)):
                ts = ts.to_pydatetime()
            result = {
                "row_index": i,
                "timestamp": ts,
                "feature_values": {
                    col: float(data["X_scaled"][i, j])
                    for j, col in enumerate(data["feature_cols"])
                },
                "is_anomaly": bool(ensemble_preds[i]),
                "anomaly_score": float(
                    if_preds[i] + dbscan_preds[i] + lstm_preds[i]
                ),
                "if_vote": int(if_preds[i]),
                "dbscan_vote": int(dbscan_preds[i]),
                "lstm_vote": int(lstm_preds[i]),
            }
            if has_label:
                result["ground_truth"] = int(data["df"][settings.LABEL_COLUMN].iloc[i])
            results.append(result)

        logger.info("Computing SHAP explanations...")
        explanations = explain_anomalies(
            self.if_model, data["X_scaled"], data["feature_cols"],
            ensemble_preds, if_preds,
        )
        for i in range(len(results)):
            if explanations[i] is not None:
                results[i]["shap_explanation"] = explanations[i]

        logger.info("Saving %d results to database...", len(results))
        db.save_results(run_id, results)

        logger.info("Saving models to disk...")
        save_models(run_id, self.if_model, self.dbscan_model, self.lstm_model, self.metadata)

        logger.info("Training complete. Run ID: %s", run_id)

        return {
            "run_id": run_id,
            "if_metrics": if_metrics,
            "dbscan_metrics": dbscan_metrics,
            "lstm_metrics": lstm_metrics,
            "ensemble_metrics": ensemble_metrics,
        }

    # -- Detect ---------------------------------------------------------------

    def detect(self, run_id: str | None = None) -> dict:
        """Read results from DB for a given run. Returns ensemble predictions."""
        from src import db

        if run_id is None:
            run_id = db.get_latest_run_id()
            if run_id is None:
                raise FileNotFoundError(
                    "No trained model found. Call train() first."
                )

        results = db.get_results(run_id)
        if not results:
            raise ValueError(f"No results found for run {run_id}.")

        if_preds = np.array([r["if_vote"] for r in results])
        dbscan_preds = np.array([r["dbscan_vote"] for r in results])
        lstm_preds = np.array([r["lstm_vote"] for r in results])

        ensemble_preds = self._ensemble_vote(if_preds, dbscan_preds, lstm_preds)

        return {
            "run_id": run_id,
            "predictions": ensemble_preds,
            "if_votes": if_preds,
            "dbscan_votes": dbscan_preds,
            "lstm_votes": lstm_preds,
        }

    def detect_from_file(self, file_path: str, run_id: str | None = None) -> dict:
        """Load models from disk, preprocess file, predict, ensemble vote."""
        from src import db

        if run_id is None:
            run_id = db.get_latest_run_id()
            if run_id is None:
                raise FileNotFoundError(
                    "No trained model found. Call train() first."
                )

        loaded = load_models(run_id)
        self.if_model = loaded["if_model"]
        self.dbscan_model = loaded["dbscan_model"]
        self.lstm_model = loaded["lstm_model"]
        self.metadata = loaded["metadata"]

        data = preprocess(file_path)

        if_preds = self.if_model.predict(data["X_scaled"])
        dbscan_preds = self.dbscan_model.predict(data["X_scaled"])
        lstm_seq_preds = self.lstm_model.predict(data["X_seq"])
        lstm_preds = map_sequences_to_points(
            lstm_seq_preds, data["X_scaled"].shape[0], data["seq_length"]
        )

        ensemble_preds = self._ensemble_vote(if_preds, dbscan_preds, lstm_preds)

        return {
            "run_id": run_id,
            "predictions": ensemble_preds,
            "if_votes": if_preds,
            "dbscan_votes": dbscan_preds,
            "lstm_votes": lstm_preds,
        }

    # -- Ensemble ---------------------------------------------------------------

    def _ensemble_vote(
        self,
        if_preds: np.ndarray,
        dbscan_preds: np.ndarray,
        lstm_preds: np.ndarray,
    ) -> np.ndarray:
        """Majority voting: >= ENSEMBLE_MIN_VOTES means anomaly."""
        votes = if_preds.astype(int) + dbscan_preds.astype(int) + lstm_preds.astype(int)
        return (votes >= settings.ENSEMBLE_MIN_VOTES).astype(int)
