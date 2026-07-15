import logging

import mlflow
import numpy as np

try:
    from cuml.ensemble import IsolationForest
except ImportError:
    from sklearn.ensemble import IsolationForest

from src.config import settings

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """Isolation Forest anomaly detection model."""

    def __init__(self):
        self.model = None
        self.params = {
            "n_estimators": settings.IF_N_ESTIMATORS,
            "contamination": settings.IF_CONTAMINATION,
            "random_state": settings.IF_RANDOM_STATE,
        }

    def train(self, X: np.ndarray) -> dict:
        """Train the model, log to MLflow, return metrics."""
        self.model = IsolationForest(**self.params)
        self.model.fit(X)

        predictions = self.predict(X)
        scores = self.score(X)

        metrics = {
            "n_anomalies": int(np.sum(predictions)),
            "anomaly_rate": float(np.mean(predictions)),
            "mean_score": float(np.mean(scores)),
        }

        run_id = self.log_to_mlflow(metrics)
        metrics["run_id"] = run_id

        logger.info(
            "Isolation Forest trained: %d anomalies (%.2f%%)",
            metrics["n_anomalies"],
            metrics["anomaly_rate"] * 100,
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores (lower = more anomalous)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.score_samples(X)

    def log_to_mlflow(self, metrics: dict) -> None:
        """Log params, metrics, and model artifact to MLflow."""

        with mlflow.start_run(run_name="isolation_forest") as run:
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "isolation_forest_model", serialization_format="pickle")
            return run.info.run_id