import logging

import mlflow
import numpy as np

try:
    from cuml.cluster import DBSCAN
    from cuml.neighbors import NearestNeighbors

    logger_cuml = logging.getLogger(__name__)
    logger_cuml.info("Using cuML (GPU) for DBSCAN")
except ImportError:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

from src.config import settings

logger = logging.getLogger(__name__)


class DBSCANModel:
    """DBSCAN clustering-based anomaly detection model (Two-Phase)."""

    def __init__(self):
        self.model = None
        self.nn_model = None
        self.core_points = None
        self.params = {
            "eps": settings.DBSCAN_EPS,
            "min_samples": settings.DBSCAN_MIN_SAMPLES,
            "metric": settings.DBSCAN_METRIC,
        }

    def train(self, X: np.ndarray, subsample_size: int = 0) -> dict:
        """Fit DBSCAN, detect anomalies, log to MLflow, return metrics.

        If subsample_size > 0 and X is larger, train on a random subsample
        and use NearestNeighbors on core points for prediction on full data.
        """
        if subsample_size > 0 and X.shape[0] > subsample_size:
            rng = np.random.default_rng(42)
            indices = rng.choice(X.shape[0], size=subsample_size, replace=False)
            X_train = X[indices]
            logger.info("DBSCAN subsampled: %d -> %d rows", X.shape[0], subsample_size)
        else:
            X_train = X

        self.model = DBSCAN(**self.params)
        labels = self.model.fit_predict(X_train)

        core_indices = self.model.core_sample_indices_
        if len(core_indices) > 0:
            self.core_points = X_train[core_indices]
            self.nn_model = NearestNeighbors(n_neighbors=1)
            self.nn_model.fit(self.core_points)
        else:
            self.core_points = None
            self.nn_model = None

        predictions = self.predict(X)
        scores = self.score(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        metrics = {
            "n_anomalies": int(np.sum(predictions)),
            "anomaly_rate": float(np.mean(predictions)),
            "n_clusters": n_clusters,
            "mean_score": float(np.mean(scores)),
        }

        run_id = self.log_to_mlflow(metrics)
        metrics["run_id"] = run_id

        logger.info(
            "DBSCAN trained: %d anomalies (%.2f%%), %d clusters",
            metrics["n_anomalies"],
            metrics["anomaly_rate"] * 100,
            metrics["n_clusters"],
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal.

        Uses distance to nearest core point: if distance > eps -> anomaly.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.nn_model is None:
            return np.ones(X.shape[0], dtype=int)
        distances, _ = self.nn_model.kneighbors(X)
        return np.where(distances[:, 0] > self.params["eps"], 1, 0).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return Euclidean distance to nearest core point as anomaly score."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.nn_model is None:
            return np.full(X.shape[0], np.inf)
        distances, _ = self.nn_model.kneighbors(X)
        return distances[:, 0]

    def log_to_mlflow(self, metrics: dict) -> None:
        """Log params, metrics, and model artifact to MLflow."""

        with mlflow.start_run(run_name="dbscan") as run:
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "dbscan_model", serialization_format="pickle")
            return run.info.run_id
