import logging

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import settings

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM Encoder-Decoder Autoencoder for anomaly detection."""

    def __init__(self, n_features: int, hidden_dim: int, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x)

        decoder_input = hidden[-1].unsqueeze(1).repeat(1, self.seq_length, 1)

        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        output = self.output_layer(decoder_output)
        return output


class LSTMAutoencoderModel:
    """LSTM Autoencoder anomaly detection model."""

    def __init__(self):
        self.model = None
        self.threshold = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = {
            "seq_length": settings.LSTM_SEQ_LENGTH,
            "hidden_dim": settings.LSTM_HIDDEN_DIM,
            "num_epochs": settings.LSTM_NUM_EPOCHS,
            "learning_rate": settings.LSTM_LEARNING_RATE,
            "threshold_percentile": settings.LSTM_THRESHOLD_PERCENTILE,
        }

    def train(self, X_seq: np.ndarray) -> dict:
        """Train LSTM Autoencoder, compute threshold, log to MLflow."""
        n_features = X_seq.shape[2]
        seq_length = X_seq.shape[1]

        self.model = LSTMAutoencoder(n_features, self.params["hidden_dim"], seq_length)
        self.model.to(self.device)

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        criterion = nn.MSELoss()

        self.model.train()
        final_loss = 0.0

        for epoch in range(self.params["num_epochs"]):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(dataloader)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
            errors_np = errors.cpu().numpy()

        self.threshold = float(np.percentile(errors_np, self.params["threshold_percentile"]))

        predictions = self.predict(X_seq)
        n_anomalies = int(np.sum(predictions))

        metrics = {
            "n_anomalies": n_anomalies,
            "anomaly_rate": float(np.mean(predictions)),
            "mean_reconstruction_error": float(np.mean(errors_np)),
            "threshold": self.threshold,
            "final_loss": final_loss,
        }

        run_id = self.log_to_mlflow(metrics)
        metrics["run_id"] = run_id

        logger.info(
            "LSTM Autoencoder trained: %d anomalies (%.2f%%), threshold=%.4f",
            metrics["n_anomalies"],
            metrics["anomaly_rate"] * 100,
            self.threshold,
        )

        return metrics

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns 1 for anomaly, 0 for normal."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        scores = self.score(X_seq)
        return (scores > self.threshold).astype(int)

    def score(self, X_seq: np.ndarray) -> np.ndarray:
        """Return reconstruction errors as anomaly scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.model.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
        return errors.cpu().numpy()

    def log_to_mlflow(self, metrics: dict) -> None:
        """Log params, metrics, and model artifact to MLflow."""

        with mlflow.start_run(run_name="lstm_autoencoder") as run:
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(
                self.model,
                "lstm_autoencoder_model",
                serialization_format="pickle",
            )
            return run.info.run_id

def map_sequences_to_points(
    seq_predictions: np.ndarray,
    n_points: int,
    seq_length: int,
) -> np.ndarray:
    """Map sequence-level predictions back to point-level.

    If any sequence covering a point is anomalous, the point is anomalous.
    """
    point_predictions = np.zeros(n_points, dtype=int)
    n_sequences = len(seq_predictions)

    for i in range(n_sequences):
        if seq_predictions[i] == 1:
            point_predictions[i : i + seq_length] = 1

    return point_predictions
