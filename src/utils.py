import json
import logging
from pathlib import Path

import joblib
import torch

from src.models.lstm_autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def save_models(
    run_id: str,
    if_model,
    dbscan_model,
    lstm_model,
    metadata: dict,
) -> None:
    """Save all models and metadata to disk."""
    model_dir = MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(if_model.model, model_dir / "isolation_forest.joblib")
    joblib.dump(dbscan_model.model, model_dir / "dbscan.joblib")
    joblib.dump(dbscan_model.nn_model, model_dir / "dbscan_nn.joblib")

    torch.save(
        lstm_model.model.state_dict(),
        model_dir / "lstm_autoencoder.pt",
    )
    joblib.dump(lstm_model.threshold, model_dir / "lstm_threshold.joblib")
    joblib.dump(
        {
            "n_features": lstm_model.model.output_layer.out_features,
            "hidden_dim": lstm_model.params["hidden_dim"],
            "seq_length": metadata.get("seq_length", lstm_model.params["seq_length"]),
        },
        model_dir / "lstm_config.joblib",
    )

    meta_json = {
        "feature_cols": metadata.get("feature_cols", []),
        "datetime_col": metadata.get("datetime_col", ""),
        "seq_length": metadata.get("seq_length", 0),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta_json, f)

    logger.info("Models saved to %s", model_dir)


def load_models(run_id: str) -> dict:
    """Load all models and metadata from disk. Returns dict with models and metadata."""
    model_dir = MODELS_DIR / run_id
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No trained model found for run {run_id}. Call train() first."
        )

    from src.models.dbscan import DBSCANModel
    from src.models.isolation_forest import IsolationForestModel
    from src.models.lstm_autoencoder import LSTMAutoencoderModel

    if_model = IsolationForestModel()
    if_model.model = joblib.load(model_dir / "isolation_forest.joblib")

    dbscan_model = DBSCANModel()
    dbscan_model.model = joblib.load(model_dir / "dbscan.joblib")
    dbscan_model.nn_model = joblib.load(model_dir / "dbscan_nn.joblib")

    lstm_model = LSTMAutoencoderModel()
    lstm_config = joblib.load(model_dir / "lstm_config.joblib")
    lstm_model.model = LSTMAutoencoder(
        n_features=lstm_config["n_features"],
        hidden_dim=lstm_config["hidden_dim"],
        seq_length=lstm_config["seq_length"],
    )
    lstm_model.model.load_state_dict(
        torch.load(model_dir / "lstm_autoencoder.pt", weights_only=True)
    )
    lstm_model.model.eval()
    lstm_model.threshold = joblib.load(model_dir / "lstm_threshold.joblib")

    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)

    metadata = {
        "run_id": run_id,
        "feature_cols": meta["feature_cols"],
        "datetime_col": meta["datetime_col"],
        "seq_length": meta["seq_length"],
    }

    logger.info("Models loaded from %s", model_dir)

    return {
        "if_model": if_model,
        "dbscan_model": dbscan_model,
        "lstm_model": lstm_model,
        "metadata": metadata,
    }
