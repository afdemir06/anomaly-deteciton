import logging

import numpy as np
import shap

from src.config import settings

logger = logging.getLogger(__name__)

SHAP_SAMPLE_SIZE = settings.SHAP_SAMPLE_SIZE
SHAP_KERNEL_NSAMPLES = 200


def _get_shap_values(if_model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values, falling back to KernelExplainer for single-feature data.

    SHAP >=0.50 TreeExplainer crashes on IsolationForest with n_features=1
    (IndexError: index -2 is out of bounds for axis 0 with size 1).
    """
    n_features = X.shape[1]

    if n_features == 1:
        return _shap_kernel(if_model, X)

    try:
        return _shap_tree(if_model, X)
    except Exception:
        logger.warning(
            "TreeExplainer failed (n_features=%d), falling back to KernelExplainer",
            n_features,
        )
        return _shap_kernel(if_model, X)


def _shap_tree(if_model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(if_model.model)
    raw = explainer.shap_values(X)
    return _normalize_shap_output(raw, X.shape)


def _shap_kernel(if_model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values using KernelExplainer with score_samples."""
    background = shap.sample(X, min(50, len(X)))

    def score_fn(data):
        return if_model.model.score_samples(data)

    explainer = shap.KernelExplainer(score_fn, background)
    raw = explainer.shap_values(X, nsamples=SHAP_KERNEL_NSAMPLES)
    return _normalize_shap_output(raw, X.shape)


def _normalize_shap_output(raw, expected_shape: tuple) -> np.ndarray:
    """Normalize any SHAP return type to a plain 2D numpy array (n_samples, n_features)."""
    if isinstance(raw, list):
        raw = raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim == 1:
        raw = raw.reshape(-1, expected_shape[1])
    return raw


def get_top_contributors(
    shap_values: np.ndarray,
    feature_cols: list[str],
    top_n: int = settings.SHAP_MAX_DISPLAY,
) -> list[dict]:
    """Return top N features sorted by absolute SHAP value."""
    shap_values = np.asarray(shap_values).flatten()
    n_features = len(feature_cols)
    if n_features == 1:
        return [
            {
                "feature": feature_cols[0],
                "shap_value": float(shap_values[0]),
            }
        ]
    indices = np.argsort(np.abs(shap_values))[::-1][: min(top_n, n_features)]
    return [
        {
            "feature": feature_cols[i],
            "shap_value": float(shap_values[i]),
        }
        for i in indices
    ]


def explain_anomalies(
    if_model,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    ensemble_preds: np.ndarray,
    if_preds: np.ndarray,
) -> list[dict]:
    """Generate SHAP explanations for anomalies.

    - ensemble=1 AND if=1 -> compute SHAP, return top contributors
    - ensemble=1 AND if=0 -> return message: IF did not flag this point
    - ensemble=0 -> skip (not anomalous)
    """
    n_samples = len(ensemble_preds)
    explanations: list[dict] = [None] * n_samples

    anomaly_indices = np.where(ensemble_preds == 1)[0]
    if len(anomaly_indices) == 0:
        return explanations

    explainable_indices = anomaly_indices[if_preds[anomaly_indices] == 1]
    non_explainable_indices = anomaly_indices[if_preds[anomaly_indices] == 0]

    if len(explainable_indices) > 0:
        if len(explainable_indices) > SHAP_SAMPLE_SIZE:
            rng = np.random.default_rng(42)
            selected = np.sort(
                rng.choice(explainable_indices, size=SHAP_SAMPLE_SIZE, replace=False)
            )
            logger.info(
                "Subsampled %d -> %d anomalies for SHAP",
                len(explainable_indices),
                SHAP_SAMPLE_SIZE,
            )
        else:
            selected = explainable_indices

        shap_values = _get_shap_values(if_model, X_scaled[selected])

        for idx, row_idx in enumerate(selected):
            explanations[row_idx] = {
                "row_index": int(row_idx),
                "is_anomaly": True,
                "top_features": get_top_contributors(
                    shap_values[idx], feature_cols
                ),
                "message": None,
            }

        logger.info("SHAP explanations computed for %d anomalies", len(selected))

    for idx in non_explainable_indices:
        explanations[idx] = {
            "row_index": int(idx),
            "is_anomaly": True,
            "top_features": [],
            "message": "IF did not flag this point, no explanation available",
        }

    return explanations
