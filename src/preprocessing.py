import logging

import numpy as np
import pandas as pd

try:
    from cuml.preprocessing import StandardScaler
except ImportError:
    from sklearn.preprocessing import StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)

DATETIME_KEYWORDS = [
    "date",
    "time",
    "year",
    "month",
    "day",
    "hour",
    "timestamp",
    "datetime",
    "ts",
]

MIN_ROWS = 3


def detect_datetime_column(df: pd.DataFrame) -> str:
    """Detect the datetime column by name keywords or object dtype."""
    lower_cols = {c.lower(): c for c in df.columns}

    for keyword in DATETIME_KEYWORDS:
        for col_lower, col_original in lower_cols.items():
            if keyword in col_lower:
                if pd.api.types.is_numeric_dtype(df[col_original]):
                    logger.info(
                        "Datetime column '%s' is numeric, converting seconds to datetime",
                        col_original,
                    )
                else:
                    logger.info("Datetime column detected by keyword '%s': %s", keyword, col_original)
                return col_original

    for col in df.columns:
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].head(10))
                logger.info("Datetime column detected by dtype parsing: %s", col)
                return col
            except (ValueError, TypeError):
                continue

    raise ValueError("No datetime column detected in the dataset")


def detect_numeric_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Detect numeric columns, excluding the specified ones."""
    numeric_dtypes = ("int64", "float64", "int32", "float32")
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype.name in numeric_dtypes and c not in exclude
    ]
    if not numeric_cols:
        raise ValueError("No numeric columns found in the dataset")
    return numeric_cols


def parse_and_sort_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse datetime column and sort the dataframe by time."""
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], unit="s", origin="unix")
        logger.info("Converted numeric column '%s' to datetime (seconds since epoch)", col)
    else:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        n_invalid = df[col].isna().sum()
        if n_invalid > 0:
            logger.warning("Dropped %d rows with unparseable datetime values", n_invalid)
            df = df.dropna(subset=[col])
    df = df.sort_values(by=col).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Handle missing values using interpolation, drop columns with >50% NaN."""
    df = df.copy()
    cols_to_drop = [
        c for c in numeric_cols
        if df[c].isna().mean() > 0.5
    ]
    if cols_to_drop:
        logger.warning("Dropping columns with >50%% NaN: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
        numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]

    if not numeric_cols:
        raise ValueError("All numeric columns were dropped due to excessive missing values")

    for c in numeric_cols:
        df[c] = df[c].interpolate(method="linear")
        df[c] = df[c].ffill().bfill()
    return df, numeric_cols


def validate_row_count(n_rows: int, seq_length: int) -> int:
    """Validate row count against seq_length, auto-adjust if needed."""
    if n_rows < MIN_ROWS:
        raise ValueError(f"Dataset too small, minimum {MIN_ROWS} rows required, got {n_rows}")

    if n_rows < seq_length:
        new_seq_length = max(2, n_rows - 1)
        logger.info(
            "Row count (%d) < seq_length (%d). Adjusting seq_length to %d",
            n_rows, seq_length, new_seq_length,
        )
        return new_seq_length
    return seq_length


def scale_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Scale features using StandardScaler. Returns scaled array and fitted scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def create_sequences(X: np.ndarray, seq_length: int) -> np.ndarray:
    """Create sliding window sequences for LSTM. Shape: (n_samples, seq_length, n_features)."""
    if seq_length <= 1 or X.shape[0] <= seq_length:
        return X.reshape(X.shape[0], 1, X.shape[1])

    sequences = []
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i : i + seq_length])
    return np.array(sequences)


def preprocess(file_path: str, label_column: str = "") -> dict:
    """Full preprocessing pipeline. Returns dict ready for all models."""
    df = pd.read_csv(file_path)

    datetime_col = detect_datetime_column(df)
    df = parse_and_sort_datetime(df, datetime_col)

    exclude = [datetime_col]
    if label_column and label_column in df.columns:
        exclude.append(label_column)
        logger.info("Excluding label column: %s", label_column)

    feature_cols = detect_numeric_columns(df, exclude=exclude)

    df, feature_cols = handle_missing_values(df, feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("No usable numeric columns remain after preprocessing")

    X = df[feature_cols].to_numpy(dtype=np.float32)

    seq_length = validate_row_count(X.shape[0], settings.LSTM_SEQ_LENGTH)

    X_scaled, scaler = scale_features(X)
    X_seq = create_sequences(X_scaled, seq_length)

    logger.info(
        "Preprocessing complete: %d rows, %d features, seq_length=%d",
        X_scaled.shape[0], X_scaled.shape[1], seq_length,
    )

    return {
        "df": df,
        "datetime_col": datetime_col,
        "feature_cols": feature_cols,
        "X_scaled": X_scaled,
        "X_seq": X_seq,
        "seq_length": seq_length,
        "scaler": scaler,
    }
