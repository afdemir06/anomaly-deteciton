import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file with datetime and numeric columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": np.random.uniform(20, 30, 100),
        "humidity": np.random.uniform(40, 80, 100),
        "pressure": np.random.uniform(1000, 1020, 100),
    })
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_with_missing():
    """Create a CSV with missing values in numeric columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": np.random.uniform(20, 30, 100),
        "humidity": np.random.uniform(40, 80, 100),
    })
    df.loc[0:10, "temperature"] = np.nan
    df.loc[50:60, "humidity"] = np.nan
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_heavy_nan():
    """Create a CSV where one column has >50% NaN (should be dropped)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": np.random.uniform(20, 30, 100),
        "bad_column": [np.nan] * 60 + list(np.random.uniform(0, 1, 40)),
    })
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_small():
    """Create a CSV with too few rows (< MIN_ROWS)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
        "temperature": [22.0, 23.0],
    })
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_short():
    """Create a CSV with fewer rows than seq_length."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "temperature": np.random.uniform(20, 30, 5),
        "humidity": np.random.uniform(40, 80, 5),
    })
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def scaled_data():
    """Return pre-scaled numpy array for model tests."""
    np.random.seed(42)
    return np.random.randn(100, 3).astype(np.float32)


@pytest.fixture
def sequence_data():
    """Return sequence data (n_samples, seq_length, n_features) for LSTM tests."""
    np.random.seed(42)
    return np.random.randn(50, 10, 3).astype(np.float32)
