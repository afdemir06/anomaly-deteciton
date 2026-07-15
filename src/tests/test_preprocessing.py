import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    create_sequences,
    detect_datetime_column,
    detect_numeric_columns,
    handle_missing_values,
    parse_and_sort_datetime,
    preprocess,
    scale_features,
    validate_row_count,
)


class TestDetectDatetimeColumn:
    def test_keyword_match(self):
        df = pd.DataFrame({"ts": ["2024-01-01"], "value": [1.0]})
        assert detect_datetime_column(df) == "ts"

    def test_case_insensitive(self):
        df = pd.DataFrame({"Timestamp": ["2024-01-01"], "value": [1.0]})
        assert detect_datetime_column(df) == "Timestamp"

    def test_dtype_fallback(self):
        df = pd.DataFrame({"col_a": ["2024-01-01", "2024-01-02"], "value": [1.0, 2.0]})
        assert detect_datetime_column(df) == "col_a"

    def test_no_datetime_raises(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="No datetime column"):
            detect_datetime_column(df)


class TestDetectNumericColumns:
    def test_basic(self):
        df = pd.DataFrame({"ts": ["2024-01-01"], "temp": [22.0], "hum": [50.0]})
        result = detect_numeric_columns(df, exclude=["ts"])
        assert result == ["temp", "hum"]

    def test_excludes_specified(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = detect_numeric_columns(df, exclude=["b"])
        assert "b" not in result

    def test_no_numeric_raises(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        with pytest.raises(ValueError, match="No numeric columns"):
            detect_numeric_columns(df, exclude=[])


class TestParseAndSortDatetime:
    def test_sorts_ascending(self):
        df = pd.DataFrame({
            "ts": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "val": [3, 1, 2],
        })
        result = parse_and_sort_datetime(df, "ts")
        assert list(result["ts"]) == sorted(result["ts"])

    def test_drops_invalid_dates(self):
        df = pd.DataFrame({
            "ts": ["2024-01-01", "invalid", "2024-01-03"],
            "val": [1, 2, 3],
        })
        result = parse_and_sort_datetime(df, "ts")
        assert len(result) == 2


class TestHandleMissingValues:
    def test_interpolation(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        result, cols = handle_missing_values(df, ["a", "b"])
        assert not result["a"].isna().any()
        assert cols == ["a", "b"]

    def test_drops_heavy_nan_column(self):
        df = pd.DataFrame({
            "a": [1.0] * 10,
            "b": [np.nan] * 6 + [1.0] * 4,
        })
        result, cols = handle_missing_values(df, ["a", "b"])
        assert "b" not in result.columns
        assert cols == ["a"]

    def test_all_dropped_raises(self):
        df = pd.DataFrame({
            "a": [np.nan] * 10,
            "b": [np.nan] * 10,
        })
        with pytest.raises(ValueError, match="All numeric columns were dropped"):
            handle_missing_values(df, ["a", "b"])


class TestValidateRowCount:
    def test_too_small_raises(self):
        with pytest.raises(ValueError, match="minimum 3 rows"):
            validate_row_count(2, 30)

    def test_auto_adjust(self):
        result = validate_row_count(10, 30)
        assert result == 9

    def test_no_adjustment_needed(self):
        result = validate_row_count(100, 30)
        assert result == 30

    def test_minimum_seq_length(self):
        result = validate_row_count(3, 30)
        assert result == 2


class TestScaleFeatures:
    def test_returns_scaled_and_scaler(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_scaled, scaler = scale_features(X)
        assert X_scaled.shape == X.shape
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)


class TestCreateSequences:
    def test_normal_case(self):
        X = np.random.randn(20, 3)
        result = create_sequences(X, 5)
        assert result.shape == (16, 5, 3)

    def test_small_input(self):
        X = np.random.randn(3, 2)
        result = create_sequences(X, 30)
        assert result.shape == (3, 1, 2)

    def test_seq_length_1(self):
        X = np.random.randn(10, 4)
        result = create_sequences(X, 1)
        assert result.shape == (10, 1, 4)


class TestPreprocess:
    def test_full_pipeline(self, sample_csv):
        result = preprocess(sample_csv)
        assert "df" in result
        assert "X_scaled" in result
        assert "X_seq" in result
        assert "feature_cols" in result
        assert "seq_length" in result
        assert "scaler" in result
        assert result["X_scaled"].shape[1] == 3

    def test_with_missing_values(self, sample_csv_with_missing):
        result = preprocess(sample_csv_with_missing)
        assert not np.isnan(result["X_scaled"]).any()

    def test_heavy_nan_dropped(self, sample_csv_heavy_nan):
        result = preprocess(sample_csv_heavy_nan)
        assert result["X_scaled"].shape[1] == 1

    def test_small_dataset_raises(self, sample_csv_small):
        with pytest.raises(ValueError, match="minimum 3 rows"):
            preprocess(sample_csv_small)

    def test_short_dataset_adjusts_seq(self, sample_csv_short):
        result = preprocess(sample_csv_short)
        assert result["seq_length"] < 30
