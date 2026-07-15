import pytest
from unittest.mock import patch

from sqlalchemy import inspect as sa_inspect

from src import db


class TestDatabaseOperations:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Use SQLite in-memory for each test."""
        from sqlalchemy import create_engine

        self.test_engine = create_engine("sqlite:///:memory:")
        db.Base.metadata.create_all(self.test_engine)

        with patch.object(db, "engine", self.test_engine):
            yield

    def test_init_db(self):
        db.init_db()
        inspector = sa_inspect(self.test_engine)
        assert inspector.has_table("runs")
        assert inspector.has_table("results")

    def test_save_run(self):
        run_id = db.save_run(
            run_id="test-run-1",
            file_name="data.csv",
            n_rows=100,
            n_features=3,
            feature_cols=["a", "b", "c"],
            datetime_col="ts",
            seq_length=30,
            if_metrics={"n_anomalies": 5},
            dbscan_metrics={"n_anomalies": 3},
            lstm_metrics={"n_anomalies": 4},
            ensemble_metrics={"n_anomalies": 2},
        )
        assert run_id == "test-run-1"

    def test_get_run(self):
        db.save_run(
            run_id="test-run-2",
            file_name="data.csv",
            n_rows=100,
            n_features=3,
            feature_cols=["a", "b", "c"],
            datetime_col="ts",
            seq_length=30,
            if_metrics={},
            dbscan_metrics={},
            lstm_metrics={},
            ensemble_metrics={},
        )
        run = db.get_run("test-run-2")
        assert run is not None
        assert run["id"] == "test-run-2"
        assert run["file_name"] == "data.csv"
        assert run["n_rows"] == 100

    def test_get_run_not_found(self):
        assert db.get_run("nonexistent") is None

    def test_save_results(self):
        db.save_run(
            run_id="test-run-3",
            file_name="data.csv",
            n_rows=3,
            n_features=2,
            feature_cols=["a", "b"],
            datetime_col="ts",
            seq_length=5,
            if_metrics={},
            dbscan_metrics={},
            lstm_metrics={},
            ensemble_metrics={},
        )
        results = [
            {"row_index": 0, "is_anomaly": True, "if_vote": 1, "dbscan_vote": 1, "lstm_vote": 0},
            {"row_index": 1, "is_anomaly": False, "if_vote": 0, "dbscan_vote": 0, "lstm_vote": 0},
        ]
        db.save_results("test-run-3", results)

        fetched = db.get_results("test-run-3")
        assert len(fetched) == 2
        assert fetched[0]["is_anomaly"] is True
        assert fetched[1]["is_anomaly"] is False

    def test_get_latest_run_id(self):
        assert db.get_latest_run_id() is None

        db.save_run(
            run_id="run-a",
            file_name="a.csv",
            n_rows=10, n_features=2,
            feature_cols=["x", "y"],
            datetime_col="ts", seq_length=5,
            if_metrics={}, dbscan_metrics={},
            lstm_metrics={}, ensemble_metrics={},
        )
        db.save_run(
            run_id="run-b",
            file_name="b.csv",
            n_rows=10, n_features=2,
            feature_cols=["x", "y"],
            datetime_col="ts", seq_length=5,
            if_metrics={}, dbscan_metrics={},
            lstm_metrics={}, ensemble_metrics={},
        )
        latest = db.get_latest_run_id()
        assert latest in ("run-a", "run-b")

    def test_get_results_empty(self):
        assert db.get_results("nonexistent") == []
