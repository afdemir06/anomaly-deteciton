import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    JSON,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file_name = Column(String, nullable=False)
    n_rows = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    feature_cols = Column(JSON, nullable=False)
    datetime_col = Column(String, nullable=False)
    seq_length = Column(Integer, nullable=False)
    if_metrics = Column(JSON)
    dbscan_metrics = Column(JSON)
    lstm_metrics = Column(JSON)
    ensemble_metrics = Column(JSON)


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False, index=True)
    row_index = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    feature_values = Column(JSON)
    is_anomaly = Column(Boolean, nullable=False)
    anomaly_score = Column(Float)
    if_vote = Column(Integer)
    dbscan_vote = Column(Integer)
    lstm_vote = Column(Integer)
    shap_explanation = Column(JSON, nullable=True)
    ground_truth = Column(Integer, nullable=True)


engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(engine)
    logger.info("Database tables created.")


def save_run(
    run_id: str,
    file_name: str,
    n_rows: int,
    n_features: int,
    feature_cols: list[str],
    datetime_col: str,
    seq_length: int,
    if_metrics: dict,
    dbscan_metrics: dict,
    lstm_metrics: dict,
    ensemble_metrics: dict,
) -> str:
    """Insert a run record and return its ID."""
    with Session(engine) as session:
        run = Run(
            id=run_id,
            file_name=file_name,
            n_rows=n_rows,
            n_features=n_features,
            feature_cols=feature_cols,
            datetime_col=datetime_col,
            seq_length=seq_length,
            if_metrics=if_metrics,
            dbscan_metrics=dbscan_metrics,
            lstm_metrics=lstm_metrics,
            ensemble_metrics=ensemble_metrics,
        )
        session.add(run)
        session.commit()
        logger.info("Run saved: %s", run_id)
    return run_id


def save_results(run_id: str, results: list[dict]) -> None:
    """Bulk insert result rows for a given run."""
    with Session(engine) as session:
        objects = [
            Result(
                run_id=run_id,
                row_index=r["row_index"],
                timestamp=r.get("timestamp"),
                feature_values=r.get("feature_values"),
                is_anomaly=r["is_anomaly"],
                anomaly_score=r.get("anomaly_score"),
                if_vote=r.get("if_vote"),
                dbscan_vote=r.get("dbscan_vote"),
                lstm_vote=r.get("lstm_vote"),
                shap_explanation=r.get("shap_explanation"),
                ground_truth=r.get("ground_truth"),
            )
            for r in results
        ]
        session.add_all(objects)
        session.commit()
        logger.info("Saved %d results for run %s", len(results), run_id)


def get_latest_run_id() -> str | None:
    """Return the most recent run_id, or None if no runs exist."""
    with Session(engine) as session:
        run = session.query(Run).order_by(Run.created_at.desc()).first()
        return run.id if run else None


def get_run(run_id: str) -> dict | None:
    """Return run metadata as dict, or None if not found."""
    with Session(engine) as session:
        run = session.query(Run).filter(Run.id == run_id).first()
        if not run:
            return None
        return {
            "id": run.id,
            "created_at": run.created_at,
            "file_name": run.file_name,
            "n_rows": run.n_rows,
            "n_features": run.n_features,
            "feature_cols": run.feature_cols,
            "datetime_col": run.datetime_col,
            "seq_length": run.seq_length,
            "if_metrics": run.if_metrics,
            "dbscan_metrics": run.dbscan_metrics,
            "lstm_metrics": run.lstm_metrics,
            "ensemble_metrics": run.ensemble_metrics,
        }


def get_results(run_id: str) -> list[dict]:
    """Return all results for a given run."""
    with Session(engine) as session:
        rows = session.query(Result).filter(Result.run_id == run_id).all()
        return [
            {
                "row_index": r.row_index,
                "timestamp": r.timestamp,
                "feature_values": r.feature_values,
                "is_anomaly": r.is_anomaly,
                "anomaly_score": r.anomaly_score,
                "if_vote": r.if_vote,
                "dbscan_vote": r.dbscan_vote,
                "lstm_vote": r.lstm_vote,
                "shap_explanation": r.shap_explanation,
                "ground_truth": r.ground_truth,
            }
            for r in rows
        ]
