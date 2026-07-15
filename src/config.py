from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── PostgreSQL ──
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "anomaly_detection"

    # ── MLflow ──
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "anomaly-detection"

    # ── Isolation Forest ──
    IF_N_ESTIMATORS: int = 100
    IF_CONTAMINATION: float = 0.05
    IF_RANDOM_STATE: int = 42

    # ── DBSCAN ──
    DBSCAN_EPS: float = 0.5
    DBSCAN_MIN_SAMPLES: int = 5
    DBSCAN_METRIC: str = "euclidean"

    # ── LSTM Autoencoder ──
    LSTM_SEQ_LENGTH: int = 30
    LSTM_HIDDEN_DIM: int = 64
    LSTM_NUM_EPOCHS: int = 50
    LSTM_LEARNING_RATE: float = 0.001
    LSTM_THRESHOLD_PERCENTILE: float = 95.0

    # ── Ensemble (Majority Voting) ──
    ENSEMBLE_MIN_VOTES: int = 2

    # ── Label ──
    LABEL_COLUMN: str = "Class"

    # ── SHAP ──
    SHAP_MAX_DISPLAY: int = 20
    SHAP_SAMPLE_SIZE: int = 100

    # ── DBSCAN Subsample ──
    DBSCAN_SUBSAMPLE_SIZE: int = 10000

    # ── Docker Service URLs ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MLFLOW_HOST: str = "mlflow"
    MLFLOW_PORT: int = 5000

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
