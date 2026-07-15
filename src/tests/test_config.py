from src.config import Settings, settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.POSTGRES_PORT == 5432
        assert s.POSTGRES_USER == "postgres"
        assert s.POSTGRES_DB == "anomaly_detection"
        assert s.IF_N_ESTIMATORS == 100
        assert s.IF_CONTAMINATION == 0.05
        assert s.DBSCAN_EPS == 0.5
        assert s.DBSCAN_MIN_SAMPLES == 5
        assert s.LSTM_SEQ_LENGTH == 30
        assert s.LSTM_HIDDEN_DIM == 64
        assert s.LSTM_NUM_EPOCHS == 25
        assert s.LSTM_LEARNING_RATE == 0.001
        assert s.ENSEMBLE_MIN_VOTES == 2
        assert s.SHAP_MAX_DISPLAY == 20

    def test_database_url_property(self):
        s = Settings()
        url = s.DATABASE_URL
        assert "postgresql://" in url
        assert s.POSTGRES_USER in url
        assert s.POSTGRES_DB in url

    def test_async_database_url_property(self):
        s = Settings()
        url = s.ASYNC_DATABASE_URL
        assert "postgresql+asyncpg://" in url

    def test_settings_singleton(self):
        assert settings.POSTGRES_HOST is not None
        assert settings.MLFLOW_TRACKING_URI is not None
