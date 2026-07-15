import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src import db
from src.pipeline import AnomalyDetectionPipeline

logger = logging.getLogger(__name__)

pipeline: AnomalyDetectionPipeline = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    db.init_db()
    pipeline = AnomalyDetectionPipeline()
    logger.info("API started, pipeline initialized.")
    yield
    logger.info("API shutting down.")


app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)


@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file.filename or "data.csv")
            with open(file_path, "wb") as f:
                f.write(await file.read())
            result = pipeline.train(file_path)
            return result
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect(run_id: Optional[str] = Query(None)):
    try:
        result = pipeline.detect(run_id)
        results = db.get_results(result["run_id"])
        return {
            "run_id": result["run_id"],
            "predictions": result["predictions"].tolist(),
            "if_votes": result["if_votes"].tolist(),
            "dbscan_votes": result["dbscan_votes"].tolist(),
            "lstm_votes": result["lstm_votes"].tolist(),
            "timestamps": [str(r["timestamp"]) if r["timestamp"] else None for r in results],
            "feature_values": [r["feature_values"] for r in results],
            "is_anomaly": [r["is_anomaly"] for r in results],
            "anomaly_score": [r["anomaly_score"] for r in results],
            "ground_truth": [r.get("ground_truth") for r in results],
        }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Detect failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/file")
async def detect_file(
    file: UploadFile = File(...),
    run_id: Optional[str] = Query(None),
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, file.filename or "data.csv")
            with open(file_path, "wb") as f:
                f.write(await file.read())
            result = pipeline.detect_from_file(file_path, run_id)
            return {
                "run_id": result["run_id"],
                "predictions": result["predictions"].tolist(),
                "if_votes": result["if_votes"].tolist(),
                "dbscan_votes": result["dbscan_votes"].tolist(),
                "lstm_votes": result["lstm_votes"].tolist(),
            }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Detect from file failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detect/explain")
async def detect_explain(run_id: Optional[str] = Query(None)):
    try:
        if run_id is None:
            run_id = db.get_latest_run_id()
            if run_id is None:
                raise FileNotFoundError("No trained model found. Call train() first.")
        results = db.get_results(run_id)
        if not results:
            raise ValueError(f"No results found for run {run_id}.")
        explanations = [
            r for r in results
            if r.get("shap_explanation") is not None
        ]
        return {
            "run_id": run_id,
            "n_anomalies_explained": len(explanations),
            "explanations": explanations,
        }
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info(run_id: Optional[str] = Query(None)):
    try:
        if run_id is None:
            run_id = db.get_latest_run_id()
            if run_id is None:
                raise FileNotFoundError("No trained model found. Call train() first.")
        run = db.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found.")
        return run
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Model info failed")
        raise HTTPException(status_code=500, detail=str(e))
