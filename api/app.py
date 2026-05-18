"""FastAPI inference service for Emigria MLP + PMI risk layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pmi_rules import combine_ml_and_pmi_rules
from .preprocessing import FEATURE_COLUMNS, prepare_base_dataframe, transform_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "emigria_mlp_model.keras"
TARGET_ENCODER_PATH = MODEL_DIR / "target_encoder.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"
COUNTRY_SALARY_AVG_PATH = MODEL_DIR / "country_salary_avg.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"
PREPROCESSING_CONFIG_PATH = MODEL_DIR / "preprocessing_config.json"


class JobPostingRequest(BaseModel):
    title: str = ""
    location: str = ""
    country: str = ""
    salary_range: str = ""
    description: str = ""
    requirements: str = ""
    company_profile: str = ""
    employment_type: str = "Unknown"
    industry: str = "Unknown"
    benefits: str = ""
    required_experience: str = "Not Specified"
    required_education: str = "Not Specified"
    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    records: list[JobPostingRequest]


app = FastAPI(
    title="Emigria AI Service",
    description="MLP fraud detection with PMI-specific risk rules.",
    version="1.0.0",
)


model: tf.keras.Model | None = None
target_encoder: Any = None
tfidf_vectorizer: Any = None
scaler: Any = None
country_salary_data: dict[str, Any] | None = None
feature_columns: list[str] = FEATURE_COLUMNS
threshold: float = 0.5
preprocessing_config: dict[str, Any] = {}


def _missing_artifacts() -> list[str]:
    required_paths = [
        MODEL_PATH,
        TARGET_ENCODER_PATH,
        TFIDF_PATH,
        SCALER_PATH,
        COUNTRY_SALARY_AVG_PATH,
        THRESHOLD_PATH,
    ]
    return [str(path.relative_to(BASE_DIR)) for path in required_paths if not path.exists()]


def load_artifacts() -> None:
    global model
    global target_encoder
    global tfidf_vectorizer
    global scaler
    global country_salary_data
    global feature_columns
    global threshold
    global preprocessing_config

    missing = _missing_artifacts()
    if missing:
        raise FileNotFoundError(
            "Artifact model belum lengkap. Taruh file berikut di api/models/: "
            + ", ".join(missing)
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    country_salary_data = joblib.load(COUNTRY_SALARY_AVG_PATH)

    if FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
            feature_columns = json.load(file)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold = float(file.read().strip())

    if PREPROCESSING_CONFIG_PATH.exists():
        with open(PREPROCESSING_CONFIG_PATH, "r", encoding="utf-8") as file:
            preprocessing_config = json.load(file)


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_artifacts()
    except FileNotFoundError as exc:
        # Keep API startable so /health can explain which artifact is missing.
        print(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    missing = _missing_artifacts()
    return {
        "status": "ready" if not missing and model is not None else "missing_artifacts",
        "model_dir": str(MODEL_DIR),
        "missing_artifacts": missing,
        "threshold": threshold,
    }


def _ensure_loaded() -> None:
    if model is None:
        try:
            load_artifacts()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


def _record_to_dict(record: JobPostingRequest) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        data = record.model_dump()
    else:
        data = record.dict()
    extra = data.pop("extra", {}) or {}
    data.update(extra)
    return data


def predict_one(record: dict[str, Any]) -> dict[str, Any]:
    _ensure_loaded()
    assert model is not None
    assert country_salary_data is not None

    raw_df = pd.DataFrame([record])
    base_df = prepare_base_dataframe(raw_df, has_target=False)
    scaled_features = transform_features(
        base_df,
        target_encoder=target_encoder,
        tfidf_vectorizer=tfidf_vectorizer,
        scaler=scaler,
        country_salary_data=country_salary_data,
        feature_columns=feature_columns,
    )

    ml_probability = float(model.predict(scaled_features, verbose=0).ravel()[0])
    result = combine_ml_and_pmi_rules(
        record=record,
        ml_probability=ml_probability,
        threshold=threshold,
    )
    return result


@app.post("/predict")
def predict(request: JobPostingRequest) -> dict[str, Any]:
    record = _record_to_dict(request)
    return predict_one(record)


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    records = [_record_to_dict(record) for record in request.records]
    return {"predictions": [predict_one(record) for record in records]}
