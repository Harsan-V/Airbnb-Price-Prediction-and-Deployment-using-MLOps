from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "airbnb_price_prediction_model.joblib"
TRAINING_MONITORING_PATH = BASE_DIR / "airbnb_monitoring_report.csv"
PREDICTION_LOG_PATH = BASE_DIR / "prediction_logs.csv"

USD_TO_INR = 94.5
LOW_PRICE_WARNING_INR = 500
HIGH_PRICE_WARNING_INR = 50000


class PredictionInput(BaseModel):
    city: Literal["Chennai", "Bangalore", "Hyderabad", "Mumbai", "New Delhi"]
    latitude: float = Field(..., ge=6.0, le=38.0, description="Latitude inside India range")
    longitude: float = Field(..., ge=68.0, le=98.0, description="Longitude inside India range")
    rating: float | None = Field(None, ge=0.0, le=5.0)
    reviews: int | None = Field(None, ge=0)
    bedrooms: float | None = Field(None, ge=0)
    bathrooms: float | None = Field(None, ge=0)
    beds: int | None = Field(None, ge=0)
    guest_capacity: float | None = Field(None, ge=1)
    is_superhost: bool | int = False
    host_rating: float | None = Field(None, ge=0.0, le=5.0)
    amenities_count: int = Field(..., ge=0, le=150)
    has_wifi: bool | int = False
    has_kitchen: bool | int = False
    has_ac: bool | int = False
    has_parking: bool | int = False
    has_washer: bool | int = False
    has_tv: bool | int = False
    has_pool: bool | int = False
    has_workspace: bool | int = False
    description_length: int = Field(..., ge=0, le=10000)
    title_length: int = Field(..., ge=0, le=500)
    property_type_inferred: Literal["Room", "Apartment", "House", "Villa", "Condo", "Other", "Unknown"]

    @field_validator(
        "is_superhost",
        "has_wifi",
        "has_kitchen",
        "has_ac",
        "has_parking",
        "has_washer",
        "has_tv",
        "has_pool",
        "has_workspace",
        mode="before",
    )
    @classmethod
    def convert_bool_to_int(cls, value):
        if isinstance(value, bool):
            return int(value)
        if value in (0, 1):
            return int(value)
        raise ValueError("Boolean feature must be true/false or 0/1")


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR"
    model_name: str
    model_target: str
    warnings: list[str] = []


class ErrorResponse(BaseModel):
    error: str
    detail: object | None = None




def get_best_model_test_mae(metrics, name):
    if not isinstance(metrics, dict):
        return None
    model_metrics = metrics.get(name, {})
    test_mae = model_metrics.get("test_mae")
    try:
        return float(test_mae)
    except (TypeError, ValueError):
        return None


def detect_model_currency(metrics, name, target):
    target_text = str(target).lower()
    if "inr" in target_text:
        return "INR"
    if "usd" in target_text:
        return "USD"

    test_mae = get_best_model_test_mae(metrics, name)
    if test_mae is not None and test_mae < 1000:
        return "USD"
    return "INR"


def convert_prediction_to_inr(prediction, source_currency):
    if source_currency == "USD":
        return prediction * USD_TO_INR
    return prediction

def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    required_keys = {"model", "model_name", "feature_columns", "target"}
    missing_keys = required_keys.difference(artifact.keys())
    if missing_keys:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing_keys)}")
    return artifact


def read_prediction_logs():
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTION_LOG_PATH)


def make_json_safe(df):
    return df.astype(object).where(pd.notna(df), None)


def append_prediction_log(input_data, prediction, warnings):
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "predicted_price_inr": round(float(prediction), 2),
        "warning_count": len(warnings),
        "warnings": " | ".join(warnings),
        **input_data,
    }
    log_df = pd.DataFrame([row])
    write_header = not PREDICTION_LOG_PATH.exists()
    log_df.to_csv(PREDICTION_LOG_PATH, mode="a", index=False, header=write_header)


def build_prediction_warnings(input_data, prediction):
    warnings = []

    if prediction < LOW_PRICE_WARNING_INR:
        warnings.append(f"Predicted price is unusually low: INR {prediction:.2f}")
    if prediction > HIGH_PRICE_WARNING_INR:
        warnings.append(f"Predicted price is unusually high: INR {prediction:.2f}")
    if input_data.get("amenities_count", 0) > 80:
        warnings.append("Amenities count is unusually high compared with normal listing inputs")
    if input_data.get("description_length", 0) > 5000:
        warnings.append("Description length is unusually high and may create unstable predictions")
    if input_data.get("title_length", 0) > 150:
        warnings.append("Title length is unusually high and may create unstable predictions")
    if input_data.get("bedrooms") == 0 and input_data.get("guest_capacity", 1) > 2:
        warnings.append("Guest capacity is high for a listing with zero bedrooms")

    return warnings


try:
    model_artifact = load_model_artifact()
    model = model_artifact["model"]
    model_name = model_artifact["model_name"]
    feature_columns = model_artifact["feature_columns"]
    model_target = model_artifact.get("target", "price_inr")
    model_metrics = model_artifact.get("metrics", {})
    model_source_currency = detect_model_currency(model_metrics, model_name, model_target)
    startup_error = None
except Exception as exc:
    model_artifact = None
    model = None
    model_name = "unavailable"
    feature_columns = []
    model_target = "price_inr"
    model_metrics = {}
    model_source_currency = "unknown"
    startup_error = str(exc)


app = FastAPI(
    title="Airbnb Hotel Stay Price Prediction ",
    description="This is a hotel stay prediction app using fastapi to predict room stay price in India.",
    version="1.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Airbnb India Price Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "monitoring": "/monitoring/summary",
    }


@app.get("/health", tags=["Health"])
def health_check():
    if startup_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"model_loaded": False, "error": startup_error},
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": model_name,
        "target": "price_inr",
        "model_source_currency": model_source_currency,
        "usd_to_inr_rate": USD_TO_INR if model_source_currency == "USD" else None,
        "features_count": len(feature_columns),
        "prediction_log_exists": PREDICTION_LOG_PATH.exists(),
    }


@app.get("/features", tags=["Model"])
def get_features():
    if startup_error:
        raise HTTPException(status_code=503, detail=startup_error)
    return {"feature_columns": feature_columns}


@app.get("/model/metrics", tags=["Model"])
def get_model_metrics():
    if startup_error:
        raise HTTPException(status_code=503, detail=startup_error)
    return {
        "model_name": model_name,
        "model_source_currency": model_source_currency,
        "api_output_currency": "INR",
        "usd_to_inr_rate": USD_TO_INR if model_source_currency == "USD" else None,
        "metrics": model_metrics,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Prediction"],
)
def predict_price(payload: PredictionInput):
    if startup_error or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"model_loaded": False, "error": startup_error},
        )

    try:
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])[feature_columns]
        raw_prediction = float(model.predict(input_df)[0])
        prediction = convert_prediction_to_inr(raw_prediction, model_source_currency)

        if prediction <= 0:
            raise ValueError(f"Invalid predicted price: {prediction}")

        warnings = build_prediction_warnings(input_dict, prediction)
        if model_source_currency == "USD":
            warnings.append(f"Model predicted {raw_prediction:.2f} USD; API converted it to INR using rate {USD_TO_INR}")
        try:
            append_prediction_log(input_dict, prediction, warnings)
        except Exception as log_error:
            warnings.append(f"Prediction succeeded, but monitoring log failed: {log_error}")

        return PredictionResponse(
            predicted_price=round(prediction, 2),
            currency="INR",
            model_name=model_name,
            model_target="price_inr",
            warnings=warnings,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required model feature: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.get("/monitoring/summary", tags=["Monitoring"])
def monitoring_summary():
    logs = read_prediction_logs()
    if logs.empty:
        return {
            "message": "No predictions logged yet. Call /predict first.",
            "total_predictions": 0,
            "model_name": model_name,
            "model_target": "price_inr",
            "model_source_currency": model_source_currency,
        }

    city_counts = logs["city"].value_counts().to_dict() if "city" in logs else {}
    warning_rows = int((logs.get("warning_count", pd.Series(dtype=int)) > 0).sum())

    return {
        "total_predictions": int(len(logs)),
        "model_name": model_name,
        "model_target": "price_inr",
        "model_source_currency": model_source_currency,
        "average_predicted_price_inr": round(float(logs["predicted_price_inr"].mean()), 2),
        "min_predicted_price_inr": round(float(logs["predicted_price_inr"].min()), 2),
        "max_predicted_price_inr": round(float(logs["predicted_price_inr"].max()), 2),
        "warning_rows": warning_rows,
        "city_counts": city_counts,
        "latest_prediction": make_json_safe(logs.tail(1)).to_dict(orient="records")[0],
    }


@app.get("/monitoring/logs", tags=["Monitoring"])
def monitoring_logs(limit: int = Query(20, ge=1, le=100)):
    logs = read_prediction_logs()
    if logs.empty:
        return {"total_predictions": 0, "logs": []}
    recent_logs = make_json_safe(logs.tail(limit).sort_index(ascending=False))
    return {
        "total_predictions": int(len(logs)),
        "returned": int(len(recent_logs)),
        "logs": recent_logs.to_dict(orient="records"),
    }


@app.get("/monitoring/training-report", tags=["Monitoring"])
def training_monitoring_report():
    if not TRAINING_MONITORING_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training monitoring report not found",
        )
    report = pd.read_csv(TRAINING_MONITORING_PATH)
    safe_report = make_json_safe(report)
    return {
        "rows": int(len(safe_report)),
        "status_counts": safe_report["status"].value_counts().to_dict() if "status" in safe_report else {},
        "report": safe_report.to_dict(orient="records"),
    }
