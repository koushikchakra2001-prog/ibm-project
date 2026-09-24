"""
main.py  –  FastAPI Backend
---------------------------
Endpoints:
  POST /upload          – upload CSV, triggers preprocessing + model training
  GET  /summary         – dataset KPIs (totals, brand breakdown, state breakdown)
  GET  /forecast        – Prophet + XGBoost forecasts up to 2027
  GET  /brand-forecast  – per-brand Prophet forecast to 2027
  GET  /health          – liveness probe
"""
import io
import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Sibling imports
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    load_csv, clean, aggregate_monthly, prepare_features,
    aggregate_by_brand, aggregate_by_state, aggregate_by_category, get_yearly_trend,
)
from model import (
    train_prophet, train_xgb, forecast_prophet, forecast_xgb, load_models,
)

# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bike Sales India – Prediction API",
    description="Upload bike_sales_india.csv to train models and get 2027 forecasts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state (reset on every /upload)
# ---------------------------------------------------------------------------
_state: dict = {
    "df": None,
    "monthly": None,
    "prophet": None,
    "xgb": None,
    "scaler": None,
    "metrics": {},
    "data_path": None,
}


def _require_models():
    if _state["prophet"] is None or _state["xgb"] is None:
        raise HTTPException(
            status_code=400,
            detail="Models not trained. POST a CSV to /upload first.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_trained": _state["prophet"] is not None}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a CSV file, preprocess it, and train forecasting models."""
    contents = await file.read()

    # Save to a temp file so data_loader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        raw = load_csv(tmp_path)
        df = clean(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {e}")
    finally:
        os.unlink(tmp_path)

    if len(df) < 24:
        raise HTTPException(
            status_code=422,
            detail=f"Too few rows ({len(df)}) after cleaning. Need at least 24.",
        )

    monthly = aggregate_monthly(df)
    feat_df = prepare_features(monthly)

    # Train
    p_result = train_prophet(monthly)
    x_result = train_xgb(feat_df)

    # Store state
    _state["df"] = df
    _state["monthly"] = monthly
    _state["prophet"] = p_result["model"]
    _state["xgb"] = x_result["model"]
    _state["scaler"] = x_result["scaler"]
    _state["metrics"] = {
        "prophet": p_result["metrics"],
        "xgboost": x_result["metrics"],
    }

    return {
        "message": "Training complete",
        "rows": len(df),
        "monthly_observations": len(monthly),
        "date_range": {
            "start": str(monthly["ds"].min().date()),
            "end": str(monthly["ds"].max().date()),
        },
        "metrics": _state["metrics"],
    }


@app.get("/summary")
def summary():
    """Return dataset-wide KPIs and breakdowns."""
    _require_models()
    df: pd.DataFrame = _state["df"]

    result = {
        "total_units_sold": int(df["units_sold"].sum()),
        "total_records": len(df),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
    }

    brand_df = aggregate_by_brand(df)
    if not brand_df.empty:
        result["top_brands"] = brand_df.head(10).to_dict(orient="records")

    state_df = aggregate_by_state(df)
    if not state_df.empty:
        result["top_states"] = state_df.head(10).to_dict(orient="records")

    cat_df = aggregate_by_category(df)
    if not cat_df.empty:
        result["categories"] = cat_df.to_dict(orient="records")

    yearly = get_yearly_trend(df)
    result["yearly_trend"] = yearly.to_dict(orient="records")

    monthly = _state["monthly"]
    result["monthly_history"] = monthly.rename(
        columns={"ds": "date", "y": "units_sold"}
    ).assign(date=monthly["ds"].dt.strftime("%Y-%m")).to_dict(orient="records")

    return result


@app.get("/forecast")
def forecast(horizon: int = Query(default=36, ge=1, le=60)):
    """
    Return Prophet and XGBoost forecasts for `horizon` months ahead.
    Default: 36 months (takes predictions to ~2027).
    """
    _require_models()

    monthly = _state["monthly"]

    # Prophet forecast
    p_fc = forecast_prophet(_state["prophet"], periods=horizon)
    p_fc["ds"] = p_fc["ds"].dt.strftime("%Y-%m")

    # XGBoost forecast  (only future months)
    x_fc = forecast_xgb(_state["xgb"], _state["scaler"], monthly, horizon_months=horizon)
    x_fc["ds"] = x_fc["ds"].dt.strftime("%Y-%m")

    # Only return future rows from prophet (history already in /summary)
    hist_end = monthly["ds"].max().strftime("%Y-%m")
    p_future = p_fc[p_fc["ds"] > hist_end]
    x_future = x_fc[x_fc["ds"] > hist_end]

    return {
        "horizon_months": horizon,
        "metrics": _state["metrics"],
        "prophet_forecast": _nan_safe(p_future.to_dict(orient="records")),
        "xgboost_forecast": _nan_safe(x_future.to_dict(orient="records")),
        "prophet_full": _nan_safe(p_fc.to_dict(orient="records")),
    }


@app.get("/brand-forecast")
def brand_forecast(brand: Optional[str] = Query(default=None)):
    """
    Forecast monthly sales per brand using Prophet.
    If `brand` is provided, return only that brand's forecast.
    """
    _require_models()
    df: pd.DataFrame = _state["df"]

    if "brand" not in df.columns:
        raise HTTPException(status_code=404, detail="Dataset has no 'brand' column.")

    brands = [brand] if brand else df["brand"].unique().tolist()
    results = {}

    for b in brands[:15]:  # cap at 15 brands for performance
        b_df = df[df["brand"] == b].copy()
        b_monthly = aggregate_monthly(b_df)
        if len(b_monthly) < 12:
            continue
        try:
            p_result = train_prophet(b_monthly)
            fc = forecast_prophet(p_result["model"], periods=36)
            fc["ds"] = fc["ds"].dt.strftime("%Y-%m")
            results[b] = _nan_safe(fc.to_dict(orient="records"))
        except Exception:
            pass

    return {"brands": list(results.keys()), "forecasts": results}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _nan_safe(records: list) -> list:
    """Replace NaN/inf with None so JSON serialisation doesn't fail."""
    cleaned = []
    for rec in records:
        cleaned.append(
            {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
             for k, v in rec.items()}
        )
    return cleaned


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
