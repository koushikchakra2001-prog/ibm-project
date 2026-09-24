"""
model.py
--------
Trains two complementary forecasting models:
  1. Prophet   – captures trend + seasonality automatically
  2. XGBoost   – gradient-boosted tree model on engineered lag features

Both models are persisted with joblib so the API can load them without retraining.
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

PROPHET_PATH = MODEL_DIR / "prophet_model.pkl"
XGB_PATH = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ---------------------------------------------------------------------------
# Prophet model
# ---------------------------------------------------------------------------
def train_prophet(monthly_df: pd.DataFrame) -> dict:
    """
    Train a Facebook Prophet model on a monthly time-series DataFrame
    with columns ['ds', 'y'].
    Returns the fitted model and in-sample metrics.
    """
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
    )
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    model.fit(monthly_df)

    # In-sample predictions for metrics
    forecast = model.predict(monthly_df[["ds"]])
    y_pred = forecast["yhat"].values
    y_true = monthly_df["y"].values

    metrics = _calc_metrics(y_true, y_pred)

    joblib.dump(model, PROPHET_PATH)
    print(f"[Prophet] Saved → {PROPHET_PATH}")
    print(f"[Prophet] MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  R²={metrics['r2']:.4f}")
    return {"model": model, "metrics": metrics}


def forecast_prophet(model, periods: int = 36) -> pd.DataFrame:
    """Generate future forecast for `periods` months ahead."""
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ---------------------------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------------------------
def train_xgb(feature_df: pd.DataFrame) -> dict:
    """
    Train an XGBoost regressor on lag/rolling features.
    `feature_df` must contain columns produced by data_loader.prepare_features().
    """
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler

    feature_cols = ["month_num", "year_num", "trend", "lag_1", "lag_3", "lag_12",
                    "rolling_3", "rolling_6"]
    X = feature_df[feature_cols]
    y = feature_df["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
    )

    # Time-series cross-validation for honest metric estimates
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for train_idx, val_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y.iloc[train_idx])
        preds = model.predict(X_scaled[val_idx])
        cv_maes.append(mean_absolute_error(y.iloc[val_idx], preds))

    # Final fit on all data
    model.fit(X_scaled, y)

    metrics = {
        "cv_mae": float(np.mean(cv_maes)),
        **_calc_metrics(y.values, model.predict(X_scaled)),
    }

    joblib.dump(model, XGB_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[XGBoost] Saved → {XGB_PATH}")
    print(f"[XGBoost] CV-MAE={metrics['cv_mae']:.1f}  RMSE={metrics['rmse']:.1f}  R²={metrics['r2']:.4f}")
    return {"model": model, "scaler": scaler, "metrics": metrics}


def forecast_xgb(model, scaler, last_known: pd.DataFrame, horizon_months: int = 36) -> pd.DataFrame:
    """
    Iteratively forecast `horizon_months` months into the future using
    recursive one-step-ahead predictions (lag features updated each step).
    """
    feature_cols = ["month_num", "year_num", "trend", "lag_1", "lag_3", "lag_12",
                    "rolling_3", "rolling_6"]

    history = last_known["y"].tolist()
    last_date = last_known["ds"].max()
    last_trend = len(last_known) - 1

    results = []
    for i in range(horizon_months):
        next_date = last_date + pd.DateOffset(months=i + 1)
        t = last_trend + i + 1
        lag1 = history[-1]
        lag3 = history[-3] if len(history) >= 3 else history[0]
        lag12 = history[-12] if len(history) >= 12 else history[0]
        roll3 = np.mean(history[-3:]) if len(history) >= 3 else history[-1]
        roll6 = np.mean(history[-6:]) if len(history) >= 6 else history[-1]

        row = [[next_date.month, next_date.year, t, lag1, lag3, lag12, roll3, roll6]]
        X_scaled = scaler.transform(row)
        pred = float(model.predict(X_scaled)[0])
        pred = max(0, pred)

        results.append({"ds": next_date, "yhat": pred})
        history.append(pred)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)}


def load_models():
    """Load persisted models from disk. Returns (prophet, xgb, scaler) or None if not trained."""
    if not PROPHET_PATH.exists() or not XGB_PATH.exists():
        return None, None, None
    prophet_model = joblib.load(PROPHET_PATH)
    xgb_model = joblib.load(XGB_PATH)
    scaler = joblib.load(SCALER_PATH)
    return prophet_model, xgb_model, scaler
