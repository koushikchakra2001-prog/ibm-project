"""
data_loader.py
--------------
Loads and preprocesses the bike_sales_india.csv dataset.
Handles multiple common column name variants found in Kaggle India bike sales datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Column aliases — maps common variant names → canonical internal names
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    # date
    "date": "date", "year": "year", "month": "month",
    "sale_date": "date", "sales_date": "date",
    # brand
    "brand": "brand", "company": "brand", "manufacturer": "brand",
    # model
    "model": "model", "bike_model": "model", "vehicle_model": "model",
    # category / segment
    "category": "category", "segment": "category", "type": "category",
    "bike_type": "category",
    # state / region
    "state": "state", "city": "state", "region": "state",
    "location": "state",
    # units sold
    "units_sold": "units_sold", "sales": "units_sold",
    "quantity": "units_sold", "quantity_sold": "units_sold",
    "number_of_bikes_sold": "units_sold",
    # price
    "price": "price", "ex_showroom_price": "price",
    "selling_price": "price", "price_inr": "price",
}


def load_csv(filepath: str) -> pd.DataFrame:
    """Load raw CSV and normalise column names."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns={c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}, inplace=True)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Create a proper datetime column from whatever date representation exists."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
    elif "year" in df.columns and "month" in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
    elif "year" in df.columns:
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
        df["month"] = 1
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls in critical columns, coerce numerics, remove obvious outliers."""
    df = parse_dates(df)

    if "units_sold" in df.columns:
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
        # Remove zero / negative sales
        df = df[df["units_sold"] > 0]

    if "price" in df.columns:
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(",", "").str.replace("₹", ""), errors="coerce"
        )

    df.dropna(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total units sold per month across all brands/models."""
    grp = df.groupby(pd.Grouper(key="date", freq="MS"))["units_sold"].sum().reset_index()
    grp.columns = ["ds", "y"]
    return grp


def aggregate_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    if "brand" not in df.columns:
        return pd.DataFrame()
    return df.groupby("brand")["units_sold"].sum().sort_values(ascending=False).reset_index()


def aggregate_by_state(df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in df.columns:
        return pd.DataFrame()
    return df.groupby("state")["units_sold"].sum().sort_values(ascending=False).reset_index()


def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns:
        return pd.DataFrame()
    return df.groupby("category")["units_sold"].sum().sort_values(ascending=False).reset_index()


def get_yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("year")["units_sold"].sum().reset_index()


def prepare_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag & rolling features for ML models."""
    df = monthly_df.copy()
    df["month_num"] = df["ds"].dt.month
    df["year_num"] = df["ds"].dt.year
    df["trend"] = np.arange(len(df))
    df["lag_1"] = df["y"].shift(1)
    df["lag_3"] = df["y"].shift(3)
    df["lag_12"] = df["y"].shift(12)
    df["rolling_3"] = df["y"].shift(1).rolling(3).mean()
    df["rolling_6"] = df["y"].shift(1).rolling(6).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
