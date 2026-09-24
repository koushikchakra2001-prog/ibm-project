"""
train.py
--------
Entry-point script for training both forecasting models.
Usage:
    python train.py --data path/to/bike_sales_india.csv
"""
import argparse
import sys
from pathlib import Path

# Make sure sibling imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_csv, clean, aggregate_monthly, prepare_features
from model import train_prophet, train_xgb


def main():
    parser = argparse.ArgumentParser(description="Train bike sales forecasting models.")
    parser.add_argument("--data", required=True, help="Path to bike_sales_india.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("  Bike Sales India – Model Training")
    print("=" * 60)

    # Load & clean
    print(f"\n[1/4] Loading data from: {args.data}")
    raw = load_csv(args.data)
    df = clean(raw)
    print(f"      Rows after cleaning: {len(df):,}")

    # Monthly aggregation
    print("\n[2/4] Aggregating to monthly series …")
    monthly = aggregate_monthly(df)
    print(f"      Monthly observations: {len(monthly)}")
    print(f"      Date range: {monthly['ds'].min().date()} → {monthly['ds'].max().date()}")
    print(f"      Total units sold: {monthly['y'].sum():,.0f}")

    # Train Prophet
    print("\n[3/4] Training Prophet model …")
    prophet_result = train_prophet(monthly)

    # Train XGBoost
    print("\n[4/4] Training XGBoost model …")
    feat_df = prepare_features(monthly)
    xgb_result = train_xgb(feat_df)

    print("\n" + "=" * 60)
    print("  Training complete! Models saved to backend/models/")
    print("  Prophet  R²:", round(prophet_result["metrics"]["r2"], 4))
    print("  XGBoost  R²:", round(xgb_result["metrics"]["r2"], 4))
    print("=" * 60)


if __name__ == "__main__":
    main()
