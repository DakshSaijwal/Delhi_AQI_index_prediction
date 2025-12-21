# ============================================================
# 04a_train_xgboost.py
# Train XGBoost models for all 6 pollutants
# Time-based split (last 60 days)
# Defensive against object/categorical columns
# ============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_FILE = "data/processed/dl_data_features.csv"
OUT_DIR = "models/xgboost"
METRICS_FILE = "models/xgboost/metrics.csv"

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
TEST_DAYS = 60

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
    "random_state": 42,
    "tree_method": "hist"
}
# ---------------------------------------

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("Loading feature-engineered data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.sort_values("datetime")

# ---------------- Train / Test Split ----------------
cutoff_time = df["datetime"].max() - pd.Timedelta(days=TEST_DAYS)

train_df = df[df["datetime"] < cutoff_time]
test_df  = df[df["datetime"] >= cutoff_time]

print(f"Train period end : {train_df.datetime.max()}")
print(f"Test period start: {test_df.datetime.min()}")
print(f"Train rows: {len(train_df):,}")
print(f"Test  rows: {len(test_df):,}")

metrics = []

for pollutant in POLLUTANTS:
    print(f"\nTraining XGBoost for {pollutant.upper()}")

    # Drop rows where target is missing
    train_p = train_df.dropna(subset=[pollutant])
    test_p  = test_df.dropna(subset=[pollutant])

    y_train = train_p[pollutant]
    y_test  = test_p[pollutant]

    # ---------------- Feature selection ----------------
    X_train = train_p.drop(columns=POLLUTANTS + ["datetime", "station_id"], errors="ignore")
    X_test  = test_p.drop(columns=POLLUTANTS + ["datetime", "station_id"], errors="ignore")

    # DROP NON-NUMERIC COLUMNS (CRITICAL FIX)
    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test.select_dtypes(include=[np.number])

    model = xgb.XGBRegressor(**XGB_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")

    metrics.append({
        "pollutant": pollutant,
        "rmse": rmse,
        "mae": mae,
        "train_rows": len(train_p),
        "test_rows": len(test_p)
    })

    model_path = f"{OUT_DIR}/xgb_{pollutant.replace('.', '')}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model → {model_path}")

# ---------------- Save Metrics ----------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_FILE, index=False)

print("\nTraining complete.")
print("Metrics saved to:", METRICS_FILE)
print(metrics_df)
