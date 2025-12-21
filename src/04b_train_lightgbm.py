# ============================================================
# 04b_train_lightgbm.py
# Train LightGBM models (NUMERIC FEATURES ONLY)
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import joblib

# ---------------- CONFIG ----------------
DATA_FILE = "data/processed/dl_data_features.csv"
OUT_DIR = Path("models/lightgbm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
TEST_DAYS = 60
# ---------------------------------------

print("Loading feature-engineered data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.sort_values("datetime")

# ------------------------------------------------
# HARD RULE: NO CATEGORICAL FEATURES IN LIGHTGBM
# ------------------------------------------------
DROP_COLS = ["season"]  # explicitly remove
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Force everything except datetime to numeric
for c in df.columns:
    if c != "datetime":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------------------------------------------
# Train / Test split (last 60 days)
# ------------------------------------------------
cutoff = df["datetime"].max() - pd.Timedelta(days=TEST_DAYS)

train_df = df[df["datetime"] <= cutoff]
test_df  = df[df["datetime"] > cutoff]

print("Train period end :", train_df["datetime"].max())
print("Test period start:", test_df["datetime"].min())
print(f"Train rows: {len(train_df):,}")
print(f"Test  rows: {len(test_df):,}")

metrics = []

# ------------------------------------------------
# Train one model per pollutant
# ------------------------------------------------
for pollutant in POLLUTANTS:
    print(f"\nTraining LightGBM for {pollutant.upper()}")

    tr = train_df.dropna(subset=[pollutant])
    te = test_df.dropna(subset=[pollutant])

    y_train = tr[pollutant]
    y_test  = te[pollutant]

    X_train = tr.drop(columns=POLLUTANTS + ["datetime"])
    X_test  = te.drop(columns=POLLUTANTS + ["datetime"])

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test  = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_test],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae  = mean_absolute_error(y_test, preds)

    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")

    joblib.dump(model, OUT_DIR / f"lgb_{pollutant.replace('.', '')}.joblib")

    metrics.append({
        "pollutant": pollutant,
        "rmse": rmse,
        "mae": mae,
        "train_rows": len(tr),
        "test_rows": len(te)
    })

# ------------------------------------------------
# Save metrics
# ------------------------------------------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)

print("\nTraining complete.")
print("Metrics saved to:", OUT_DIR / "metrics.csv")
print(metrics_df)
