# ============================================================
# 04e_export_lightgbm_predictions.py
# Export LightGBM predictions (SAFE categorical handling)
# ============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- CONFIG ----------------
DATA_FILE = "data/processed/dl_data_features.csv"
MODEL_DIR = Path("models/lightgbm")
OUT_FILE = "data/processed/lightgbm_predictions.csv"

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
TEST_DAYS = 60
# ---------------------------------------

print("Loading feature-engineered data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.sort_values("datetime")

Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# Drop season completely (critical fix)
if "season" in df.columns:
    df = df.drop(columns=["season"])

# Train-test split
cutoff = df["datetime"].max() - pd.Timedelta(days=TEST_DAYS)
test_df = df[df["datetime"] > cutoff]

all_preds = []
metrics = []

for pollutant in POLLUTANTS:
    print(f"\nPredicting {pollutant.upper()}")

    model_path = MODEL_DIR / f"lgb_{pollutant.replace('.', '')}.joblib"
    if not model_path.exists():
        print(f"  Model not found → skipped")
        continue

    model = joblib.load(model_path)

    te = test_df.dropna(subset=[pollutant]).copy()
    if len(te) == 0:
        print("  No test data → skipped")
        continue

    y_true = te[pollutant].values

    X = te.drop(
        columns=POLLUTANTS + ["datetime"],
        errors="ignore"
    )

    preds = model.predict(X)

    rmse = mean_squared_error(y_true, preds) ** 0.5
    mae = mean_absolute_error(y_true, preds)

    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")

    out = te[["datetime", "station_id"]].copy()
    out["pollutant"] = pollutant
    out["actual"] = y_true
    out["predicted"] = preds

    all_preds.append(out)

    metrics.append({
        "pollutant": pollutant,
        "rmse": rmse,
        "mae": mae,
        "rows": len(te)
    })

# ---------------- Save ----------------
if not all_preds:
    raise RuntimeError("No predictions generated")

final_preds = pd.concat(all_preds, ignore_index=True)
final_preds.to_csv(OUT_FILE, index=False)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(MODEL_DIR / "prediction_metrics.csv", index=False)

print("\nPrediction export complete.")
print("Saved predictions →", OUT_FILE)
print("Saved metrics →", MODEL_DIR / "prediction_metrics.csv")
print(metrics_df)
