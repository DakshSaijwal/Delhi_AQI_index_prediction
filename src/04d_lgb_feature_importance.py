# ============================================================
# 04d_lgb_feature_importance.py
# Feature importance for LightGBM models
# ============================================================

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# ---------------- CONFIG ----------------
DATA_FILE = "data/processed/dl_data_features.csv"
MODEL_DIR = Path("models/lightgbm")
OUT_DIR = MODEL_DIR / "feature_importance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
# ---------------------------------------

print("Loading feature-engineered data...")
df = pd.read_csv(DATA_FILE)

# Drop non-feature columns
drop_cols = POLLUTANTS + ["datetime"]
X_cols = [c for c in df.columns if c not in drop_cols]

for pollutant in POLLUTANTS:
    print(f"\nProcessing {pollutant.upper()}")

    model_path = MODEL_DIR / f"lgb_{pollutant.replace('.', '')}.joblib"
    model = joblib.load(model_path)

    importance = model.feature_importance(importance_type="gain")
    features = model.feature_name()

    imp_df = (
        pd.DataFrame({
            "feature": features,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Save CSV
    csv_path = OUT_DIR / f"importance_{pollutant.replace('.', '')}.csv"
    imp_df.to_csv(csv_path, index=False)

    # Plot top 20
    top = imp_df.head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title(f"LightGBM Feature Importance — {pollutant.upper()}")
    plt.xlabel("Gain")
    plt.tight_layout()

    png_path = OUT_DIR / f"importance_{pollutant.replace('.', '')}.png"
    plt.savefig(png_path)
    plt.close()

    print(f"Saved → {png_path}")

print("\nLightGBM feature importance completed.")
