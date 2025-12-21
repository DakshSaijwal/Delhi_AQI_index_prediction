# ============================================================
# 04c_feature_importance.py
# Extract and visualize feature importance from XGBoost models
# ============================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_DIR = Path("models/xgboost")
OUT_DIR = Path("models/xgboost/feature_importance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
TOP_K = 20
# ---------------------------------------

print("Extracting feature importance...")

all_importance = []

for pollutant in POLLUTANTS:
    model_path = MODEL_DIR / f"xgb_{pollutant.replace('.', '')}.joblib"

    if not model_path.exists():
        print(f"Model not found for {pollutant}, skipping.")
        continue

    model = joblib.load(model_path)

    # XGBoost feature importance (gain-based)
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")

    imp_df = (
        pd.DataFrame(score.items(), columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    imp_df["pollutant"] = pollutant
    all_importance.append(imp_df)

    # Save CSV
    csv_path = OUT_DIR / f"importance_{pollutant}.csv"
    imp_df.to_csv(csv_path, index=False)

    # Plot top K features
    top_df = imp_df.head(TOP_K)

    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"][::-1], top_df["gain"][::-1])
    plt.title(f"Top {TOP_K} Features — {pollutant.upper()}")
    plt.xlabel("Gain")
    plt.tight_layout()

    plot_path = OUT_DIR / f"importance_{pollutant}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved feature importance for {pollutant}")

# Combined table
final_df = pd.concat(all_importance, ignore_index=True)
final_df.to_csv(OUT_DIR / "feature_importance_all_pollutants.csv", index=False)

print("\nFeature importance extraction complete.")
print("Outputs saved in:", OUT_DIR)
