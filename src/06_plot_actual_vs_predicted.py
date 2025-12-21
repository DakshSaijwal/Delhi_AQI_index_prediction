# ============================================================
# 06_plot_actual_vs_predicted_all_stations.py
# Generate Actual vs Predicted plots
# For all 40 stations × all 6 pollutants
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_FILE = "data/processed/lightgbm_predictions.csv"
OUT_DIR = Path("outputs/actual_vs_predicted")

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
# ---------------------------------------

print("Loading predictions...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])

required = {"datetime", "station_id", "pollutant", "actual", "predicted"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing required columns. Found: {df.columns}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Plotting ----------------
print("\nGenerating plots...")

for pollutant in POLLUTANTS:
    print(f"\nPollutant: {pollutant.upper()}")

    pol_dir = OUT_DIR / pollutant.replace(".", "")
    pol_dir.mkdir(exist_ok=True)

    df_p = df[df["pollutant"] == pollutant]

    for station_id in tqdm(sorted(df_p["station_id"].unique()), desc=pollutant):
        df_s = df_p[df_p["station_id"] == station_id].sort_values("datetime")

        if df_s.empty:
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(df_s["datetime"], df_s["actual"], label="Actual", linewidth=1)
        plt.plot(df_s["datetime"], df_s["predicted"], label="Predicted", linewidth=1)

        plt.title(f"{pollutant.upper()} — Station {station_id}")
        plt.xlabel("Datetime")
        plt.ylabel(pollutant.upper())
        plt.legend()
        plt.grid(alpha=0.3)

        fname = pol_dir / f"station_{station_id}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

print("\nAll plots generated successfully.")
print(f"Saved under: {OUT_DIR.resolve()}")
