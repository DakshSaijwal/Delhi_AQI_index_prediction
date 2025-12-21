# ============================================================
# 05_generate_7day_heatmaps.py
# Generate IDW heatmaps for next 7 days (hourly)
# ============================================================

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
PRED_FILE = "data/processed/lightgbm_predictions.csv"
STATION_FILE = "data/raw/dl_details.csv"
GRID_FILE = "data/raw/locs_pred.csv"
P_FILE = "data/interim/idw_p_values.csv"

OUT_DIR = Path("outputs/heatmaps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
N_NEIGHBORS = 5
# ---------------------------------------

print("Loading data...")

preds = pd.read_csv(PRED_FILE, parse_dates=["datetime"])
stations = pd.read_csv(STATION_FILE)
grid = pd.read_csv(GRID_FILE)
p_vals = pd.read_csv(P_FILE).set_index("pollutant")["best_p"].to_dict()

# ---- Validate schema ----
required_cols = {"datetime", "station_id", "pollutant", "predicted"}
if not required_cols.issubset(preds.columns):
    raise ValueError(
        f"Predictions file missing required columns.\n"
        f"Required: {required_cols}\n"
        f"Found: {set(preds.columns)}"
    )

# ---- Merge coordinates ----
preds = preds.merge(
    stations[["station_id", "lon", "lat"]],
    on="station_id",
    how="left"
)

# ---- Grid ----
grid = grid.rename(columns={"x": "lon", "y": "lat"})
xy_target = grid[["lon", "lat"]].values


# ---------------- IDW ----------------
def idw_interpolate(xy_known, values, xy_target, p):
    dists = cdist(xy_target, xy_known)
    dists = np.maximum(dists, 1e-6)
    weights = 1.0 / (dists ** p)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values


# ---------------- Heatmaps ----------------
print("\nGenerating heatmaps...\n")

for pollutant in POLLUTANTS:
    print(f"Pollutant: {pollutant.upper()}")

    p = max(p_vals.get(pollutant, 1.0), 0.2)
    pol_dir = OUT_DIR / pollutant.replace(".", "")
    pol_dir.mkdir(exist_ok=True)

    df_p = preds[preds["pollutant"] == pollutant]
    start_time = df_p["datetime"].min()
    end_time = start_time + pd.Timedelta(days=7)

    times = sorted(
        df_p[
            (df_p["datetime"] >= start_time) &
            (df_p["datetime"] < end_time)
        ]["datetime"].unique()
)

    for t in tqdm(times, desc=pollutant):
        snap = df_p[df_p["datetime"] == t]

        if len(snap) < N_NEIGHBORS:
            continue

        vals = snap["predicted"].values
        if np.all(np.isnan(vals)):
            continue

        xy_known = snap[["lon", "lat"]].values

        z = idw_interpolate(xy_known, vals, xy_target, p)

        # ---- Plot ----
        plt.figure(figsize=(8, 6))
        plt.tricontourf(
            grid["lon"],
            grid["lat"],
            z,
            levels=15,
            cmap="viridis"
        )
        plt.scatter(xy_known[:, 0], xy_known[:, 1], c="black", s=5)
        plt.colorbar(label=pollutant.upper())
        plt.title(f"{pollutant.upper()} forecast\n{pd.to_datetime(t)}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        fname = pol_dir / f"{pd.to_datetime(t).strftime('%Y%m%d_%H')}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

print("\nAll heatmaps generated successfully.")
print(f"Saved under: {OUT_DIR.resolve()}")
