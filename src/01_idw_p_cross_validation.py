# ============================================================
# 01_idw_p_cross_validation.py
# Cross-validation to select optimal IDW power parameter (p)
# for each pollutant
# ============================================================

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_FILE = "data/raw/dl_data.csv"
STATIONS_FILE = "data/raw/dl_details.csv"
OUT_FILE = "data/interim/idw_p_values.csv"

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]
P_VALUES = np.round(np.arange(0.2, 2.01, 0.01), 2)

N_NEIGHBORS = 5
HIST_DAYS = 30
MAX_TIMESTAMPS = 200   # subsampling for speed
RANDOM_SEED = 42
# ----------------------------------------

np.random.seed(RANDOM_SEED)

print("Loading data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
stations = pd.read_csv(STATIONS_FILE)

# Restrict to recent history
end_time = df["datetime"].max()
start_time = end_time - pd.Timedelta(days=HIST_DAYS)
df = df[df["datetime"] >= start_time]

# Merge station coordinates
df = df.merge(
    stations[["station_id", "lon", "lat"]],
    on="station_id",
    how="inner"
)

# ---------------- IDW helper ----------------
def idw_predict(target, others, p):
    coords_t = target[["lon", "lat"]].values.reshape(1, -1)
    coords_o = others[["lon", "lat"]].values
    vals = others["val"].values

    dists = cdist(coords_t, coords_o)[0]
    idx = np.argsort(dists)[:N_NEIGHBORS]

    d = np.maximum(dists[idx], 1e-3)  # avoid division by zero
    w = 1.0 / (d ** p)

    return np.sum(w * vals[idx]) / np.sum(w)

# ---------------- Cross-validation ----------------
results = []

for pollutant in POLLUTANTS:
    print(f"\nOptimizing p for {pollutant.upper()}")

    data_p = (
        df[["station_id", "datetime", "lon", "lat", pollutant]]
        .dropna()
        .rename(columns={pollutant: "val"})
    )

    unique_times = np.sort(data_p["datetime"].unique())

    # Subsample timestamps for computational safety
    if len(unique_times) > MAX_TIMESTAMPS:
        sampled_times = np.random.choice(
            unique_times, size=MAX_TIMESTAMPS, replace=False
        )
    else:
        sampled_times = unique_times

    rmses = []

    for p in tqdm(P_VALUES, desc=f"p grid ({pollutant})"):
        errors = []

        for t in sampled_times:
            snap = data_p[data_p["datetime"] == t]
            if len(snap) < N_NEIGHBORS + 1:
                continue

            for i in range(len(snap)):
                target = snap.iloc[i:i+1]
                others = snap.drop(snap.index[i])

                try:
                    pred = idw_predict(target, others, p)
                    errors.append(pred - target["val"].values[0])
                except Exception:
                    continue

        rmse = np.sqrt(np.mean(np.square(errors))) if errors else np.inf
        rmses.append(rmse)

    best_idx = int(np.argmin(rmses))
    best_p = float(P_VALUES[best_idx])

    results.append({
        "pollutant": pollutant,
        "best_p": best_p,
        "rmse": rmses[best_idx]
    })

# ---------------- Save results ----------------
out = pd.DataFrame(results)
out.to_csv(OUT_FILE, index=False)

print("\nOptimized IDW p-values saved to:", OUT_FILE)
print(out)
