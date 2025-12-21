# ============================================================
# 02_imputation.py
# Hybrid imputation for Delhi AQI data
# ============================================================

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pykalman import KalmanFilter
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_FILE = "data/interim/dl_data_trimmed.csv"
STATIONS_FILE = "data/raw/dl_details.csv"
P_FILE = "data/interim/idw_p_values.csv"
OUT_FILE = "data/interim/dl_data_imputed.csv"

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]

SHORT_GAP_HRS = 6
MEDIUM_GAP_HRS = 72
MIN_P_CLIP = 0.2
N_NEIGHBORS = 5
# ---------------------------------------

print("Loading data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
stations = pd.read_csv(STATIONS_FILE)
p_vals = pd.read_csv(P_FILE).set_index("pollutant")["best_p"].to_dict()

df = df.merge(
    stations[["station_id", "lon", "lat"]],
    on="station_id",
    how="left"
)

df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

# ---------------- Helpers ----------------

def find_nan_blocks(series):
    is_na = series.isna().values
    blocks = []
    i = 0
    while i < len(is_na):
        if is_na[i]:
            j = i
            while j < len(is_na) and is_na[j]:
                j += 1
            blocks.append((i, j, j - i))
            i = j
        else:
            i += 1
    return blocks


def kalman_fill(values):
    obs = values[~np.isnan(values)]
    if len(obs) == 0:
        return values

    kf = KalmanFilter(
        initial_state_mean=np.mean(obs),
        observation_covariance=1.0,
        transition_covariance=0.01
    )

    filled, _ = kf.smooth(values)
    return filled.flatten()


def idw_predict(target_row, others_df, p):
    coords_t = target_row[["lon", "lat"]].values.reshape(1, -1)
    coords_o = others_df[["lon", "lat"]].values
    vals = others_df["val"].values

    dists = cdist(coords_t, coords_o)[0]
    idx = np.argsort(dists)[:N_NEIGHBORS]

    d = np.maximum(dists[idx], 1e-3)
    w = 1.0 / (d ** p)

    return np.sum(w * vals[idx]) / np.sum(w)

# ---------------- Imputation ----------------

print("Starting hybrid imputation...")
out_frames = []

for pollutant in POLLUTANTS:
    print(f"\nImputing {pollutant.upper()}")

    p_used = max(p_vals.get(pollutant, 1.0), MIN_P_CLIP)

    df_p = df[["station_id", "datetime", "lon", "lat", pollutant]].copy()
    df_p = df_p.rename(columns={pollutant: "val"})

    for station_id, g in tqdm(df_p.groupby("station_id"), desc="Stations"):
        g = g.sort_values("datetime").reset_index(drop=True)

        first_valid = g["val"].first_valid_index()
        if first_valid is None:
            continue

        g.loc[:first_valid - 1, "val"] = np.nan

        nan_blocks = find_nan_blocks(g["val"])

        for start, end, length in nan_blocks:
            if start < first_valid:
                continue

            if length <= SHORT_GAP_HRS:
                g.loc[start:end - 1, "val"] = g["val"].interpolate().iloc[start:end]

            elif length <= MEDIUM_GAP_HRS:
                window_start = max(0, start - 10)
                window_end = min(len(g), end + 10)
                segment = g["val"].iloc[window_start:window_end].values
                filled = kalman_fill(segment)
                g.loc[start:end - 1, "val"] = filled[
                    (start - window_start):(end - window_start)
                ]

            else:
                for idx in range(start, end):
                    t = g.loc[idx, "datetime"]
                    snap = df_p[
                        (df_p["datetime"] == t) &
                        (df_p["station_id"] != station_id) &
                        (~df_p["val"].isna())
                    ]

                    if len(snap) < N_NEIGHBORS:
                        continue

                    try:
                        g.at[idx, "val"] = idw_predict(
                            g.loc[idx:idx],
                            snap,
                            p_used
                        )
                    except Exception:
                        pass

        g["pollutant"] = pollutant
        out_frames.append(g)

# ---------------- Save ----------------

final_df = pd.concat(out_frames)
final_df = final_df.rename(columns={"val": "value"})

final_df.to_csv(OUT_FILE, index=False)

print("\nImputation complete.")
print("Saved to:", OUT_FILE)
