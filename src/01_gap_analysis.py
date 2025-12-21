# ============================================================
# 01_gap_analysis.py
# Purpose:
#   - Remove structural missing data (pre-station start)
#   - Detect consecutive missing-value gaps
#   - Classify gaps into short / medium / long
#   - NO imputation performed here
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Paths ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

DATA_INTERIM.mkdir(parents=True, exist_ok=True)

DL_DATA_PATH = DATA_RAW / "dl_data.csv"
OUT_GAP_SUMMARY = DATA_INTERIM / "gap_summary.csv"
OUT_TRIMMED_DATA = DATA_INTERIM / "dl_data_trimmed.csv"

# --- Parameters ---------------------------------------------
POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]

SHORT_GAP_HOURS = 6
MEDIUM_GAP_HOURS = 72

# --- Load data ----------------------------------------------
print("Loading data...")
df = pd.read_csv(DL_DATA_PATH, parse_dates=["datetime"])

print("Rows before trimming:", len(df))

# --- Helper: detect station start ---------------------------
def find_station_start(station_df):
    """
    Station start = first timestamp where ANY pollutant is observed.
    """
    mask = station_df[POLLUTANTS].notna().any(axis=1)
    if not mask.any():
        return None
    return station_df.loc[mask, "datetime"].min()

# --- Trim pre-station structural missing data ---------------
print("Detecting station start times and trimming data...")

trimmed_frames = []
station_start_times = {}

for station_id, sdf in tqdm(df.groupby("station_id"), desc="Stations"):
    sdf = sdf.sort_values("datetime")

    start_time = find_station_start(sdf)
    station_start_times[station_id] = start_time

    if start_time is None:
        continue  # drop fully empty stations (unlikely but safe)

    sdf = sdf[sdf["datetime"] >= start_time]
    trimmed_frames.append(sdf)

df_trimmed = pd.concat(trimmed_frames, ignore_index=True)

print("Rows after trimming:", len(df_trimmed))

# Save trimmed data (still unfilled)
df_trimmed.to_csv(OUT_TRIMMED_DATA, index=False)
print("Saved trimmed data to:", OUT_TRIMMED_DATA)

# --- Gap detection ------------------------------------------
print("Detecting and classifying gaps...")

gap_records = []

for station_id, sdf in tqdm(df_trimmed.groupby("station_id"), desc="Gap analysis"):
    sdf = sdf.sort_values("datetime").reset_index(drop=True)

    for pollutant in POLLUTANTS:
        is_na = sdf[pollutant].isna()

        # Run-length encoding on missingness
        grp = (is_na != is_na.shift()).cumsum()

        for _, block in sdf[is_na].groupby(grp):
            start_time = block["datetime"].iloc[0]
            end_time = block["datetime"].iloc[-1]
            gap_len = len(block)

            if gap_len <= SHORT_GAP_HOURS:
                gap_type = "short"
            elif gap_len <= MEDIUM_GAP_HOURS:
                gap_type = "medium"
            else:
                gap_type = "long"

            gap_records.append({
                "station_id": station_id,
                "pollutant": pollutant,
                "start_time": start_time,
                "end_time": end_time,
                "gap_hours": gap_len,
                "gap_type": gap_type
            })

gap_df = pd.DataFrame(gap_records)

# --- Save gap summary ---------------------------------------
gap_df.to_csv(OUT_GAP_SUMMARY, index=False)

print("Saved gap summary to:", OUT_GAP_SUMMARY)

# --- Report -------------------------------------------------
print("\nGap type distribution:")
print(gap_df["gap_type"].value_counts())

print("\nGap analysis complete.")
print("No imputation performed.")
