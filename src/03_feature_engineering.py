# ============================================================
# 03_feature_engineering.py
# Create temporal, lag, and rolling features (per-pollutant)
# ============================================================

import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
INPUT_FILE = "data/processed/dl_data_final.csv"
OUTPUT_FILE = "data/processed/dl_data_features.csv"

POLLUTANTS = ["pm2.5", "pm10", "nox", "so2", "co", "o3"]

LAGS = [1, 24, 72]        # hours
ROLL_WINDOW = 24         # hours
# ---------------------------------------

print("Loading cleaned data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])

# ------------------------------------------------
# Pivot LONG → WIDE
# ------------------------------------------------
print("Pivoting to wide format...")
df_wide = (
    df.pivot_table(
        index=["station_id", "datetime"],
        columns="pollutant",
        values="value"
    )
    .reset_index()
)

df_wide.columns.name = None

# ------------------------------------------------
# Time-based features
# ------------------------------------------------
print("Creating time features...")
df_wide["hour"] = df_wide["datetime"].dt.hour
df_wide["day_of_week"] = df_wide["datetime"].dt.dayofweek
df_wide["month"] = df_wide["datetime"].dt.month

# Explicit season (India-specific)
def get_season(m):
    if m in [3, 4, 5]:
        return "summer"
    elif m in [6, 7, 8, 9]:
        return "monsoon"
    elif m in [10, 11]:
        return "post_monsoon"
    else:
        return "winter"

df_wide["season"] = df_wide["month"].apply(get_season)

# ------------------------------------------------
# Lag & rolling features (per pollutant)
# ------------------------------------------------
print("Creating lag and rolling features...")

df_wide = df_wide.sort_values(["station_id", "datetime"])

for p in POLLUTANTS:
    if p not in df_wide.columns:
        continue

    # Lags
    for lag in LAGS:
        df_wide[f"{p}_lag{lag}"] = (
            df_wide.groupby("station_id")[p].shift(lag)
        )

    # Rolling stats
    df_wide[f"{p}_roll{ROLL_WINDOW}_mean"] = (
        df_wide.groupby("station_id")[p]
        .rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df_wide[f"{p}_roll{ROLL_WINDOW}_std"] = (
        df_wide.groupby("station_id")[p]
        .rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW)
        .std()
        .reset_index(level=0, drop=True)
    )

# ------------------------------------------------
# Final cleanup
# ------------------------------------------------
print("Finalizing feature table...")

# Drop rows that cannot support lag/rolling features
min_lag = max(LAGS + [ROLL_WINDOW])
df_wide = df_wide.groupby("station_id").apply(
    lambda x: x.iloc[min_lag:]
).reset_index(drop=True)

print("Final feature table shape:", df_wide.shape)

# Save
df_wide.to_csv(OUTPUT_FILE, index=False)

print("\nFeature engineering complete.")
print("Saved to:", OUTPUT_FILE)
