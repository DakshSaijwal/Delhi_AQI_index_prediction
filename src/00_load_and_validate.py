# ============================================================
# 00_load_and_validate.py
# Purpose:
#   - Load raw Delhi AQI CSV data
#   - Validate structure
#   - Report missingness
# ============================================================

import pandas as pd
from pathlib import Path

# --- Paths ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

DL_DATA_PATH = DATA_RAW / "dl_data.csv"
DL_DETAILS_PATH = DATA_RAW / "dl_details.csv"
LOCS_PRED_PATH = DATA_RAW / "locs_pred.csv"

# --- Load data ----------------------------------------------
print("Loading raw data...")

dl_data = pd.read_csv(DL_DATA_PATH, parse_dates=["datetime"])
dl_details = pd.read_csv(DL_DETAILS_PATH)
locs_pred = pd.read_csv(LOCS_PRED_PATH)

print("Loaded:")
print(f"  dl_data     : {dl_data.shape}")
print(f"  dl_details  : {dl_details.shape}")
print(f"  locs_pred   : {locs_pred.shape}")

# --- Basic validation ---------------------------------------
print("\nValidating data...")

# Station count
n_stations = dl_data["station_id"].nunique()
print(f"Unique stations in dl_data: {n_stations}")

assert n_stations == dl_details.shape[0], \
    "Mismatch between stations in dl_data and dl_details"

# Datetime checks
print("Datetime range:")
print("  Start:", dl_data["datetime"].min())
print("  End  :", dl_data["datetime"].max())

# Expected columns
expected_pollutants = {"pm2.5", "pm10", "nox", "so2", "co", "o3"}
actual_pollutants = set(dl_data.columns) & expected_pollutants

print("Pollutant columns found:", actual_pollutants)
assert len(actual_pollutants) >= 5, "Too few pollutant columns found"

# --- Missing data report ------------------------------------
print("\nMissing data summary (percent):")

missing_pct = (
    dl_data[sorted(actual_pollutants)]
    .isna()
    .mean()
    .mul(100)
    .round(2)
)

print(missing_pct)

# --- Per-station missingness --------------------------------
print("\nTop 5 stations by total missing values:")

station_missing = (
    dl_data
    .groupby("station_id")[sorted(actual_pollutants)]
    .apply(lambda x: x.isna().sum().sum())
    .sort_values(ascending=False)
    .head(5)
)

print(station_missing)

print("\nValidation complete. No data modified.")
