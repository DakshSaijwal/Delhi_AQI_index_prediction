# ============================================================
# 02c_trim_low_coverage.py
# Remove station-years with insufficient data coverage
# ============================================================

import pandas as pd

# ---------------- CONFIG ----------------
INPUT_FILE = "data/interim/dl_data_imputed.csv"
OUTPUT_FILE = "data/processed/dl_data_final.csv"

MIN_YEARLY_COVERAGE = 0.80   # 80%
# ---------------------------------------

print("Loading imputed data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])

# Extract year
df["year"] = df["datetime"].dt.year

print("Computing yearly coverage per station...")

# Coverage per station-year
coverage = (
    df.groupby(["station_id", "year"])["value"]
      .apply(lambda x: x.notna().mean())
      .reset_index(name="coverage")
)

# Identify valid station-years
valid_station_years = coverage[
    coverage["coverage"] >= MIN_YEARLY_COVERAGE
][["station_id", "year"]]

print(
    f"Keeping {len(valid_station_years)} station-years "
    f"with ≥ {int(MIN_YEARLY_COVERAGE*100)}% coverage"
)

# Inner join to keep only valid station-years
df_final = df.merge(
    valid_station_years,
    on=["station_id", "year"],
    how="inner"
)

# Drop helper column
df_final = df_final.drop(columns=["year"])

# Final stats
print("\nFinal dataset stats:")
print("Rows:", len(df_final))
print("Stations:", df_final["station_id"].nunique())

missing_pct = df_final["value"].isna().mean() * 100
print(f"Remaining missing %: {missing_pct:.2f}%")

# Save
df_final.to_csv(OUTPUT_FILE, index=False)

print("\nTrimmed dataset saved to:", OUTPUT_FILE)
