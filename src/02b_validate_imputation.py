import pandas as pd

df = pd.read_csv(
    "data/interim/dl_data_imputed.csv",
    parse_dates=["datetime"]
)

print("Rows:", len(df))
print("Stations:", df["station_id"].nunique())

missing_pct = df["value"].isna().mean() * 100
print(f"\nOverall missing % after imputation: {missing_pct:.2f}%")

print("\nTop 5 stations by missing %:")
print(
    df.groupby("station_id")["value"]
      .apply(lambda x: x.isna().mean() * 100)
      .sort_values(ascending=False)
      .head()
)
