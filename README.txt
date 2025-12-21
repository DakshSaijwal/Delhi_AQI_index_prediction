🌫️ Delhi AQI Index Prediction

Spatio-Temporal Air Quality Forecasting using Machine Learning

📌 Project Overview

Air quality monitoring data suffers from severe missingness, spatial sparsity, and strong temporal dependencies.
This project builds an end-to-end machine learning pipeline to predict major air pollutants in Delhi by combining:

Hybrid data imputation (temporal + spatial)

Time-aware feature engineering

Robust model validation

City-wide spatial visualization (heatmaps)

Two tree-based models — XGBoost and LightGBM — are trained and compared across six pollutants.

🧠 Key Contributions

Hybrid Imputation Strategy

Short gaps: Linear interpolation

Medium gaps: Kalman smoothing

Long gaps: Spatial IDW using tuned power parameters

Spatio-Temporal Modeling

Per-station time series combined with geographic coordinates

Lag and rolling-window features

Leakage-Safe Evaluation

Time-based train–test split (Expanding-Window/Single-Shot Forecast)

No future information used in training

Per-Pollutant Modeling

Separate models for PM2.5, PM10, NOx, SO₂, CO, and O₃

City-Scale Visualization

7-day hourly heatmaps using spatial interpolation

🗂️ Dataset Description

Stations: 40 monitoring stations across Delhi

Time span: 2009 – 2023 (hourly data)

Pollutants:

PM2.5

PM10

NOx

SO₂

CO

O₃

Raw data structure:

DL.data → station-wise time series

DL.details → station coordinates

locs.pred → spatial prediction grid

⚙️ Project Pipeline

Raw AQI Data
        ↓        ↓
IDW p-value Cross-Validation
        ↓
Hybrid Imputation (Temporal + Spatial)
        ↓
Coverage-Based Trimming
        ↓
Feature Engineering
        ↓
Model Training (XGBoost / LightGBM)
        ↓
Evaluation & Visualization

🧩 Hybrid Imputation Strategy

Missing data is classified per pollutant per station:

Gap Length	Method Used
≤ 6 hours	Linear interpolation
≤ 72 hours	Kalman smoothing
> 72 hours	Spatial IDW
For long gaps, Inverse Distance Weighting (IDW) is applied using neighboring stations, with the power parameter p optimized separately for each pollutant.

🔍 IDW Power Optimization

For each pollutant, IDW power p ∈ [0.2, 2.0] was selected using cross-validated RMSE over recent historical data.

Pollutant	Best p
PM2.5    	0.20
PM10    	0.20
NOx		0.20
SO₂	     			0.20
CO		0.29
O₃      		0.46

🛠 Feature Engineering

Temporal Features

Hour of day

Day of week

Month

Season

Lag & Rolling Features

Lag 1h, 24h, 48h, 72h

Rolling means (24h, 72h)

Spatial Features

Latitude

Longitude

Station ID (categorical)

Final feature table:

~1.75 million rows

42 features

🤖 Model Training

Two models were trained per pollutant:

Models Used

XGBoost

LightGBM

Validation Strategy

Last 60 days used as test set

All training data strictly precedes test data

Prevents temporal leakage

📊 Model Performance
LightGBM Results (RMSE / MAE)
Pollutant	RMSE	MAE
PM2.5		22.07	13.15
PM10		39.84	25.34
NOx		22.66	11.81
SO₂					3.66	  1.87
CO		0.43	 0.20
O₃					8.73	  4.72

LightGBM consistently performed slightly better than XGBoost.

📈 Model Interpretation

Feature importance analysis shows:

Lagged pollutant values dominate predictions

Strong daily and seasonal patterns

Spatial coordinates help distinguish station behavior

Feature importance plots available in results/feature_importance/

🌍 Heatmap Visualization

For each pollutant:

Hourly predictions for 7 days

Interpolated to a spatial grid using IDW

Heatmaps available in results/heatmaps/

🧪 Time-Series Validation

For selected stations (e.g., Station 5 and Station 33):

Actual vs predicted pollutant concentrations plotted

Confirms temporal consistency and trend capture

Available in results/time_series/

🗃️ Repository Structure

Delhi_AQI_index_prediction/
│
├── data/
│   ├── raw/
│   ├── analysis/
│   ├── cleaned/
│   ├── interim/
│   └── processed/
│
├── src/
│   ├── 00_load_and_validate.py
│   ├── 01_gap_analysis.py
│   ├── 01_idw_p_cross_validation.py
│   ├── 02_imputation.py
│   ├── 02b_validate_imputation.py
│   ├── 02c_trim_low_coverage.py
│   ├── 03_feature_engineering.py
│   ├── 04a_train_xgboost.py
│   ├── 04b_train_lightgbm.py
│   ├── 04c_feature_importance.py
│   ├── 04d_Igb_feature_importance.py
│   ├── 04e_export_lightgbm_predictions.py
│   ├── 05_error_regime_analysis.py
│   ├── 05_forecasting.py
│   ├── 05_generate_7day_heatmaps.py
│   └── 06_accuracy_timeseries.py
│
├── models/
│   ├── xgboost/
│   └── lightgbm/
│
├── results/
│   ├── heatmaps/
│   └── accuracy_plots/
│
└── README.md
