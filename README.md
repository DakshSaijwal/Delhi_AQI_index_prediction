# 🌫️ Delhi AQI Index Prediction  
### 🧠 Spatio-Temporal Air Quality Forecasting using Machine Learning

> 📍 A research-grade, leakage-safe spatio-temporal forecasting pipeline for predicting major air pollutants across Delhi using tree-based machine learning models.

---

## 🔍 Why This Project?

Air quality data is hard — not because of models, but because of **data reality**:

- ❌ Severe missingness
- 📍 Sparse spatial coverage
- ⏱️ Strong temporal dependencies

This project tackles **all three simultaneously**, end-to-end, with methodological rigor.

---

## 🧩 Key Capabilities

- ✅ Hybrid **temporal + spatial imputation**
- ✅ Leakage-safe **time-aware validation**
- ✅ Per-pollutant forecasting (6 pollutants)
- ✅ City-wide **7-day hourly AQI heatmaps**
- ✅ XGBoost vs LightGBM comparison
- ✅ Interpretability via feature importance & time-series plots

---

## 🗂️ Dataset Snapshot

📍 **Location**: Delhi, India  
🏭 **Stations**: 40 monitoring stations  
⏳ **Time Span**: 2009 – 2023  
⏱️ **Resolution**: Hourly  

### 🌬️ Pollutants Modeled
- PM2.5
- PM10
- NOx
- SO₂
- CO
- O₃

---

## 🧠 Core Modules Overview

| Module | Description |
|------|------------|
| 🧩 Data Ingestion | Validates raw AQI data and station metadata |
| 🛠️ Gap Analysis | Classifies missing segments by duration |
| 🧪 IDW Tuning | Cross-validates IDW power per pollutant |
| 🔄 Imputation Engine | Hybrid temporal + spatial gap filling |
| 🧮 Feature Engineering | Time, lag, rolling, and spatial features |
| 🤖 Model Training | XGBoost & LightGBM per pollutant |
| 📊 Evaluation | Leakage-safe temporal validation |
| 🌍 Visualization | City-wide heatmaps & station time-series |

---

## 🔧 End-to-End Pipeline

```mermaid
flowchart TD
    A["Raw AQI Data"]
    B["IDW Power Cross-Validation"]
    C["Hybrid Imputation – Temporal + Spatial"]
    D["Coverage-Based Trimming"]
    E["Feature Engineering"]
    F["Model Training – XGBoost / LightGBM"]
    G["Evaluation & Visualization"]

    A --> B --> C --> D --> E --> F --> G
