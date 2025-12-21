# <font size="7">🌫️ Delhi AQI Index Prediction</font>
### <font size="5">Spatio-Temporal Air Quality Forecasting using Machine Learning</font>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-orange)](https://github.com/microsoft/LightGBM)

---

## 📌 <font size="6">**Project Overview**</font>
**<font size="4">Air quality monitoring in Delhi is a complex challenge. Data often suffers from severe missingness, spatial sparsity, and heavy temporal dependencies.</font>**

This project builds an end-to-end pipeline to predict major air pollutants by combining **advanced data recovery (imputation)** with **high-performance gradient boosting models**.

---

## ⚙️ <font size="6">**The Pipeline**</font>



1.  **<font size="4">Hybrid Imputation:</font>** Fixing gaps using time and neighbor-based logic.
2.  **<font size="4">Feature Engineering:</font>** Creating "memory" for the model using lags.
3.  **<font size="4">Model Training:</font>** Comparing **XGBoost** and **LightGBM**.
4.  **<font size="4">Spatial Visualization:</font>** Generating 7-day city-wide heatmaps.

---

## 🧩 <font size="6">**Hybrid Imputation Strategy**</font>
**<font size="4">We don't use the same fix for every hole in the data. We choose the best method based on how long the sensor was offline:</font>**

| Gap Length | Method | Simple Example |
| :--- | :--- | :--- |
| **Short (≤ 6h)** | **Linear Interpolation** | If it's 10:00 (AQI 50) and 12:00 (AQI 60), we assume 11:00 was 55. |
| **Medium (≤ 72h)** | **Kalman Smoothing** | Follows the "wave" or trend of the day to fill the curve. |
| **Long (> 72h)** | **Spatial IDW** | Uses data from the **nearest 3 working stations** to fill the gap. |

---

## 🤖 <font size="6">**Model Performance**</font>
**<font size="4">LightGBM consistently outperformed XGBoost in both speed and accuracy.</font>**

### **LightGBM Performance Metrics**
| Pollutant | RMSE (Error) | MAE (Avg Error) |
| :--- | :--- | :--- |
| **PM2.5** | **22.07** | **13.15** |
| **PM10** | **39.84** | **25.34** |
| **NOx** | **22.66** | **11.81** |
| **CO** | **0.43** | **0.20** |

---

## 🗂️ <font size="6">**Repository Structure**</font>
```bash
Delhi_AQI_index_prediction/
├── data/              # Raw and processed datasets
├── src/               # Python scripts (00_load to 06_viz)
├── models/            # Saved ML model files
├── results/           # Generated Heatmaps and Accuracy plots
└── README.md          # Project documentation
