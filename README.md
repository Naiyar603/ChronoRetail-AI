📊 ChronoRetail AI — Retail Sales Forecasting

End-to-end time-series forecasting system for predicting daily retail sales using a LightGBM + XGBoost ensemble.

🚀 Overview

Forecasts daily store revenue

Built on Rossmann sales dataset (1M+ rows, 1,115 stores)

Uses 40+ engineered features

Provides real-time predictions via Streamlit dashboard

✨ Key Features

Ensemble model: LightGBM + XGBoost

R² Score: 0.9773

MAE: ₹290

Promo impact detection: +38.8% lift

Interactive dashboard with analytics + live prediction

📁 Dataset

Source: Kaggle — Rossmann Store Sales

Time range: 2013–2015

🧠 Pipeline Summary

Data merge + cleaning

Feature engineering (lags, rolling stats, promos)

Time-based split

Train LightGBM + XGBoost

Ensemble predictions

Evaluation (MAE, R²)

📊 Dashboard

Trends & seasonality

Store-level insights

Model performance

Raw data explorer

Live sales predictor


🏗️ Structure
ChronoRetail-AI/
├── app_folder/
├── train.py
├── dashboard.py
├── model.pkl
├── eval_results.pkl
├── requirements.txt
└── README.md


🧰 Tech Stack

Python, pandas, NumPy

LightGBM, XGBoost, scikit-learn

Streamlit, Plotly, Matplotlib

🧠 Key Takeaways

Feature engineering drives performance

Time-aware validation is essential

Ensembles outperform single models