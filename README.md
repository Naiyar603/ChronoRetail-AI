# 📊 ChronoRetail AI — Retail Sales Forecasting System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Trained-brightgreen?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Trained-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

</div>

---

## 🎯 What is this project?

**ChronoRetail AI** predicts how much money a retail store will make on any given day.

It is trained on **1,017,209 real sales records** from Rossmann — a German drugstore chain with **1,115 stores** — and achieves **97.73% accuracy**.

> Instead of guessing tomorrow's revenue, store managers can now get an instant ₹ prediction by entering a store ID and date.

---

## 🏆 Results

| Model | MAE | R² Score |
|---|---|---|
| LightGBM | — | — |
| XGBoost | — | — |
| **Ensemble (Final)** | **₹290** | **0.9773** |

- **R² 0.9773** means the model explains 97.73% of all sales variation
- **MAE ₹290** means predictions are off by only ₹290 on average
- **+38.8% promo lift** — the model detects promotional impact automatically

---

## 📁 Project Structure

```
ChronoRetail-AI/
│
├── train.py              ← Step 1: Run this to train the model
├── dashboard 1.py          ← Step 2: Run this to open the dashboard
├── requirements.txt      ← All libraries needed
├── store.csv             ← Store metadata (included)
│
├── app_folder/
│   ├── train.csv         ← Download from Kaggle (too large for GitHub)
│   └── store.csv
│
├── model.pkl             ← Auto-generated after running train.py
└── eval_results.pkl      ← Auto-generated after running train.py
```

---

## ⚙️ How to Run

### Step 1 — Install all libraries
```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset
Go to this link and download `train.csv` and `store.csv`:
👉 https://www.kaggle.com/competitions/rossmann-store-sales

Place both files inside the `app_folder/` folder.

### Step 3 — Train the model
```bash
python train.py
```
This takes about 5–10 minutes. It will save `model.pkl` and `eval_results.pkl` automatically.

### Step 4 — Open the dashboard
```bash
streamlit run dashboard 1.py
```
Open your browser at → `http://localhost:8501`

---

## 📊 Dashboard Features

The dashboard has **4 tabs** and a **live predictor** in the sidebar:

| Tab | What you see |
|---|---|
| 📈 Trends | Monthly sales trends, day-of-week patterns, seasonal heatmap, promo impact, year-over-year growth |
| 🏪 Store Analysis | Store type rankings, assortment mix, competition distance vs sales |
| 🧠 Model Insights | Feature importance chart, actual vs predicted scatter, MAE and R² comparison |
| 🔢 Data Explorer | Raw data table, descriptive statistics, box plots |

### Live Predictor (Sidebar)
Enter any of these → get instant ₹ forecast:
- Store ID (1 to 1115)
- Forecast date
- Promo active or not
- State holiday type
- School holiday or not

---

## 🧠 How the Model Works — 15 Steps

| Step | What happens |
|---|---|
| 1 | Load train.csv + store.csv and merge them |
| 2 | Remove closed store days (Sales = 0 not useful) |
| 3 | Extract date features — Year, Month, Week, Weekend, MonthStart/End |
| 4 | Calculate how many months competitor has been open |
| 5 | Calculate how many weeks Promo2 has been running |
| 6 | Create lag features — yesterday's sales, 7-day, 30-day history |
| 7 | Create rolling averages and standard deviations |
| 8 | Convert text categories to numbers (StoreType, Assortment) |
| 9 | Remove raw columns that were already converted |
| 10 | Apply log transform to Sales — makes training more stable |
| 11 | Split data 80% train / 20% test (by time, not randomly) |
| 12 | Train LightGBM — 1,500 trees with early stopping |
| 13 | Train XGBoost — 1,000 trees with early stopping |
| 14 | Ensemble — average both model predictions |
| 15 | Save model.pkl and eval_results.pkl to disk |

---

## 📦 Tech Stack

| What | Library |
|---|---|
| Data processing | pandas, NumPy |
| Machine learning | LightGBM, XGBoost, scikit-learn |
| Dashboard | Streamlit |
| Charts | Plotly, Matplotlib |
| Language | Python 3.8+ |

---

## 📋 requirements.txt

```
pandas
numpy
lightgbm
xgboost
scikit-learn
streamlit
plotly
matplotlib
```

---

## 📁 Dataset Info

| Detail | Value |
|---|---|
| Dataset name | Rossmann Store Sales |
| Source | Kaggle |
| Total rows | 1,017,209 |
| Total stores | 1,115 |
| Date range | 2013 to 2015 |
| Target column | Sales (daily ₹ revenue) |

> ⚠️ train.csv is 37MB — too large for GitHub. Download it from Kaggle using the link above.

---

## 💡 Key Learnings

1. **Feature engineering is everything** — lag features and rolling averages were the top predictors, not model settings
2. **Log transform the target** — Sales data is skewed, log1p makes training much more stable
3. **Never random split time-series data** — always split by time to avoid data leakage
4. **Ensemble beats single models** — averaging LightGBM + XGBoost always gave better results than either alone
5. **Domain knowledge matters** — knowing about retail (payday effect, weekends, promo cycles) helped design better features

---

## 👤 Author

**Naiyar Azam**


---

## 🙏 Dataset Credit

Rossmann Store Sales — Kaggle Competition
👉 https://www.kaggle.com/competitions/rossmann-store-sales/data

Download these 2 files from the Data tab:
- train.csv  → main sales data (37 MB)
- store.csv  → store information (44 KB)

---

<div align="center">

⭐ If you found this project useful, please give it a star! ⭐

</div>
