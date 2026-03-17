"""
╔══════════════════════════════════════════════════════════════════╗
║                  BACKEND — train.py                              ║
║   Run this FIRST to train the model                              ║
║   Command:  python train.py                                      ║
║   Output:   model.pkl  +  eval_results.pkl                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, pickle, warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost  as xgb

# ───────────────────────────────────────────────
# PATHS
# ───────────────────────────────────────────────
TRAIN_CSV  = "app_folder/train.csv.csv"
STORE_CSV  = "app_folder/store.csv"
MODEL_PATH = "model.pkl"
EVAL_PATH  = "eval_results.pkl"

# auto-detect CSV location
for t in ["app_folder/train.csv.csv", "app_folder/train.csv", "train.csv.csv", "train.csv"]:
    if os.path.exists(t): TRAIN_CSV = t; break
for s in ["app_folder/store.csv", "store.csv"]:
    if os.path.exists(s): STORE_CSV = s; break

# ───────────────────────────────────────────────
# 1. LOAD & MERGE
# ───────────────────────────────────────────────
print("=" * 55)
print("   CHRONIC RETAIL ANALYTICS — Model Training")
print("=" * 55)
print(f"\n📂 Loading data...")
print(f"   Train CSV : {TRAIN_CSV}")
print(f"   Store CSV : {STORE_CSV}")

train = pd.read_csv(TRAIN_CSV, dtype={"StateHoliday": str})
store = pd.read_csv(STORE_CSV)
df    = train.merge(store, on="Store", how="left")
print(f"   Merged shape: {df.shape}")

# ───────────────────────────────────────────────
# 2. BASIC CLEANING
# ───────────────────────────────────────────────
print("\n🔧 Cleaning data...")
df["Open"] = df["Open"].fillna(1)
df = df[df["Open"] == 1].copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
print(f"   After filtering closed stores: {df.shape}")

# ───────────────────────────────────────────────
# 3. DATE FEATURES
# ───────────────────────────────────────────────
print("\n📅 Engineering date features...")
df["Year"]         = df["Date"].dt.year
df["Month"]        = df["Date"].dt.month
df["Day"]          = df["Date"].dt.day
df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
df["DayOfYear"]    = df["Date"].dt.dayofyear
df["IsWeekend"]    = (df["DayOfWeek"] >= 6).astype(int)
df["Quarter"]      = df["Date"].dt.quarter
df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
df["IsMonthEnd"]   = df["Date"].dt.is_month_end.astype(int)

# ───────────────────────────────────────────────
# 4. COMPETITION FEATURES
# ───────────────────────────────────────────────
print("🏪 Engineering competition features...")
df["CompetitionOpenSinceYear"]  = df["CompetitionOpenSinceYear"].fillna(0)
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
df["CompetitionMonthsOpen"] = (
    (df["Year"]  - df["CompetitionOpenSinceYear"])  * 12
    + (df["Month"] - df["CompetitionOpenSinceMonth"])
).clip(lower=0)

# ───────────────────────────────────────────────
# 5. PROMO FEATURES
# ───────────────────────────────────────────────
print("🔖 Engineering promo features...")
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
df["PromoOpenWeeks"]  = (
    (df["Year"]      - df["Promo2SinceYear"]) * 52
    + (df["WeekOfYear"] - df["Promo2SinceWeek"])
).clip(lower=0)

# ───────────────────────────────────────────────
# 6. LAG & ROLLING FEATURES
# ───────────────────────────────────────────────
print("📊 Engineering lag & rolling features...")
for lag in [1, 2, 3, 7, 14, 30]:
    df[f"Sales_Lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

for w in [7, 30]:
    base = df.groupby("Store")["Sales"].shift(1)
    df[f"Sales_RollMean_{w}"] = (
        base.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df[f"Sales_RollStd_{w}"] = (
        base.rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
    )
    df[f"Customers_RollMean_{w}"] = (
        df.groupby("Store")["Customers"].shift(1)
          .rolling(w, min_periods=1).mean()
          .reset_index(level=0, drop=True)
    )

df["Sales_SameWeekday_1w"] = df.groupby(["Store", "DayOfWeek"])["Sales"].shift(1)
df["Sales_SameWeekday_4w"] = df.groupby(["Store", "DayOfWeek"])["Sales"].shift(4)

ss = (df.groupby("Store")["Sales"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "StoreMeanSales", "median": "StoreMedianSales"})
        .reset_index())
df = df.merge(ss, on="Store", how="left")
df = df.dropna(subset=[c for c in df.columns if "Lag" in c or "Roll" in c]).copy()
print(f"   After lag features: {df.shape}")

# ───────────────────────────────────────────────
# 7. FILL MISSING & ENCODE
# ───────────────────────────────────────────────
print("\n🔢 Encoding categorical features...")
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
df["PromoInterval"]       = df["PromoInterval"].fillna("None")
df["StateHoliday"]        = df["StateHoliday"].astype(str).map(
    {"0": 0, "a": 1, "b": 2, "c": 3}).fillna(0)
df = pd.get_dummies(df, columns=["StoreType", "Assortment", "PromoInterval"], drop_first=True)

# ── Fix special characters in column names (LightGBM requirement) ──
df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', str(c)) for c in df.columns]

# ───────────────────────────────────────────────
# 8. DROP UNUSED COLUMNS
# ───────────────────────────────────────────────
drop_cols = ["Date", "Open", "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
             "Promo2SinceYear", "Promo2SinceWeek"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ───────────────────────────────────────────────
# 9. TARGET + FEATURES
# ───────────────────────────────────────────────
df["Sales_log"] = np.log1p(df["Sales"])
X     = df.drop(columns=["Sales", "Sales_log"])
y     = df["Sales_log"]
y_raw = df["Sales"]

split = int(len(df) * 0.8)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]
y_te_raw   = y_raw.iloc[split:]

print(f"\n📐 Train size : {X_tr.shape}")
print(f"   Test size  : {X_te.shape}")
print(f"   Features   : {X_tr.shape[1]}")

# ───────────────────────────────────────────────
# 10. TRAIN LIGHTGBM
# ───────────────────────────────────────────────
print("\n🚀 Training LightGBM...")
lgb_mdl = lgb.LGBMRegressor(
    objective="regression_l1", metric="mae", n_estimators=1500,
    learning_rate=0.05, num_leaves=127, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=42, verbose=-1,
)
lgb_mdl.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
)
lgb_pred = np.expm1(lgb_mdl.predict(X_te))
lgb_mae  = mean_absolute_error(y_te_raw, lgb_pred)
lgb_r2   = r2_score(y_te_raw, lgb_pred)
print(f"   ✅ LightGBM  →  MAE: {lgb_mae:,.2f}   R²: {lgb_r2:.4f}")

# ───────────────────────────────────────────────
# 11. TRAIN XGBOOST
# ───────────────────────────────────────────────
print("\n🚀 Training XGBoost...")
xgb_mdl = xgb.XGBRegressor(
    objective="reg:absoluteerror", eval_metric="mae", n_estimators=1000,
    learning_rate=0.05, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
    early_stopping_rounds=50,
)
xgb_mdl.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
xgb_pred = np.expm1(xgb_mdl.predict(X_te))
xgb_mae  = mean_absolute_error(y_te_raw, xgb_pred)
xgb_r2   = r2_score(y_te_raw, xgb_pred)
print(f"   ✅ XGBoost   →  MAE: {xgb_mae:,.2f}   R²: {xgb_r2:.4f}")

# ───────────────────────────────────────────────
# 12. ENSEMBLE
# ───────────────────────────────────────────────
ens_pred = (lgb_pred + xgb_pred) / 2
ens_mae  = mean_absolute_error(y_te_raw, ens_pred)
ens_r2   = r2_score(y_te_raw, ens_pred)
print(f"\n   ✅ Ensemble  →  MAE: {ens_mae:,.2f}   R²: {ens_r2:.4f}")

# ───────────────────────────────────────────────
# 13. SUMMARY
# ───────────────────────────────────────────────
print("\n" + "=" * 45)
print(f"{'Model':<15} {'MAE':>12} {'R²':>10}")
print("-" * 45)
print(f"{'LightGBM':<15} {lgb_mae:>12,.2f} {lgb_r2:>10.4f}")
print(f"{'XGBoost':<15} {xgb_mae:>12,.2f} {xgb_r2:>10.4f}")
print(f"{'Ensemble':<15} {ens_mae:>12,.2f} {ens_r2:>10.4f}")
print("=" * 45)

# ───────────────────────────────────────────────
# 14. FEATURE IMPORTANCE
# ───────────────────────────────────────────────
importance = pd.Series(
    lgb_mdl.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n📊 Top 15 Features (LightGBM):")
for feat, val in importance.head(15).items():
    print(f"   {feat:<35} {val}")

# ───────────────────────────────────────────────
# 15. SAVE MODEL + EVAL RESULTS
# ───────────────────────────────────────────────
print("\n💾 Saving model and results...")

rng = np.random.RandomState(99)
idx = rng.choice(len(y_te_raw), min(2000, len(y_te_raw)), replace=False)

bundle = dict(
    lgb=lgb_mdl,
    xgb=xgb_mdl,
    feature_names=list(X.columns),
)
results = dict(
    lgb_mae=lgb_mae, lgb_r2=lgb_r2,
    xgb_mae=xgb_mae, xgb_r2=xgb_r2,
    ens_mae=ens_mae, ens_r2=ens_r2,
    importance=importance,
    actual=y_te_raw.values[idx],
    ens_pred=ens_pred[idx],
)

with open(MODEL_PATH, "wb") as f: pickle.dump(bundle, f)
with open(EVAL_PATH,  "wb") as f: pickle.dump(results, f)

print(f"   ✅ model.pkl        saved")
print(f"   ✅ eval_results.pkl saved")
print("\n🎉 Training complete! Now run:  streamlit run dashboard.py")
print("=" * 55)
