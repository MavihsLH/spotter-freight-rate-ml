import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

print("Loading data for model benchmarking...")
df = pd.read_csv("train-test.csv")

# 1. Cleaning
df["weight"] = df["weight"].abs()
# Median weight by equipment
median_weights = df.groupby("equipment")["weight"].median().to_dict()
df["weight"] = df["weight"].fillna(df["equipment"].map(median_weights))

# Market index imputation: forward fill / date mean
date_mi = df.groupby("date")["market_index"].mean().to_dict()
df["market_index"] = df["market_index"].fillna(df["date"].map(date_mi))
df["market_index"] = df["market_index"].fillna(df["market_index"].median())

# 2. Feature Engineering
def engineer_features(data):
    f = pd.DataFrame(index=data.index)
    
    # Dates & Calendar
    dates = pd.to_datetime(data["date"])
    f["month"] = dates.dt.month
    f["day"] = dates.dt.day
    f["dayofweek"] = dates.dt.dayofweek
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(int)
    f["dayofyear"] = dates.dt.dayofyear
    
    # Cyclical encodings
    f["sin_dow"] = np.sin(2 * np.pi * f["dayofweek"] / 7)
    f["cos_dow"] = np.cos(2 * np.pi * f["dayofweek"] / 7)
    f["sin_month"] = np.sin(2 * np.pi * f["month"] / 12)
    f["cos_month"] = np.cos(2 * np.pi * f["month"] / 12)
    
    # Geographic features
    f["pickup_lat"] = data["pickup_lat"]
    f["pickup_lon"] = data["pickup_lon"]
    f["delivery_lat"] = data["delivery_lat"]
    f["delivery_lon"] = data["delivery_lon"]
    f["lat_diff"] = data["delivery_lat"] - data["pickup_lat"]
    f["lon_diff"] = data["delivery_lon"] - data["pickup_lon"]
    
    # Haversine distance
    lat1, lon1 = np.radians(data["pickup_lat"]), np.radians(data["pickup_lon"])
    lat2, lon2 = np.radians(data["delivery_lat"]), np.radians(data["delivery_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    f["haversine_dist"] = 3958.8 * c  # miles
    
    f["distance"] = data["distance"]
    f["dist_ratio"] = f["distance"] / (f["haversine_dist"] + 1e-5)
    f["weight"] = data["weight"]
    f["weight_per_mile"] = f["weight"] / (f["distance"] + 1e-5)
    
    # Categorical equipment encoding
    equipment_dummies = pd.get_dummies(data["equipment"], prefix="eq", dtype=float)
    for col in ["eq_Dry Van", "eq_Flatbed", "eq_Reefer"]:
        if col in equipment_dummies.columns:
            f[col] = equipment_dummies[col]
        else:
            f[col] = 0.0
            
    # Signals
    if "market_index" in data.columns:
        f["market_index"] = data["market_index"]
    if "quote_signal" in data.columns:
        f["quote_signal"] = data["quote_signal"]
        f["base_estimate"] = data["distance"] * data["quote_signal"]
        
    return f

X_all = engineer_features(df)
y_all = df["posted_rate"].values
dates_all = pd.to_datetime(df["date"])

# 3. Temporal Out-Of-Time Split:
# Train: Jan 1 to Aug 31 (8 months)
# Val: Sep 1 to Oct 31 (2 months)
train_mask = dates_all <= pd.to_datetime("2025-08-31")
val_mask = dates_all >= pd.to_datetime("2025-09-01")

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]

print(f"OOT Train samples: {len(X_train):,} | OOT Val samples: {len(X_val):,}")

# Evaluation metric helper
def eval_metrics(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"[{name:28s}] MAE: ${mae:6.2f} | RMSE: ${rmse:6.2f} | WAPE: {wape:5.2f}% | R²: {r2:6.4f}")
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "R2": r2}

print("\n--- BENCHMARK RESULTS ON OUT-OF-TIME VAL (Sep-Oct 2025) ---")

# Model 0: Naive Baseline (Distance * Quote Signal)
naive_val_pred = X_val["base_estimate"].values
eval_metrics(y_val, naive_val_pred, "0. Naive (Dist * Signal)")

# Model 1: Ridge Regression (Direct)
ridge = Ridge(alpha=100.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)
eval_metrics(y_val, ridge_pred, "1. Ridge Regression")

# Model 2: LightGBM Direct
lgb_direct = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
lgb_direct.fit(X_train, y_train)
lgb_direct_pred = lgb_direct.predict(X_val)
eval_metrics(y_val, lgb_direct_pred, "2. LightGBM (Direct Rate)")

# Model 3: LightGBM Multiplier Formulation (Residual Modeling)
# Target = posted_rate / (distance * quote_signal)
multiplier_train = y_train / X_train["base_estimate"].values
# Filter extreme anomalies for clean multiplier training
valid_m = (multiplier_train > 0.4) & (multiplier_train < 3.0)

lgb_mult = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
lgb_mult.fit(X_train[valid_m], multiplier_train[valid_m])
mult_val_pred = lgb_mult.predict(X_val)
lgb_mult_final_pred = X_val["base_estimate"].values * mult_val_pred
eval_metrics(y_val, lgb_mult_final_pred, "3. LightGBM (Multiplier M)")

# Model 4: XGBoost Multiplier Formulation
xgb_mult = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb_mult.fit(X_train[valid_m], multiplier_train[valid_m])
xgb_mult_val_pred = xgb_mult.predict(X_val)
xgb_mult_final_pred = X_val["base_estimate"].values * xgb_mult_val_pred
eval_metrics(y_val, xgb_mult_final_pred, "4. XGBoost (Multiplier M)")

# Model 5: Ensemble (Average of LightGBM Multiplier & XGBoost Multiplier)
ensemble_pred = 0.5 * lgb_mult_final_pred + 0.5 * xgb_mult_final_pred
eval_metrics(y_val, ensemble_pred, "5. Ensemble (LGBM + XGB)")
