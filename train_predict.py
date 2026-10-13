import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)

print_header("1. LOADING & PREPROCESSING DATASETS")

train_path = Path("train-test.csv")
val_path = Path("validation.csv")
dec_path = Path("december-chart-inputs.csv")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
dec_df = pd.read_csv(dec_path)

print(f"Loaded Train-Test : {len(train_df):,} rows")
print(f"Loaded Validation : {len(val_df):,} rows")
print(f"Loaded December   : {len(dec_df):,} rows")


train_df["weight"] = train_df["weight"].abs()
val_df["weight"] = val_df["weight"].abs()
dec_df["weight"] = dec_df["weight"].abs()

median_weights = train_df.groupby("equipment")["weight"].median().to_dict()
train_df["weight"] = train_df["weight"].fillna(train_df["equipment"].map(median_weights))
val_df["weight"] = val_df["weight"].fillna(val_df["equipment"].map(median_weights))
dec_df["weight"] = dec_df["weight"].fillna(dec_df["equipment"].map(median_weights))

all_dates = pd.concat([train_df[["date", "market_index"]], val_df[["date", "market_index"]]])
daily_market = all_dates.groupby("date")["market_index"].mean().to_dict()

train_df["market_index"] = train_df["market_index"].fillna(train_df["date"].map(daily_market))
val_df["market_index"] = val_df["market_index"].fillna(val_df["date"].map(daily_market))
global_median_mi = train_df["market_index"].median()
train_df["market_index"] = train_df["market_index"].fillna(global_median_mi)
val_df["market_index"] = val_df["market_index"].fillna(global_median_mi)

# doing some feature engineering
print_header("2. FEATURE ENGINEERING & ENRICHMENT")

def extract_features(df_input: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df_input.index)
    
    dates = pd.to_datetime(df_input["date"])
    f["month"] = dates.dt.month
    f["day"] = dates.dt.day
    f["dayofweek"] = dates.dt.dayofweek
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(float)
    f["dayofyear"] = dates.dt.dayofyear
    
    # cyclic calendar encodings
    f["sin_dow"] = np.sin(2 * np.pi * f["dayofweek"] / 7)
    f["cos_dow"] = np.cos(2 * np.pi * f["dayofweek"] / 7)
    f["sin_month"] = np.sin(2 * np.pi * f["month"] / 12)
    f["cos_month"] = np.cos(2 * np.pi * f["month"] / 12)
    
    # 2. Geospatial & Lane Encodings
    f["pickup_lat"] = df_input["pickup_lat"]
    f["pickup_lon"] = df_input["pickup_lon"]
    f["delivery_lat"] = df_input["delivery_lat"]
    f["delivery_lon"] = df_input["delivery_lon"]
    f["lat_diff"] = df_input["delivery_lat"] - df_input["pickup_lat"]
    f["lon_diff"] = df_input["delivery_lon"] - df_input["pickup_lon"]
    
    # Great-circle Haversine Distance (miles)
    lat1, lon1 = np.radians(df_input["pickup_lat"]), np.radians(df_input["pickup_lon"])
    lat2, lon2 = np.radians(df_input["delivery_lat"]), np.radians(df_input["delivery_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    f["haversine_dist"] = 3958.8 * c
    
    f["distance"] = df_input["distance"]
    f["dist_ratio"] = f["distance"] / (f["haversine_dist"] + 1e-4)
    
    # 3. Payload & Equipment Encodings
    f["weight"] = df_input["weight"]
    f["weight_per_mile"] = f["weight"] / (f["distance"] + 1e-4)
    
    # One-hot equipment encoding
    for eq in ["Dry Van", "Flatbed", "Reefer"]:
        col_name = f"eq_{eq.replace(' ', '_')}"
        f[col_name] = (df_input["equipment"] == eq).astype(float)
        # Interaction between payload weight and equipment
        f[f"{col_name}_x_weight"] = f[col_name] * f["weight"]
        
    # 4. Market & Spot Signals
    f["market_index"] = df_input["market_index"]
    f["quote_signal"] = df_input["quote_signal"]
    f["base_estimate"] = f["distance"] * f["quote_signal"]
    f["signal_x_market"] = f["quote_signal"] * f["market_index"]
    
    return f

# Build feature matrix for train
X_full_train = extract_features(train_df)
y_full_train = train_df["posted_rate"].values

# -------------------------------------------------------------------------
# OUT-OF-TIME (OOT) VALIDATION BENCHMARK
# -------------------------------------------------------------------------
print_header("3. OUT-OF-TIME (OOT) TEMPORAL VALIDATION")

train_dates = pd.to_datetime(train_df["date"])
oot_train_idx = train_dates <= pd.to_datetime("2025-08-31")
oot_val_idx = train_dates >= pd.to_datetime("2025-09-01")

X_oot_tr = X_full_train[oot_train_idx]
y_oot_tr = y_full_train[oot_train_idx]
X_oot_val = X_full_train[oot_val_idx]
y_oot_val = y_full_train[oot_val_idx]

# Target Reformulation: Multiplier M = posted_rate / base_estimate
m_oot_tr = y_oot_tr / X_oot_tr["base_estimate"].values
valid_m_idx = (m_oot_tr >= 0.4) & (m_oot_tr <= 3.0)

print(f"OOT Training Set   (Jan–Aug 2025): {len(X_oot_tr):,} loads")
print(f"OOT Holdout Set    (Sep–Oct 2025): {len(X_oot_val):,} loads")

# Train LightGBM Multiplier Model
lgb_model = lgb.LGBMRegressor(
    n_estimators=350,
    learning_rate=0.04,
    num_leaves=35,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_oot_tr[valid_m_idx], m_oot_tr[valid_m_idx])
lgb_oot_pred_m = lgb_model.predict(X_oot_val)
lgb_oot_pred_rate = X_oot_val["base_estimate"].values * lgb_oot_pred_m

# Train XGBoost Multiplier Model
xgb_model = xgb.XGBRegressor(
    n_estimators=350,
    learning_rate=0.04,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
xgb_model.fit(X_oot_tr[valid_m_idx], m_oot_tr[valid_m_idx])
xgb_oot_pred_m = xgb_model.predict(X_oot_val)
xgb_oot_pred_rate = X_oot_val["base_estimate"].values * xgb_oot_pred_m

# Blended Ensemble
ensemble_oot_pred_rate = 0.60 * lgb_oot_pred_rate + 0.40 * xgb_oot_pred_rate

# Compute Metrics
def report_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<30s} | MAE: ${mae:6.2f} | RMSE: ${rmse:6.2f} | WAPE: {wape:5.2f}% | R²: {r2:6.4f}")

naive_base = X_oot_val["base_estimate"].values
report_metrics("0. Naive Baseline (Dist*Signal)", y_oot_val, naive_base)
report_metrics("1. LightGBM Multiplier Model", y_oot_val, lgb_oot_pred_rate)
report_metrics("2. XGBoost Multiplier Model", y_oot_val, xgb_oot_pred_rate)
report_metrics("3. Final Blended Ensemble", y_oot_val, ensemble_oot_pred_rate)

# -------------------------------------------------------------------------
# FINAL FULL RETRAINING
# -------------------------------------------------------------------------
print_header("4. FULL MODEL RETRAINING (ALL 48,000 LOADS)")

m_full_tr = y_full_train / X_full_train["base_estimate"].values
valid_full_idx = (m_full_tr >= 0.4) & (m_full_tr <= 3.0)

final_lgb = lgb.LGBMRegressor(
    n_estimators=450,
    learning_rate=0.035,
    num_leaves=35,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    verbose=-1
)
final_lgb.fit(X_full_train[valid_full_idx], m_full_tr[valid_full_idx])

final_xgb = xgb.XGBRegressor(
    n_estimators=450,
    learning_rate=0.035,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
final_xgb.fit(X_full_train[valid_full_idx], m_full_tr[valid_full_idx])

print("Trained final LightGBM and XGBoost models on full labeled historical data.")

# -------------------------------------------------------------------------
# GENERATING PREDICTIONS FOR VALIDATION.CSV (12,000 LOADS)
# -------------------------------------------------------------------------
print_header("5. PREDICTING VALIDATION.CSV (12,000 LOADS)")

X_val = extract_features(val_df)
val_pred_m_lgb = final_lgb.predict(X_val)
val_pred_m_xgb = final_xgb.predict(X_val)
val_pred_m = 0.60 * val_pred_m_lgb + 0.40 * val_pred_m_xgb

val_predicted_rate = np.round(X_val["base_estimate"].values * val_pred_m, 2)
# Ensure strictly positive
val_predicted_rate = np.clip(val_predicted_rate, 50.0, None)

# Save validation_predictions.csv
val_pred_df = pd.DataFrame({
    "load_id": val_df["load_id"],
    "predicted_rate": val_predicted_rate
})
val_pred_df.to_csv("validation_predictions.csv", index=False)
print(f"Saved 'validation_predictions.csv' with {len(val_pred_df):,} rows.")
print("Sample validation predictions:")
print(val_pred_df.head(5))

# -------------------------------------------------------------------------
# GENERATING PREDICTIONS FOR DECEMBER BENCHMARK (31 DAYS)
# -------------------------------------------------------------------------
print_header("6. PREDICTING DECEMBER CHART BENCHMARK (31 DAYS)")

# Complete coordinates for Lexington and Fort Wayne
coords_map = {
    "Lexington": {"lat": 36.99152, "lon": -84.99876},
    "Fort Wayne": {"lat": 41.31561, "lon": -85.36206}
}
dec_df["pickup_lat"] = dec_df["pickup"].map(lambda p: coords_map[p]["lat"])
dec_df["pickup_lon"] = dec_df["pickup"].map(lambda p: coords_map[p]["lon"])
dec_df["delivery_lat"] = dec_df["delivery"].map(lambda d: coords_map[d]["lat"])
dec_df["delivery_lon"] = dec_df["delivery"].map(lambda d: coords_map[d]["lon"])

# Extract daily market index and quote signal from validation calendar for December
val_dec_loads = val_df[pd.to_datetime(val_df["date"]).dt.month == 12]
daily_market_dec = val_dec_loads.groupby("date")["market_index"].mean().to_dict()
daily_quote_dec = val_dec_loads.groupby("date")["quote_signal"].mean().to_dict()

dec_df["market_index"] = dec_df["date"].map(daily_market_dec)
dec_df["quote_signal"] = dec_df["date"].map(daily_quote_dec)

X_dec = extract_features(dec_df)
dec_pred_m_lgb = final_lgb.predict(X_dec)
dec_pred_m_xgb = final_xgb.predict(X_dec)
dec_pred_m = 0.60 * dec_pred_m_lgb + 0.40 * dec_pred_m_xgb

dec_predicted_rate = np.round(X_dec["base_estimate"].values * dec_pred_m, 2)
dec_predicted_rate = np.clip(dec_predicted_rate, 50.0, None)

# Update december-chart-inputs.csv with the predicted rates
# Note: Format required by score.py: exactly 7 columns:
# pickup,delivery,distance,equipment,weight,date,predicted_rate
dec_output_df = dec_df[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
dec_output_df["predicted_rate"] = dec_predicted_rate

dec_output_df.to_csv("december-chart-inputs.csv", index=False)
print(f"Updated 'december-chart-inputs.csv' with {len(dec_output_df):,} daily rates.")

# Create 'data/' directory mirror for seamless compatibility with score.py default commands in readme
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
dec_output_df.to_csv(data_dir / "december_chart_inputs.csv", index=False)
dec_output_df.to_csv(data_dir / "december-chart-inputs.csv", index=False)
shutil.copy("train-test.csv", data_dir / "train_test.csv")
shutil.copy("train-test.csv", data_dir / "train-test.csv")
shutil.copy("validation.csv", data_dir / "validation.csv")
shutil.copy("validation-predictions-template.csv", data_dir / "validation_predictions_template.csv")

print("Created data/ mirror files for full backward compatibility with score.py instructions.")

# -------------------------------------------------------------------------
# RUNNING OFFICIAL SCORE.PY VERIFICATION
# -------------------------------------------------------------------------
print_header("7. EXECUTING OFFICIAL SCORE.PY VALIDATION")

score_cmd = [
    "python", "score.py",
    "--predictions", "validation_predictions.csv",
    "--december-predictions", "december-chart-inputs.csv"
]
result = subprocess.run(score_cmd, capture_output=True, text=True)
print("Score.py output:")
print(result.stdout)
if result.stderr:
    print("Score.py stderr:", result.stderr)

if result.returncode == 0:
    print("\nSUCCESS: All predictions verified and passed by score.py!")
    print("Generated chart: scorer_results/candidate_december.png")
else:
    print(f"\nFAILURE: score.py exited with code {result.returncode}")
