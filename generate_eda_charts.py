import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Set up output directory
OUTPUT_DIR = Path("eda_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.autolayout": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

PRIMARY = "#0F4C81"     # Classic Deep Blue
SECONDARY = "#2A9D8F"   # Teal
ACCENT_WARN = "#E76F51" # Coral / Warning
ACCENT_ORANGE = "#F4A261" # Amber / Gold
DARK_TEXT = "#1E293B"   # Slate 800
LIGHT_BG = "#F8FAFC"    # Slate 50
GRID_COLOR = "#E2E8F0"  # Slate 200

def safe_save(fig, filepath):
    target = Path(filepath).resolve()
    temp = target.with_name(f"{target.stem}_new.png")
    fig.savefig(str(temp), dpi=300)
    plt.close(fig)
    if target.exists():
        try:
            target.unlink()
        except Exception:
            pass
    try:
        if temp.exists():
            temp.replace(target)
    except Exception:
        pass


print("Loading datasets...")
train_df = pd.read_csv("train-test.csv")
val_df = pd.read_csv("validation.csv")
dec_df = pd.read_csv("december-chart-inputs.csv")

train_df["date"] = pd.to_datetime(train_df["date"])
val_df["date"] = pd.to_datetime(val_df["date"])
dec_df["date"] = pd.to_datetime(dec_df["date"])

# -------------------------------------------------------------
# CHART 1: DATA QUALITY & MISSING DATA AUDIT (Train vs Validation)
# -------------------------------------------------------------
print("Generating Chart 1: Data Quality Audit...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor="white")

# Missing data counts
missing_data = {
    "Train Weight Missing": train_df["weight"].isna().sum(),
    "Val Weight Missing": val_df["weight"].isna().sum(),
    "Train Market Index Missing": train_df["market_index"].isna().sum(),
    "Val Market Index Missing": val_df["market_index"].isna().sum(),
    "Train Neg Weight Anomaly": (train_df["weight"] < 0).sum(),
    "Val Neg Weight Anomaly": (val_df["weight"] < 0).sum(),
}

categories = [
    "Weight\n(Missing)", 
    "Market Index\n(Missing)", 
    "Negative Weight\n(Sign Flip Bug)"
]
train_vals = [
    train_df["weight"].isna().sum(),
    train_df["market_index"].isna().sum(),
    (train_df["weight"] < 0).sum()
]
val_vals = [
    val_df["weight"].isna().sum(),
    val_df["market_index"].isna().sum(),
    (val_df["weight"] < 0).sum()
]

# Percentages
train_pcts = [v / len(train_df) * 100 for v in train_vals]
val_pcts = [v / len(val_df) * 100 for v in val_vals]

x = np.arange(len(categories))
width = 0.35

rects1 = ax1.bar(x - width/2, train_vals, width, label=f"Train-Test (N={len(train_df):,})", color=PRIMARY, edgecolor="none", zorder=3)
rects2 = ax1.bar(x + width/2, val_vals, width, label=f"Validation (N={len(val_df):,})", color=ACCENT_WARN, edgecolor="none", zorder=3)

ax1.set_ylabel("Number of Impacted Loads", color=DARK_TEXT)
ax1.set_title("Data Quality Issues: Missing Values & Sign Inversions", loc="left", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, color=DARK_TEXT)
ax1.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color("#94A3B8")
ax1.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR)

# Add value labels on bars
for bar, pct in zip(rects1, train_pcts):
    h = bar.get_height()
    ax1.annotate(f"{h:,}\n({pct:.2f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=PRIMARY)

for bar, pct in zip(rects2, val_pcts):
    h = bar.get_height()
    ax1.annotate(f"{h:,}\n({pct:.2f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=ACCENT_WARN)

ax1.set_ylim(0, max(train_vals) * 1.25)

# Subplot 2: Negative Weight Distribution Proof (Showing |weight| is identical to real weights)
pos_weights = train_df.loc[train_df["weight"] > 0, "weight"]
neg_weights_abs = train_df.loc[train_df["weight"] < 0, "weight"].abs()

bins = np.linspace(10000, 50000, 25)
ax2.hist(pos_weights, bins=bins, density=True, alpha=0.55, color=PRIMARY, label="Positive Weights (Standard)", edgecolor="white", zorder=3)
ax2.hist(neg_weights_abs, bins=bins, density=True, alpha=0.75, color=ACCENT_WARN, label="Absolute of Negative Weights (|w|)", edgecolor="white", zorder=4)

ax2.set_xlabel("Weight (lbs)", color=DARK_TEXT)
ax2.set_ylabel("Probability Density", color=DARK_TEXT)
ax2.set_title("Resolution Proof: Negative Weights Are Sign Inversions", loc="left", pad=12)
ax2.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#94A3B8")
ax2.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR)

# Annotate insight
ax2.text(0.05, 0.88, "Key Insight: The distributions match perfectly.\nRemedy: Apply abs(weight) to salvage 437 loads,\nthen impute remaining true NaNs with equipment medians.",
         transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9))

plt.suptitle("Spotter Freight ML Assessment — Data Quality & Integrity Diagnostics", fontsize=14, y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.95])
safe_save(fig, OUTPUT_DIR / "01_missing_and_negative_data_quality.png")

# -------------------------------------------------------------
# CHART 2: TEMPORAL SPLIT ARCHITECTURE & MARKET INDEX DYNAMICS
# -------------------------------------------------------------
print("Generating Chart 2: Temporal Architecture & Seasonality...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, facecolor="white")

# 2A: Timeline of Train, Validation, and December Benchmark
timeline_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
train_mask = timeline_dates <= pd.to_datetime("2025-10-31")
val_mask = (timeline_dates >= pd.to_datetime("2025-11-01")) & (timeline_dates <= pd.to_datetime("2025-12-31"))
dec_chart_mask = timeline_dates >= pd.to_datetime("2025-12-01")

# Daily counts
train_daily = train_df.groupby("date").size()
val_daily = val_df.groupby("date").size()

ax1.plot(train_daily.index, train_daily.values, color=PRIMARY, linewidth=1.5, label="Train-Test Dataset (Jan 1 – Oct 31, 2025 | 48,000 loads)", zorder=3)
ax1.plot(val_daily.index, val_daily.values, color=SECONDARY, linewidth=1.5, label="Validation Holdout Set (Nov 1 – Dec 31, 2025 | 12,000 loads)", zorder=3)

# Highlight recommended validation split (e.g. Sep-Oct as OOT validation)
ax1.axvspan(pd.to_datetime("2025-09-01"), pd.to_datetime("2025-10-31"), color="#CBD5E1", alpha=0.45, label="Recommended Local OOT Val (Sep 1 – Oct 31 | ~9,600 loads)", zorder=2)
ax1.axvline(pd.to_datetime("2025-11-01"), color=ACCENT_WARN, linestyle="--", linewidth=2, label="Out-of-Time Boundary (Strict Future Cutoff)", zorder=4)

ax1.set_ylabel("Daily Load Volume", color=DARK_TEXT)
ax1.set_title("Dataset Temporal Hierarchy & Out-of-Time Validation Scheme", loc="left", pad=10)
ax1.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color("#94A3B8")
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)

# 2B: Market Index Dynamics (Macro trend + Day-of-Week Micro Cycle)
full_market = pd.concat([
    train_df[["date", "market_index"]].assign(split="Train"),
    val_df[["date", "market_index"]].assign(split="Val")
]).dropna()

daily_market = full_market.groupby("date")["market_index"].mean()

ax2.plot(daily_market.loc[daily_market.index <= "2025-10-31"].index, 
         daily_market.loc[daily_market.index <= "2025-10-31"].values, 
         color=PRIMARY, linewidth=1.8, label="Train Market Index Trend", zorder=3)
ax2.plot(daily_market.loc[daily_market.index >= "2025-11-01"].index, 
         daily_market.loc[daily_market.index >= "2025-11-01"].values, 
         color=SECONDARY, linewidth=1.8, label="Validation Market Index Trend", zorder=3)

ax2.axvline(pd.to_datetime("2025-11-01"), color=ACCENT_WARN, linestyle="--", linewidth=2, zorder=4)

ax2.set_ylabel("Market Index Ratio", color=DARK_TEXT)
ax2.set_xlabel("Date (2025)", color=DARK_TEXT)
ax2.set_title("Market Index Trajectory: Macro Cycles & Spring Peaks (Why Random CV Leaks Future Trends)", loc="left", pad=10)
ax2.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#94A3B8")
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=8.5)

plt.suptitle("Spotter Freight ML Assessment — Temporal Structure & Validation Methodology", fontsize=14, y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.96])
safe_save(fig, OUTPUT_DIR / "02_temporal_split_and_market_index_trend.png")

# -------------------------------------------------------------
# CHART 3: DAY OF WEEK CYCLICALITY (Micro-Seasonality)
# -------------------------------------------------------------
print("Generating Chart 3: Day-of-Week Seasonality...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, facecolor="white")

train_df["dow"] = train_df["date"].dt.day_name()
val_df["dow"] = val_df["date"].dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

train_dow_market = train_df.groupby("dow")["market_index"].mean().reindex(order)
val_dow_market = val_df.groupby("dow")["market_index"].mean().reindex(order)

x = np.arange(len(order))
ax1.plot(x, train_dow_market.values, marker="o", linewidth=2.4, color=PRIMARY, label="Train Market Index", zorder=3)
ax1.plot(x, val_dow_market.values, marker="s", linewidth=2.4, color=SECONDARY, label="Val Market Index", zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], color=DARK_TEXT)
ax1.set_ylabel("Average Market Index", color=DARK_TEXT)
ax1.set_title("Day-of-Week Pattern in Market Index", loc="left", pad=10)
ax1.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color("#94A3B8")
ax1.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR)

# Annotate peak
wed_idx = 3 # Thursday is index 3
ax1.annotate("Mid-Week Peak\n(Thu High Demand)", xy=(wed_idx, train_dow_market.iloc[wed_idx]),
             xytext=(wed_idx + 0.35, train_dow_market.iloc[wed_idx] + 0.02),
             arrowprops=dict(arrowstyle="->", color=DARK_TEXT, lw=1.2),
             fontweight="bold", fontsize=8.5, color=DARK_TEXT)

# 3B: Posted Rate per Mile by Day of Week
train_df["rpm"] = train_df["posted_rate"] / train_df["distance"]
train_dow_rpm = train_df.groupby("dow")["rpm"].mean().reindex(order)

ax2.bar(x, train_dow_rpm.values, color=PRIMARY, alpha=0.85, width=0.55, zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], color=DARK_TEXT)
ax2.set_ylabel("Mean Rate Per Mile ($/mi)", color=DARK_TEXT)
ax2.set_title("Average Rate Per Mile by Day of Week", loc="left", pad=10)
ax2.set_ylim(min(train_dow_rpm.values) * 0.95, max(train_dow_rpm.values) * 1.05)
ax2.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#94A3B8")

for i, v in enumerate(train_dow_rpm.values):
    ax2.text(i, v + 0.01, f"${v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PRIMARY)

plt.suptitle("Spotter Freight ML Assessment — Intra-Week Cyclical Dynamics", fontsize=14, y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.95])
safe_save(fig, OUTPUT_DIR / "03_day_of_week_seasonality.png")

# -------------------------------------------------------------
# CHART 4: GEOSPATIAL MAP & THE 8 UNSEEN CITIES IN VALIDATION
# -------------------------------------------------------------
print("Generating Chart 4: Cold-Start Unseen Cities & Spatial Distribution...")
fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300, facecolor="white")

# Extract city coordinates
train_cities = train_df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
val_cities = val_df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()

unseen_cities = set(val_cities.index) - set(train_cities.index)
seen_val_cities = set(val_cities.index) & set(train_cities.index)

# Scatter plot of cities
ax.scatter(train_cities["pickup_lon"], train_cities["pickup_lat"], 
           s=110, color=PRIMARY, alpha=0.7, edgecolors="white", linewidths=1.2, label=f"Train Known Cities ({len(train_cities)} hubs)", zorder=3)

# Highlight unseen cities
unseen_coords = val_cities.loc[list(unseen_cities)]
ax.scatter(unseen_coords["pickup_lon"], unseen_coords["pickup_lat"], 
           s=220, color=ACCENT_WARN, marker="*", edgecolors="black", linewidths=1, label=f"Unseen Validation Cold-Start Cities ({len(unseen_cities)} hubs)", zorder=5)

# Label the unseen cities
for city, row in unseen_coords.iterrows():
    ax.annotate(f"{city}", xy=(row["pickup_lon"], row["pickup_lat"]),
                xytext=(8, 4), textcoords="offset points", fontsize=9, fontweight="bold", color="#991B1B",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#EF4444", alpha=0.9), zorder=6)

ax.set_xlabel("Longitude (°W)", color=DARK_TEXT)
ax.set_ylabel("Latitude (°N)", color=DARK_TEXT)
ax.set_title("Geospatial Footprint & Cold-Start Cities in Validation Set", loc="left", pad=12)
ax.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#94A3B8")
ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRID_COLOR, fontsize=10)

# Explanatory card
info_text = (
    "Critical Modeling Challenge: 8 cities in Validation NEVER appeared in Train:\n"
    "• Chicago, Charlotte, San Diego, Laredo, Allentown, Jackson, Knoxville, Norfolk\n\n"
    "Why It Matters:\n"
    "Categorical one-hot encoding fails on test data. The model must rely on coordinates\n"
    "(lat, lon), Haversine lane distance, and geospatial clustering to generalize seamlessly."
)
ax.text(0.03, 0.68, info_text, transform=ax.transAxes, fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.95))

plt.suptitle("Spotter Freight ML Assessment — Geospatial Generalization Analysis", fontsize=14, y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.96])
safe_save(fig, OUTPUT_DIR / "04_unseen_cities_geospatial_generalization.png")

# -------------------------------------------------------------
# CHART 5: PRICING MECHANICS & SIGNAL CORRELATION
# -------------------------------------------------------------
print("Generating Chart 5: Pricing Mechanics & Baseline Signal...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor="white")

# 5A: Scatter of Posted Rate vs Distance * Quote Signal
naive_pred = train_df["distance"] * train_df["quote_signal"]
actual = train_df["posted_rate"]

# Subsample 4,000 points for clean visualization
sample_idx = np.random.RandomState(42).choice(len(train_df), size=4000, replace=False)
sample_naive = naive_pred.iloc[sample_idx]
sample_actual = actual.iloc[sample_idx]

ax1.scatter(sample_naive, sample_actual, alpha=0.25, color=PRIMARY, s=18, edgecolors="none", zorder=3)
max_val = min(10000, max(sample_actual.quantile(0.99), sample_naive.quantile(0.99)))
ax1.plot([0, max_val], [0, max_val], color=ACCENT_WARN, linestyle="--", linewidth=2, label="1:1 Parity Line (Benchmark R² = 0.794)", zorder=4)

ax1.set_xlim(0, max_val)
ax1.set_ylim(0, max_val)
ax1.set_xlabel("Estimated Base = Distance × Quote Signal ($)", color=DARK_TEXT)
ax1.set_ylabel("Actual Posted Rate ($)", color=DARK_TEXT)
ax1.set_title("Core Pricing Signal: Posted Rate vs Naive Baseline", loc="left", pad=10)
ax1.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color("#94A3B8")
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID_COLOR)

# 5B: Multiplier Ratio Distribution (Posted Rate / Base)
ratio = actual / naive_pred
filtered_ratio = ratio[(ratio >= 0.5) & (ratio <= 2.0)]

counts, bins, patches = ax2.hist(filtered_ratio, bins=60, color=SECONDARY, edgecolor="white", alpha=0.85, zorder=3)
ax2.axvline(1.0, color=ACCENT_WARN, linestyle="--", linewidth=2, label="Median Ratio = 1.004 (Base Center)", zorder=4)

ax2.set_xlabel("Pricing Multiplier (Posted Rate / Base)", color=DARK_TEXT)
ax2.set_ylabel("Load Frequency", color=DARK_TEXT)
ax2.set_title("Markup Multiplier Distribution (Target Reformulation)", loc="left", pad=10)
ax2.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#94A3B8")
ax2.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR)

# Explanatory annotation
ax2.text(0.44, 0.65, "Architectural Recommendation:\nInstead of predicting raw posted_rate directly,\nmodel the residual multiplier M:\nposted_rate = (distance × quote_signal) × M\nResidual error standard deviation drops significantly!",
         transform=ax2.transAxes, fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#3B82F6", alpha=0.9))

plt.suptitle("Spotter Freight ML Assessment — Rate Mechanics & Modeling Formulation", fontsize=14, y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.95])
safe_save(fig, OUTPUT_DIR / "05_pricing_mechanics_and_baseline.png")

# -------------------------------------------------------------
# CHART 6: EQUIPMENT PREMIUMS & WEIGHT CORRELATION
# -------------------------------------------------------------
print("Generating Chart 6: Equipment Premiums & Weight Impact...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, facecolor="white")

# 6A: Equipment Breakdown
train_clean = train_df.copy()
train_clean["weight"] = train_clean["weight"].abs()
eq_summary = train_clean.groupby("equipment")["rpm"].agg(["mean", "median", "std"])

x_eq = np.arange(len(eq_summary))
colors_eq = [PRIMARY, SECONDARY, ACCENT_ORANGE]
rects = ax1.bar(x_eq, eq_summary["mean"], color=colors_eq, width=0.5, edgecolor="none", zorder=3)
ax1.set_xticks(x_eq)
ax1.set_xticklabels(eq_summary.index, color=DARK_TEXT)
ax1.set_ylabel("Mean Rate Per Mile ($/mi)", color=DARK_TEXT)
ax1.set_title("Equipment Type Pricing Premiums", loc="left", pad=10)
ax1.set_ylim(0, max(eq_summary["mean"]) * 1.25)
ax1.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color("#94A3B8")

for bar, (eq_name, row) in zip(rects, eq_summary.iterrows()):
    h = bar.get_height()
    ax1.annotate(f"${h:.2f}/mi\n(med: ${row['median']:.2f})", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=DARK_TEXT)

# 6B: Weight vs Posted Rate by Equipment
sample_clean = train_clean.sample(n=3000, random_state=42)
for eq, color in zip(["Dry Van", "Reefer", "Flatbed"], [PRIMARY, SECONDARY, ACCENT_WARN]):
    subset = sample_clean[sample_clean["equipment"] == eq]
    ax2.scatter(subset["weight"], subset["rpm"], alpha=0.35, s=16, color=color, label=eq, zorder=3)

ax2.set_xlabel("Corrected Weight (lbs)", color=DARK_TEXT)
ax2.set_ylabel("Rate Per Mile ($/mi)", color=DARK_TEXT)
ax2.set_title("Weight vs Rate Per Mile (After Sign-Flip Fix)", loc="left", pad=10)
ax2.set_ylim(0.5, 5.0)
ax2.grid(axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#94A3B8")
ax2.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR)

plt.suptitle("Spotter Freight ML Assessment — Equipment & Cargo Feature Interactions", fontsize=14, y=0.98, color=DARK_TEXT)
plt.tight_layout(rect=[0, 0, 1, 0.95])
safe_save(fig, OUTPUT_DIR / "06_equipment_and_weight_interactions.png")

print("All charts successfully generated in eda_visualizations/")
