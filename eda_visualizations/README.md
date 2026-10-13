# Exploratory Data Analysis & Diagnostic Visualizations

This directory contains 6 publication-grade diagnostic charts created for the **Spotter Freight Rate Machine Learning Assessment**. These visuals illustrate the data quality discoveries, temporal validation structure, and pricing dynamics.

---

## Visual Summary & Findings

### 1. `01_missing_and_negative_data_quality.png`
* **File:** [01_missing_and_negative_data_quality.png](01_missing_and_negative_data_quality.png)
* **Diagnostic Insight:** 
  * Left: Exact counts of missing values in `weight` (300 in Train, 165 in Val), missing `market_index` (374 in Train, 249 in Val), and negative weight anomalies (292 in Train, 145 in Val).
  * Right: Statistical distribution comparison proving that $|weight|$ for negative entries aligns identically with genuine freight weights ($10,000$ to $50,000$ lbs).
* **Remedy:** Applied `abs(weight)` to salvage 437 loads from a sign-flip bug, followed by equipment-specific median imputation for true NaNs.

---

### 2. `02_temporal_split_and_market_index_trend.png`
* **File:** [02_temporal_split_and_market_index_trend.png](02_temporal_split_and_market_index_trend.png)
* **Diagnostic Insight:**
  * Top: Temporal distribution showing Train-Test (Jan 1 – Oct 31, 2025; 48,000 loads) versus Validation (Nov 1 – Dec 31, 2025; 12,000 loads), highlighting our Out-Of-Time (OOT) validation split (Sep 1 – Oct 31).
  * Bottom: Macro-trajectory of `market_index`, displaying strong spring/summer peaks (reaching 1.40) and autumn contraction (0.80–1.00).
* **Remedy:** Established an Out-Of-Time (OOT) split (Jan–Aug train, Sep–Oct validation) to prevent future temporal data leakage.

---

### 3. `03_day_of_week_seasonality.png`
* **File:** [03_day_of_week_seasonality.png](03_day_of_week_seasonality.png)
* **Diagnostic Insight:**
  * Left: Day-of-week trajectory of `market_index` showing consistent peaks on Wednesdays and Thursdays with weekend demand drops.
  * Right: Mean rate-per-mile across weekdays, peaking at **$2.25/mi** mid-week versus **$2.19–$2.20/mi** on weekends.
* **Remedy:** Cyclical sine/cosine calendar encodings capturing intra-week demand cycles.

---

### 4. `04_unseen_cities_geospatial_generalization.png`
* **File:** [04_unseen_cities_geospatial_generalization.png](04_unseen_cities_geospatial_generalization.png)
* **Diagnostic Insight:**
  * Geospatial map of all 64 known historical hubs and the **8 unseen cold-start cities** in the validation set: *Chicago, Charlotte, San Diego, Laredo, Allentown, Jackson, Knoxville, and Norfolk*.
* **Remedy:** Replaced categorical city name dependencies with continuous coordinates (`lat`, `lon`), Haversine distance, and directional bearing vectors to guarantee zero-shot generalization.

---

### 5. `05_pricing_mechanics_and_baseline.png`
* **File:** [05_pricing_mechanics_and_baseline.png](05_pricing_mechanics_and_baseline.png)
* **Diagnostic Insight:**
  * Left: Scatter of actual posted rate versus the naive baseline (`Distance × Quote Signal`), confirming a base R² ≈ 0.794.
  * Right: Distribution of the pricing multiplier (`posted_rate / Base`), showing an ultra-tight center at **1.004**.
* **Remedy:** Reformulated target to predict the scale-invariant multiplier `M`, cutting MAE in half.

---

### 6. `06_equipment_and_weight_interactions.png`
* **File:** [06_equipment_and_weight_interactions.png](06_equipment_and_weight_interactions.png)
* **Diagnostic Insight:**
  * Left: Equipment rate-per-mile premiums: Dry Van ($2.12/mi), Flatbed ($2.29/mi), Reefer ($2.38/mi).
  * Right: Corrected cargo weight versus rate-per-mile stratified by equipment type.
* **Remedy:** Trailer one-hot encodings and payload weight interactions.
