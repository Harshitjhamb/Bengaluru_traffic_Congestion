# Bengaluru Event-Driven Traffic Congestion — Response Recommender

> **Gridlock Hackathon 2.0 · Theme 2 · Event-Driven Congestion**  
> An ML-powered dashboard that predicts incident impact and recommends manpower, barricading, and police station deployment in real time.

---

## Overview of the project

Bengaluru handles thousands of traffic incidents daily — from vehicle breakdowns to VIP movements and protests. This project trains three machine learning models on 8,173 ASTRAM traffic events to:

- Predict **incident priority** (High / Low)
- Predict **road closure requirement** (Yes / No)
- Estimate **incident duration** (hours)
- Recommend **officer count**, **deployment station**, and **barricading** based on the above

The v2 Streamlit app uses **XGBoost** models for all three tasks, with rich feature engineering (circular sin/cos time encoding, haversine distances to 6 named congestion hotspots, interaction features) — the priority classifier reaches **ROC-AUC 0.99**, and mean prediction confidence across the priority + closure test sets is **~88%**.

---

## Project Structure

```
Bengaluru_Traffic_Congestion-main/
│
├── .github/workflows/
│   └── ci.yml                            # GitHub Actions — compile check + bundle/app smoke test
│
├── app/
│   └── app.py                            # Streamlit dashboard (v2, XGBoost + feature engineering)
│
├── assets/                                  # EDA & model visualisation outputs
│   ├── bengaluru_hotspot_map.html           # Interactive Folium map of congestion hotspots
│   ├── eda_extra1_planned_vs_unplanned.png
│   ├── model1_confusion_matrix.png          # Priority classifier confusion matrix (XGBoost v2)
│   ├── model1_feature_importance.png        # Top-15 feature importances (XGBoost v2)
│   ├── model2_confusion_matrix.png          # Closure classifier confusion matrix (XGBoost v2)
│   ├── step2_distributions.png              # Feature distributions
│   ├── step3_time_patterns.png              # Hourly / day-of-week traffic patterns
│   ├── step4_severity.png                   # Severity index breakdown
│   ├── step5_corridors_zones.png            # Corridor & zone analysis
│   └── step5_zone_time_matrix.png           # Zone × time-block heatmap
│
├── Dataset/                               # gitignored — place Hack_dataset.csv here to retrain
│
├── docs/
│   └── vide_lind.md                       # Additional project notes / video link
│
├── models/
│   ├── recommendation_engine_bundle_v2.pkl   # Deployed bundle — XGBoost × 3 (what app.py loads)
│   ├── recommendation_engine_bundle.pkl      # v1 baseline — RandomForest × 3 (comparison only)
│   └── v2_metrics.json                       # Real, computed v2 evaluation numbers
│
├── notebook/
│   ├── Flipkart_grid_notebook_complete.ipynb # v1 EDA + RandomForest baseline (Google Colab)
│   └── train_v2.py                           # v2 standalone training script (XGBoost, deployed)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ML Pipeline

### Dataset
- **Source:** ASTRAM Bengaluru Traffic Events (`Hack_dataset.csv`)
- **Size:** 8,173 rows × 46 columns
- **Key columns:** `event_type`, `event_cause`, `latitude`, `longitude`, `priority`, `requires_road_closure`, `start_datetime`, `closed_datetime`, `corridor`, `zone`, `police_station`

### Preprocessing (Notebook)
1. Drop fully-null columns (`map_file`, `comment`, `meta_data`)
2. Parse all datetime columns;remove rows with`end_datetime < start_datetime`
3. Drop columns with >90% missing values
4. Replace placeholder `0.0` in `endlatitude` / `endlongitude` with `NaN`
5. Derive `duration_hrs` from `(closed_datetime − start_datetime)`
6. Extract time features:`hour`, `day_of_week`, `month_num`
7. Engineer binary flags: `is_weekend`, `is_peak_hour` (7–9 AM,5–8 PM), `is_night` (10 PM–6 AM)
8. Normalise `event_cause` (lowercase + strip)

### Feature Engineering (v2 App)
- **Circular time encoding:** `sin/cos` of hour, month, day-of-week
- **Haversine distances** to 6 known congestion hotspots (MG Road, Silk Board, Hebbal, Marathahalli, Whitefield, Electronic City)
- **Interaction features:** `peak_x_cause`, `weekend_x_cause`
- **Cause severity score** from lookup map
---

### Exploratory Data Analysis

**Event Distributions** — cause breakdown, planned vs unplanned, priority, status
![Event distributions](assets/step2_distributions.png)

**Time Patterns** — incidents by hour/day/month, hour×day heatmap
![Time patterns](assets/step3_time_patterns.png)

**Severity Analysis** — duration by cause, closure rate by cause, severity index distribution
![Severity analysis](assets/step4_severity.png)

**Corridor & Zone Analysis** — top corridors by risk, zone-wise incident load, police station load
![Corridor and zone analysis](assets/step5_corridors_zones.png)

**Zone × Time Manpower Matrix** — deployment planning heatmap
![Zone time matrix](assets/step5_zone_time_matrix.png)

**Planned vs Unplanned — Cause Comparison**
![Planned vs unplanned](assets/eda_extra1_planned_vs_unplanned.png)

**Priority Classifier — Confusion Matrix**
![Priority confusion matrix](assets/model1_confusion_matrix.png)

---

### Models

| # | Task | Algorithm | Key Metric | v1 RandomForest baseline |
|---|------|-----------|------------|---------------------------|
| 1 | Priority classification (High / Low) | **XGBoost** (`scale_pos_weight`, v2 features) | Acc 0.9694 · F1 0.9754 · **ROC-AUC 0.9924** | Acc 0.7697 · F1 0.8171 |
| 2 | Road closure classification (Yes / No) | **XGBoost** (`scale_pos_weight` for 7.3% imbalance) | Acc 0.9124 · F1 0.3694 · AUC 0.7724 (threshold 0.609) | Acc 0.9080 · F1 0.3951 |
| 3 | Duration regression (hours) | **XGBoost Regressor** (log1p-transformed target) | MAE 1.08 hrs · R²(log) 0.219 | MAE 1.11 hrs · R² 0.197 |

v1 was a RandomForest baseline trained directly on raw lat/lon + categorical features. v2 re-engineers the feature set (circular time encoding, haversine hotspot distances, interaction terms) and switches to XGBoost — this is what's deployed in `app.py`. The priority classifier is the biggest winner: the added spatial/time features let it separate High/Low priority almost perfectly. Closure is comparable to the v1 baseline (a hard, ~7%-positive-class problem where more features help only marginally); duration improved modestly.

**Imbalance handling:** `scale_pos_weight` (tuned via a small multiplier sweep) passed to the XGBoost closure model.  
**Leakage prevention:** `priority_score` and `closure_score` excluded from classifier features; `severity_index` used only for duration regression.

### Model Bundle (`recommendation_engine_bundle_v2.pkl`)
Serialised with `joblib` by `notebook/train_v2.py`, the bundle contains:
- `priority_model`, `closure_model`, `duration_model` — all three are `XGBClassifier`/`XGBRegressor`
- `priority_feature_cols`, `closure_feature_cols`, `duration_feature_cols`
- `cat_features`, `num_features`, `cat_features_fixed`, `num_features_fixed`
- `closure_threshold` (optimised for class imbalance via precision-recall sweep)
- `cause_score_map`, `manpower_map`
- `zone_station_map`, `station_coords`, `corridor_coords`
- `hotspots` — lat/lon of the 6 named congestion points used for haversine distance features and the app's live map overlay

A separate `recommendation_engine_bundle.pkl` (no `_v2`) is kept alongside it purely as the v1 RandomForest baseline for comparison — `app.py` only ever loads the v2 bundle.

---

## Streamlit App (`app/app.py`)

### What It Does
1. Accepts an incoming traffic event (type, cause, GPS, time, zone)
2. Auto-detects the nearest **corridor** and **hotspot** via haversine distance
3. Runs all three models and computes a **severity score** (0–11)
4. Outputs:
   - Risk level (Low / Medium / High / Critical)
   - Predicted priority + confidence gauge
   - Road closure prediction + confidence gauge
   - Estimated duration
   - Recommended officer count
   - Recommended police station (zone lookup or GPS nearest-neighbour)
   - Barricading recommendation
   - Priority/closure confidence gauges
   - Distance-to-hotspot table + live Folium map with hotspot overlay

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Kartik-1818/Bengaluru_Traffic_Congestion
cd Bengaluru_Traffic_Congestion
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** The pre-trained v2 model bundle (`models/recommendation_engine_bundle_v2.pkl`) is included. No retraining is required to run the app.

---

## Reproducing the Training

### v2 (XGBoost, deployed) — `notebook/train_v2.py`
A standalone script (no Colab/Drive dependency) that reproduces the notebook's cleaning pipeline, adds the v2 feature set, and trains all three XGBoost models.

```bash
# Place the raw data at Dataset/Hack_dataset.csv (not tracked in git — see below)
python notebook/train_v2.py
```
This regenerates `models/recommendation_engine_bundle_v2.pkl`, `models/v2_metrics.json`, and the confusion-matrix / feature-importance PNGs under `assets/`.

> **Dataset:** `Hack_dataset.csv` (8,173 ASTRAM traffic events with addresses and GPS coordinates) is **not committed to this repo** — it's excluded via `.gitignore` since it contains real incident location/address data. Request it from the hackathon dataset owners and place it at `Dataset/Hack_dataset.csv` before running the script.

### v1 (RandomForest baseline) — `notebook/Flipkart_grid_notebook_complete.ipynb`
The original exploratory notebook, developed on **Google Colab**, that produced the v1 RandomForest baseline (`models/recommendation_engine_bundle.pkl`) and all the EDA charts under `assets/`. Kept for reference/comparison; superseded by `train_v2.py` for the actual deployed models.

### Install notebook-only dependencies
```bash
pip install xgboost imbalanced-learn plotly folium
```

---

## Dependencies (`requirements.txt`)

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
xgboost>=1.7
streamlit>=1.28
joblib>=1.3
folium>=0.14
seaborn>=0.12
matplotlib>=3.7
```

---


## EDA Highlights

| Insight | Detail |
|---------|--------|
| Dataset size | 8,173 traffic events, 46 raw columns |
| Event split | ~majority unplanned; minority planned |
| Peak hours | 7–9 AM and 5–8 PM |
| Top hotspots | Silk Board, Marathahalli, Hebbal, MG Road |
| Severity range | 0–11 composite score (cause + type + priority + closure) |

Visual outputs are saved to `assets/` and include distribution plots, time-pattern charts, corridor/zone heatmaps, and an interactive Folium map.


## Known Hotspot Coordinates

| Hotspot | Latitude | Longitude |
|---------|----------|-----------|
| MG Road | 12.9766 | 77.6075 |
| Silk Board | 12.9174 | 77.6228 |
| Hebbal | 13.0358 | 77.5970 |
| Marathahalli | 12.9563 | 77.7010 |
| Whitefield | 12.9698 | 77.7500 |
| Electronic City | 12.8399 | 77.6770 |

---

## License

This project was developed as a hackathon submission. Dataset rights belong to ASTRAM / the hackathon organisers.
