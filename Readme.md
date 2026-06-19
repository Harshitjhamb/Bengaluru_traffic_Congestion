

# Bengaluru Event-Driven Congestion — Response Recommender

**Gridlock Hackathon 2.0 · Round 2 · Theme 2 — Event-Driven Congestion (Planned & Unplanned)**

## Problem
Political rallies, festivals, sports events, construction, and breakdowns create localized
traffic breakdowns across Bengaluru. Event impact isn't quantified in advance, resource
deployment is experience-driven, and there's no systematic post-event learning. This project
predicts incident impact from historical ASTRAM event data and recommends manpower,
barricading, and station deployment for incoming events.

## Approach
A chained 3-model pipeline feeds a rule-based recommendation engine:

1. **Priority Classifier** (RandomForest) — predicts High/Low priority from location, time,
   and event cause.
2. **Road Closure Classifier** (RandomForest, threshold-tuned) — predicts whether an event
   will require closing the road.
3. **Duration Regressor** (RandomForest, log-transformed target) — predicts expected
   resolution time in hours.

At inference time, priority and closure are predicted first, then combined with domain
severity weights to estimate a composite risk score — which drives the duration prediction,
manpower recommendation, and barricading decision. Police station assignment uses a
historical zone lookup with a GPS-nearest-neighbor fallback; corridor is auto-detected from
GPS coordinates rather than required as input.

## ⭐ A key finding: catching data leakage
Our first priority model scored 99.94% accuracy — which was a red flag, not a win. Investigation
showed `corridor` was a near-deterministic proxy for `priority` (named corridor → ~99-100% High,
non-corridor → ~99.8% Low) — almost certainly an operational labeling rule, not a learnable
pattern. Removing it dropped accuracy to a much more honest **77.0%** — but that number reflects
genuine learning from spatiotemporal signals (latitude/longitude dominate feature importance),
not memorization of a lookup table.

## Results

| Model | Metric | Score |
|---|---|---|
| Priority Classifier | Accuracy / F1 | 77.0% / 0.82 |
| Road Closure Classifier | F1 (tuned threshold=0.567) | 0.395 (Precision 0.38, Recall 0.41) |
| Duration Regressor | MAE (log-transformed) | 1.11 hrs |

Road closure is a genuinely hard problem — only 7.3% of events require one — but the tuned
model still beats the majority-class baseline by several times on the minority class.

## Repo Structure
``` text
FLIPKART_GRID/
│
├── app/
│   └── app.py
│
├── assets/
│   ├── bengaluru_hotspot_map.html
│   ├── eda_extra1_planned_vs_unplanned.png
│   ├── model1_confusion_matrix.png
│   ├── model1_feature_importance.png
│   ├── step2_distributions.png
│   ├── step3_time_patterns.png
│   ├── step4_severity.png
│   ├── step5_corridors_zones.png
│   └── step5_zone_time_matrix.png
│
├── docs/
│   └── vide_lind.md
│
├── models/
│   └── recommendation_engine_bundle.pkl
│
├── notebook/
│   └── Flipkart_grid_notebook_complete.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```
## Folder Description

### 📂 app
Contains the Streamlit application used for:
- Event Simulation
- Congestion Prediction
- Resource Recommendation
- Interactive Dashboard

---

### 📂 assets
Contains all generated visualizations and analytics outputs:

| File | Description |
|--------|-------------|
| bengaluru_hotspot_map.html | Interactive Bengaluru incident heatmap |
| eda_extra1_planned_vs_unplanned.png | Planned vs Unplanned event analysis |
| model1_confusion_matrix.png | Priority prediction confusion matrix |
| model1_feature_importance.png | Feature importance visualization |
| step2_distributions.png | Incident distribution plots |
| step3_time_patterns.png | Hourly, weekly and monthly traffic patterns |
| step4_severity.png | Severity analysis |
| step5_corridors_zones.png | Corridor and zone analysis |
| step5_zone_time_matrix.png | Zone-time incident matrix |

---

### 📂 docs
Contains project documentation, methodology, and supporting notes.

---

### 📂 models
Stores serialized machine learning models and recommendation engine artifacts.

Current:
- recommendation_engine_bundle.pkl

Future:
- priority_model.pkl
- duration_model.pkl
- road_closure_model.pkl

---

### 📂 notebook
Contains the complete development notebook including:
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Model Training
- Evaluation
- Visualization

---

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
Open `http://localhost:8501` and enter an event's type, cause, GPS coordinates, and time to
get a full recommendation: priority, closure risk, expected duration, officer count, and
which police station to deploy from.

## Limitations & Future Work
- Corridor detection from GPS uses k-nearest-neighbor on historical incident locations, not
  true road geometry — a proper geofencing layer would be more precise.
- Road closure recall (41%) leaves room for improvement with ground-condition data not present
  in this dataset.
- Manpower/barricading weights are domain-informed heuristics, not learned from outcome data
  (e.g., whether deployed manpower actually resolved incidents faster) — a natural next step
  if response-effectiveness data becomes available.

## Dataset
Provided by HackerEarth Gridlock Hackathon 2.0 (ASTRAM Bengaluru traffic event data,
anonymized). See challenge page for access.

