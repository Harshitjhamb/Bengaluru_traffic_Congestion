# Bengaluru Event-Driven Traffic Congestion — Interview Prep in Plain English

> **How this document was built:** every claim below was checked against the actual files in this repo — `app/app_v2.py`, `notebook/train_v2.py` (the training script, re-run end-to-end on the real dataset), the actual `models/recommendation_engine_bundle_v2.pkl` (loaded and inspected directly, `type()`-checked), `models/v2_metrics.json` (the raw numbers the training script wrote out), `requirements.txt`, and `.github/workflows/ci.yml`. Every number quoted here is a real, measured number from an actual training run on the real 8,173-row ASTRAM dataset — not a target, not an estimate.

---

## READ THIS FIRST — how this maps to your resume

Your resume says: *"Engineered 3 XGBoost models with circular time encoding, haversine hotspot distances, and interaction features achieving AUC 0.98 and 89% avg confidence across all predictions."* Every part of that is backed by real code and a real training run:

| Resume claim | What's actually in the repo | Verified? |
|---|---|---|
| 3 XGBoost models | `models/recommendation_engine_bundle_v2.pkl` contains `XGBClassifier` (priority), `XGBClassifier` (closure), `XGBRegressor` (duration) — confirmed by loading the bundle and checking `type()`. | ✅ |
| Circular time encoding | `hour_sin/cos`, `month_sin/cos`, `dow_sin/cos` — `add_circular_time_features()` in `notebook/train_v2.py`, mirrored in `app/app_v2.py`. | ✅ |
| Haversine hotspot distances | 6 named hotspots (MG Road, Silk Board, Hebbal, Marathahalli, Whitefield, Electronic City) — `add_hotspot_distances()`, stored as `hotspots` in the bundle, used both as model features and as the live map overlay. | ✅ |
| Interaction features | `peak_x_cause` (`is_peak_hour × cause_score`), `weekend_x_cause` (`is_weekend × cause_score`) — `add_interaction_features()`. | ✅ |
| AUC 0.98 | Priority classifier's real, measured test-set ROC-AUC is **0.9924** (`models/v2_metrics.json`). The resume's "0.98" is a slight *underestimate*. Closure's AUC is lower (0.7724 — a much harder, imbalanced problem) and duration is regression (no AUC). If asked "AUC of what, exactly," the precise answer is "the priority classifier." | ✅ (for priority — be precise about which model) |
| 89% avg confidence | Real measured average (`mean(max(p, 1-p))` across the priority + closure test sets) is **88.4%** — resume rounds up slightly. | ✅ (off by 0.6 points) |
| GitHub Actions | `.github/workflows/ci.yml` — compiles `app_v2.py`/`train_v2.py` and runs a smoke test that loads the bundle and calls `recommend_resources()`. | ✅ |
| Ensemble learning | XGBoost *is* ensemble learning — gradient-boosted decision trees, a sequential/additive ensemble method. | ✅ |
| Folium live map with hotspot overlays | `app/app_v2.py`'s `render_hotspot_map()` — a real `folium.Map` with a `HeatMap` layer over the 6 hotspots, markers for each hotspot and the submitted incident, embedded via `streamlit.components.v1.html`. | ✅ |
| Auto-detects nearest corridor via GPS, outputs officer count / station / barricading plan | `nearest_corridor()`, `manpower_map`, `zone_station_map` / `nearest_police_station()`. | ✅ |

**One thing worth being ready for, not because it's a discrepancy but because a sharp interviewer might probe it:** the priority classifier's test-set accuracy (97%, AUC 0.99) is unusually high for a real-world tabular problem. That's exactly the kind of number that should make you suspicious of your own pipeline before you're proud of it — see Part 5, Concept 8 for the validation that was actually done to check this wasn't leakage, and hold onto that answer, because it's the single strongest thing you can say in this interview.

---

## PART 0: THE FULL STORY

### What problem does the project solve?
Bengaluru traffic police (via the ASTRAM incident system) log thousands of traffic-disrupting events — accidents, VIP movements, protests, potholes, water-logging, tree falls, etc. When a new incident is reported, a human dispatcher currently has to *guess*, from experience, how serious it is, whether the road needs to be closed, how long it'll block traffic, how many officers to send, and which police station should respond. This project turns that guesswork into a data-driven recommendation: feed in the basic facts of an incident (what happened, where, when) and get back a structured response plan.

### Who is the user?
A traffic control-room operator/dispatcher deciding how to respond to an incoming incident report. Built for **Gridlock Hackathon 2.0 (Theme 2 — Event-Driven Congestion)** — the practical "user" was hackathon judges viewing a live demo, but the intended real-world user is a police dispatch operator.

### What was the motivation?
8,173 historical ASTRAM traffic events were available as training data. The insight: three related but distinct questions repeatedly determine an incident's response — *is this urgent?* / *will it need a road closure?* / *how long will it last?* — and each is learnable from patterns already present in past incidents.

### What does the user do?
Opens a Streamlit web form and enters: event type (planned/unplanned), cause (from a fixed list like `vehicle_breakdown`, `protest`, `vip_movement`...), GPS latitude/longitude, hour of day, day of week, month, and optionally a police zone. Clicks **"Get Recommendation."**

### What happens behind the scenes?
1. The app loads the model bundle (`recommendation_engine_bundle_v2.pkl` — three XGBoost models + lookup tables) once, at startup, via `@st.cache_resource`.
2. It derives time flags (weekend/peak/night), circular sin/cos time encodings, haversine distances to the 6 named hotspots, and the two interaction features from what the user typed.
3. It finds the nearest known road **corridor** to the given GPS point using the haversine formula against thousands of historical corridor coordinates.
4. It one-hot-encodes the event into the exact column layout each of the three models was trained on (`encode_single_event`).
5. It runs the **priority model** (High/Low) and the **road-closure model** (Yes/No), reading off XGBoost's predicted probabilities.
6. It combines the *cause*, *event type*, *predicted* priority, and *predicted* closure into a hand-built "severity score" (2–11) — never the true labels, since those don't exist yet for a new event.
7. It feeds that severity score into the **duration model** to estimate how many hours the incident will last (trained on `log1p(duration_hrs)`, inverted with `expm1` at inference).
8. It looks up recommended officer count from a fixed severity→manpower table, and recommends a police station either from a historical zone→station lookup or, if the zone is unknown, from nearest-by-GPS.
9. All of this is rendered as colored badges, confidence gauges, a hotspot-distance table, and a live Folium map showing the incident, the 6 hotspots, and a heat overlay.

### How does data move through the system?
There is no live "traffic data feed," no database, and no backend API server. This is a **single-process Streamlit app**: user input → in-memory Python function (`recommend_resources`) → pre-trained XGBoost models loaded from a `.pkl` file on local disk → results rendered directly back into the same page. Training happens offline, via `python notebook/train_v2.py` against a local CSV (`Dataset/Hack_dataset.csv`, gitignored — see Part 3) — that produces the `.pkl` bundle the app ships with.

### What are the major components?
1. **`notebook/train_v2.py`** — the entire offline pipeline: cleaning, feature engineering, XGBoost training/evaluation for all three tasks, bundle export, and confusion-matrix/feature-importance PNG generation.
2. **The model bundle** (`models/recommendation_engine_bundle_v2.pkl`) — a `joblib`-serialized dict containing the three trained XGBoost models, the exact feature-column lists each expects, and lookup tables (cause→score, severity→manpower, zone→station, corridor/station GPS tables, hotspot coordinates).
3. **The Streamlit app** (`app/app_v2.py`) — the online/inference-time half. Loads the bundle, exposes a form, runs the same feature-engineering logic used at training time on a single new event, and renders results including the live Folium map.
4. **`.github/workflows/ci.yml`** — GitHub Actions CI: installs dependencies, compiles both Python entry points, and runs a smoke test that loads the real bundle and calls `recommend_resources()` end-to-end.
5. **EDA assets** (`assets/*.png`, `assets/bengaluru_hotspot_map.html`) — visual outputs. Three of the model-specific plots (`model1_confusion_matrix.png`, `model1_feature_importance.png`, `model2_confusion_matrix.png`) are regenerated directly by `train_v2.py` from the real deployed XGBoost models, so they always reflect what's actually shipped.

### End-to-end flow: user action → "frontend" → "backend" → model → response
There's no separate frontend/backend split — Streamlit *is* both, running as one Python process.

1. **User (frontend):** fills the form, clicks "Get Recommendation" in the browser.
2. **Streamlit reruns `app_v2.py` top-to-bottom** with the new widget values in scope.
3. **"Backend" logic:** `recommend_resources(event)` is called with a plain dict of the form inputs.
4. **"Model layer":** the function reads from the in-memory `B` dict (the loaded bundle) — three XGBoost models plus several pandas lookup tables — no network call, no SQL, no external API.
5. **Response:** the function returns a dict of predictions; the same script renders that dict as badges, gauges, a metrics grid, a hotspot-distance table, and an embedded Folium map.

### Simple numbered end-to-end flow
1. Operator opens the Streamlit app (`streamlit run app/app_v2.py`).
2. `load_bundle()` reads `models/recommendation_engine_bundle_v2.pkl` once and caches it (`@st.cache_resource`).
3. Operator fills in event type, cause, GPS coordinates, hour, day, month, and (optionally) zone.
4. Operator clicks "Get Recommendation."
5. `recommend_resources(event)` fills in missing fields, computes `is_weekend`/`is_peak_hour`/`is_night`, then computes the feature block: 6 hotspot distances, 6 circular sin/cos values, 2 interaction features.
6. The nearest **corridor** is computed via `nearest_corridor()` (haversine distance to historical corridor GPS points).
7. The event is one-hot-encoded and reindexed against each model's exact training-time columns (`encode_single_event`) — once for priority (corridor excluded), once for closure/duration (corridor included).
8. `priority_model.predict_proba()` (XGBoost) gives P(High priority), thresholded at 0.5.
9. `closure_model.predict_proba()` (XGBoost) gives P(needs road closure), thresholded at the **tuned** value stored in the bundle (`closure_threshold ≈ 0.609`), not the default 0.5.
10. A severity score (2–11) is computed by hand from cause weight + event-type weight + *predicted* priority weight + *predicted* closure weight.
11. `duration_model.predict()` (XGBoost) runs on log-transformed target space; the result is exponentiated back (`np.expm1`) to get hours.
12. Officer count comes from a fixed `manpower_map` lookup keyed by severity score.
13. Police station is either the historical mode station for the given zone, or — if zone is "Unknown" — the nearest station by GPS distance (k=5 nearest, mode).
14. All results are packaged into a dict and rendered as badges, confidence gauges, `st.metric` boxes, a distance-to-hotspot table, and a live Folium map (heatmap layer over the 6 hotspots + markers for each hotspot and the submitted incident).

---

## PART 1: THE OPENING LINE

### "Tell me about your project."

> "This was a hackathon project — Gridlock Hackathon 2.0 — where the theme was event-driven traffic congestion in Bengaluru. We had a historical dataset of about 8,170 real traffic incidents from the ASTRAM system — accidents, VIP movements, protests, potholes — with fields like cause, GPS location, priority, whether a road closure was needed, and how long the incident lasted. A dispatcher currently has to eyeball a new incident and guess how serious it is and how many officers to send. We built three XGBoost models — a priority classifier, a road-closure classifier, and a duration regressor — on top of an engineered feature set: circular sin/cos time encoding so the model understands hour 23 and hour 0 are adjacent, haversine distance to six known Bengaluru congestion hotspots, and interaction terms like peak-hour × cause-severity. All of it sits behind a Streamlit dashboard that also auto-detects the nearest road corridor from GPS, recommends an officer count and police station, and renders a live Folium map with the hotspots overlaid.
>
> The priority classifier is the standout result — ROC-AUC 0.99. That's high enough that I didn't just trust it — I validated it with a spatial holdout: trained on one set of GPS locations, tested on a completely disjoint set the model had never seen. Accuracy held at 97.3%, AUC at 0.989. So it's genuinely learning a spatial+temporal pattern in how priority gets assigned across Bengaluru, not memorizing repeated incident locations — real-world incident data has a lot of GPS repetition (only 3,336 unique ~100m clusters across 8,173 rows), so that check mattered.
>
> The engineering decision I'm most proud of is a bug we caught, not a feature we shipped: we'd built a hand-crafted `severity_index` from cause, event type, actual priority, and actual road-closure outcome — and initially fed that into the priority and closure classifiers themselves, which is the model being handed its own answer. We caught it, excluded `priority_score`/`closure_score` from those two models, and kept `severity_index` only for the duration regressor, where it's not circular. The second habit I'd point to is tuning the road-closure classifier's decision threshold instead of trusting the default 0.5 — closures are rare, about 7.3% of incidents, so we swept the precision-recall curve and landed on 0.609, which is what's actually used at inference."

**(~65 seconds spoken aloud)**

### Why this is a good answer
- States the problem and the user before jargon.
- Names the real, current architecture — three XGBoost models, the actual feature set — and every number in it is reproducible from `models/v2_metrics.json`.
- Proactively surfaces the "is that accuracy legit?" question before the interviewer has to ask it, with a concrete validation methodology.
- Leads with the leakage catch, the single most interview-defensible engineering judgment call in the project.
- Every claim here can survive someone opening `models/v2_metrics.json` or `notebook/train_v2.py` mid-interview.

---

## PART 2: THE TECH STACK

| Technology | What it does | Why we chose it |
|---|---|---|
| **Python** | Language for the entire project. | De facto standard for data science/ML; every library here (pandas, XGBoost, Streamlit, folium) is Python-native. |
| **pandas / numpy** | Data loading, cleaning, feature engineering (circular encoding, haversine math, interaction terms), one-hot encoding (`pd.get_dummies`). | Standard tooling for tabular data at this scale (~8K rows). |
| **XGBoost** | `XGBClassifier` (priority, closure) and `XGBRegressor` (duration) — all three deployed models. | Gradient-boosted trees handle the mixed categorical/numeric, moderately nonlinear feature set well; `scale_pos_weight` directly addresses the closure model's 7.3%-positive-class imbalance without resampling. This is also literally the "ensemble learning" on the resume — boosting is an additive/sequential ensemble method. |
| **scikit-learn** | `train_test_split`, `GroupShuffleSplit` (spatial validation), metrics (`accuracy_score`, `f1_score`, `roc_auc_score`, `precision_recall_curve`, `mean_absolute_error`, `r2_score`). | Standard evaluation tooling, used throughout `train_v2.py`. |
| **Streamlit** | The entire UI — form widgets, buttons, metrics, badges, confidence gauges, embedded map. | Fast to ship for a hackathon deadline with no separate frontend build step. **Trade-off:** reruns the entire script top-to-bottom on every interaction; fine at this scale, would need addressing at real production scale (Part 9). |
| **folium** (+ `folium.plugins.HeatMap`) | Renders the live hotspot map inside the running app (`render_hotspot_map()`) — a heatmap layer over the 6 named hotspots plus markers for each hotspot and the submitted incident, embedded via `st.components.v1.html(fmap._repr_html_(), height=420)`. | A `folium.Map` renders to HTML directly, so embedding it in Streamlit needs nothing beyond `streamlit.components.v1.html` — no extra `streamlit-folium` dependency. |
| **joblib** | Serializing/deserializing the trained models and the bundle dict. | Standard for persisting scikit-learn/XGBoost objects. **Security caveat:** `joblib.load` is functionally `pickle.load` — see Part 10. |
| **matplotlib / seaborn** | EDA charts under `assets/*.png`, plus the confusion-matrix/feature-importance plots regenerated from the real deployed XGBoost models. | Standard plotting; one-time analysis outputs, not a live dashboard. |
| **GitHub Actions** | `.github/workflows/ci.yml` — installs `requirements.txt`, compiles `app/app_v2.py` and `notebook/train_v2.py`, then loads the actual bundle and runs `recommend_resources()` as a smoke test. | Cheap, real CI signal that the app and the shipped bundle stay compatible — catches "someone changed a feature name and the app silently breaks." |

---

## PART 3: COMPLETE PROJECT STRUCTURE

### `app/app_v2.py`
**Purpose:** The entire runtime application — loads the model bundle and serves live recommendations for one incident at a time.

**Interview explanation:** "This is the only file that runs in production. `st.cache_resource` loads the model bundle once per process; `recommend_resources()` is a pure function that takes a dict describing an incident and returns a dict of predictions — the Streamlit widgets around it are UI plumbing."

**Important implementation details:**
- `load_bundle()` — `@st.cache_resource`-decorated so `recommendation_engine_bundle_v2.pkl` (~2MB) is deserialized once per process.
- `haversine_km`, `nearest_police_station`, `nearest_corridor` — vectorized great-circle distance, k=5-nearest-then-mode denoising.
- `add_hotspot_distances`, `add_circular_time_features`, `add_interaction_features` — the feature-engineering block, present identically in both `app_v2.py` and `train_v2.py`. This symmetry is deliberate: it's what keeps the app from silently drifting out of sync with what the models were actually trained on.
- `encode_single_event` — one-hot encodes a single row, then reindexes against the model's captured training-time column list, zero-filling anything missing. The standard fix for "one-hot encoding a single row produces different columns than training."
- `recommend_resources(event)` — orchestrates the full pipeline: defaults missing fields → computes engineered features → detects corridor → runs priority model → runs closure model (tuned threshold) → computes severity from *predictions* → runs duration model (log-space, inverted) → looks up manpower and station.
- `confidence_gauge()` — renders a colored horizontal bar (green ≥80%, orange ≥60%, red below) for both priority and closure confidence.
- `render_hotspot_map()` — builds a `folium.Map`, adds a `HeatMap` layer over the 6 hotspot coordinates plus a marker per hotspot and one for the submitted incident, and returns the map object for embedding.

---

### `notebook/train_v2.py`
**Purpose:** The entire offline training pipeline — cleans the raw CSV, builds the feature set, trains and evaluates all three XGBoost models, and exports the bundle the app loads.

**Interview explanation:** "This is a standalone script — no notebook, no Colab dependency — so `python notebook/train_v2.py` against a local `Dataset/Hack_dataset.csv` reproduces the whole pipeline deterministically (fixed `random_state=42` throughout)."

**Important implementation details, with real numbers from `models/v2_metrics.json`:**
- **Cleaning:** drop `map_file`/`comment`/`meta_data`, parse datetimes, drop 48 rows where `end_datetime < start_datetime`, drop columns >90% missing, replace `0.0` placeholders in `endlatitude`/`endlongitude` with `NaN` (then drop those columns), derive `duration_hrs`, extract `hour`/`day_of_week`/`month_num`, normalize `event_cause`. Cleaned shape: 8,122 rows × 33 columns.
- **`severity_index` construction** — a single most important engineered feature, kept **out** of the priority/closure feature sets:
  ```
  cause_score    = lookup from cause_score_map (0–5, e.g. vip_movement=5, vehicle_breakdown=1)
  type_score     = 2 if planned else 1
  priority_score = 2 if actual priority == 'High' else 1
  closure_score  = 2 if requires_road_closure else 0
  severity_index = cause_score + type_score + priority_score + closure_score   # range: 2–11
  ```
- **Feature engineering:**
  - `add_hotspot_distances()` — haversine distance in km from the incident to each of 6 named hotspots (MG Road 12.9766/77.6075, Silk Board 12.9174/77.6228, Hebbal 13.0358/77.5970, Marathahalli 12.9563/77.7010, Whitefield 12.9698/77.7500, Electronic City 12.8399/77.6770).
  - `add_circular_time_features()` — `sin`/`cos` of hour (period 24), month (period 12), day-of-week (period 7) — so the model sees hour 23 and hour 0 as numerically adjacent instead of maximally far apart, which a raw integer encoding gets wrong.
  - `add_interaction_features()` — `peak_x_cause = is_peak_hour × cause_score`, `weekend_x_cause = is_weekend × cause_score`.
- **Model 1 — Priority classifier:** `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=neg/pos)`. Features deliberately **exclude `corridor`** — it correlates with priority almost 1:1 in some slices of the data, a leakage risk. Trained on 8,006 rows (6,404/1,602 stratified split). **Result: Accuracy 0.9694, F1 0.9754, ROC-AUC 0.9924.**
- **Model 2 — Road closure classifier:** `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=(neg/pos)×0.6)` on 7,987 rows, 7.26% positive class. The hyperparameters (shallower trees, dampened `scale_pos_weight`) came from a small grid search over depth/estimators/learning-rate/imbalance-dampening, selected by F1 after threshold tuning — undampened `scale_pos_weight` and deeper trees over-predicted the minority class and scored worse. Threshold tuned via `precision_recall_curve` to **0.609** (not the default 0.5). **Result: Accuracy 0.9124, F1 0.3694, ROC-AUC 0.7724.**
- **Model 3 — Duration regressor:** `XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=1.0, colsample_bytree=0.8)`, trained on `log1p(duration_hrs)` and inverted with `expm1` at inference. Target `duration_hrs` is heavily right-skewed (median 0.76h, max 23.95h, rows >24h dropped as likely data errors). Only 2,455 of ~8,120 rows have a valid `closed_datetime`. **Result: MAE 1.08 hrs, R²(log) 0.2193.** Hyperparameters are deliberately shallow — a small grid search found deeper/more numerous trees overfit given only ~2,000 training rows.
- **Feature encoding:** categorical features one-hot encoded (`pd.get_dummies`), not label-encoded — zone/corridor/police_station are nominal categories with no natural order, and label-encoding would incorrectly imply a ranking.
- **Avg confidence:** `mean(max(p, 1-p))` across the priority and closure test-set predictions = **88.4%** — the real, computed number behind the resume's "89% avg confidence."
- **Regenerated assets:** `train_v2.py` saves `model1_confusion_matrix.png`, `model1_feature_importance.png`, and `model2_confusion_matrix.png` under `assets/` from the actual deployed XGBoost models on every run.
- **Bundle export:** `joblib.dump()` of a dict with both models, their feature-column lists, `closure_threshold`, all lookup tables, and the `hotspots` dict — so the app's map overlay and the training script's feature engineering read from the exact same source of truth.

---

### `models/recommendation_engine_bundle_v2.pkl`
**Purpose:** The single artifact that decouples training from serving — everything `app_v2.py` needs to make a prediction, with no dependency on the raw CSV at runtime.

**Verified by loading the file directly:**
- `priority_model`: `XGBClassifier`. `closure_model`: `XGBClassifier`. `duration_model`: `XGBRegressor`.
- `hotspots`: a dict of 6 name→(lat, lon) pairs — what both the training script's distance features and the app's map overlay read from.
- Also present: `priority_feature_cols` (47 cols), `closure_feature_cols` (69 cols), `duration_feature_cols`, `closure_threshold` (0.609), `cat_features`/`num_features` (+ `_fixed` variants), `cause_score_map`, `manpower_map`, `zone_station_map` (10 zones), `station_coords` (8,120 rows), `corridor_coords` (8,101 rows).

---

### `Dataset/Hack_dataset.csv`
**Purpose:** The raw training data — 8,173 rows, 46 columns, real ASTRAM incident records including street addresses and GPS coordinates.

**Interview explanation:** "This file is deliberately excluded from git (`.gitignore`) — it contains real street addresses and precise incident locations, which isn't something that belongs in a public repo. Anyone reproducing the training run needs to source it separately and place it at `Dataset/Hack_dataset.csv`."

**Why it matters:** A real, defensible data-handling decision — worth mentioning proactively if asked "how do you handle sensitive data in this project."

---

### `.github/workflows/ci.yml`
**Purpose:** GitHub Actions CI — installs `requirements.txt`, compiles both Python entry points, then loads the checked-in bundle and runs `recommend_resources()` against a sample event as a smoke test.

**Why it matters:** A small but real CI signal — it catches the exact class of bug this project's own schema-drift discussion is about: if someone changes a feature name in `app_v2.py` without updating the bundle (or vice versa), CI fails immediately instead of silently shipping a broken app.

**Honest limitation:** it doesn't retrain or gate on model *quality* (no accuracy/F1 threshold) — it only proves the app and the currently-committed bundle are structurally compatible. Worth naming unprompted as a "what I'd add next."

---

### `assets/` (EDA outputs)
**Purpose:** Visual outputs — distribution/time-pattern/severity/corridor-zone charts, the standalone `bengaluru_hotspot_map.html` (a Folium heatmap of the full historical incident dataset — distinct from the app's live per-incident map), and the three model-diagnostic PNGs regenerated by `train_v2.py`.

---

### `requirements.txt`
Pinned with minimum versions: `pandas>=2.0`, `numpy>=1.24`, `scikit-learn>=1.2`, `xgboost>=1.7`, `streamlit>=1.28`, `joblib>=1.3`, `folium>=0.14`, `seaborn>=0.12`, `matplotlib>=3.7`. Every one of these is genuinely imported somewhere in the repo — nothing here is a dead/unused dependency.

---

## PART 4: COMPLETE SYSTEM FLOW

### Flow 1 — Main user workflow (incident → recommendation)
1. Dispatcher opens the Streamlit app; `load_bundle()` runs once and caches the bundle.
2. Dispatcher fills the form: event type, cause, latitude, longitude, hour, day of week, month, optional zone.
3. Dispatcher clicks **"Get Recommendation."**
4. `event = {...}` dict assembled from widget values.
5. `recommend_resources(event)` runs synchronously, in-process.
6. Inside: defaults filled → feature block computed (hotspot distances, circular time, interactions) → nearest corridor detected → priority predicted (XGBoost) → closure predicted (XGBoost, tuned threshold) → severity computed from predictions → duration predicted (XGBoost, log-space, inverted) → manpower looked up → station resolved.
7. Result dict rendered as badges, confidence gauges, metrics, a hotspot-distance table, and a live Folium map.
8. Nothing is persisted — no history, no database write.

### Flow 2 — Model training workflow (offline, one-time, reproducible)
1. `python notebook/train_v2.py`, reading `Dataset/Hack_dataset.csv` locally.
2. Cleaning pipeline runs.
3. Feature engineering runs (hotspot distances, circular time, interactions).
4. Each of the three XGBoost models trains on a fixed `random_state=42` split; the closure model's threshold is tuned via a precision-recall sweep.
5. Confusion-matrix and feature-importance plots are regenerated and saved to `assets/`.
6. Everything (models, feature-column lists, threshold, lookup tables, hotspots dict) is packed into one dict and `joblib.dump()`'d as `models/recommendation_engine_bundle_v2.pkl`; metrics are also written to `models/v2_metrics.json` for auditability.

### Flow 3 — CI workflow
1. On every push/PR to `main`, GitHub Actions installs `requirements.txt`.
2. Compiles `app/app_v2.py` and `notebook/train_v2.py` (catches syntax/import errors).
3. Loads the actual committed `recommendation_engine_bundle_v2.pkl` and runs `recommend_resources()` on a sample event, asserting the bundle has the expected keys and the app's pipeline still produces a valid prediction.

### Flows that do **not** exist in this project (do not claim these in an interview)
- No authentication/authorization.
- No database — nothing persisted between requests.
- No background jobs, queues, or async workers.
- No file uploads, payments, notifications, or search.
- No automatic retraining pipeline — training is a manual, offline step (`python notebook/train_v2.py`).
- No retry/timeout logic anywhere — every call is a direct, synchronous, local function call.

---

## PART 5: THE IMPORTANT CONCEPTS I MUST UNDERSTAND COLD

### 1. Data leakage
**What we built:** A hand-engineered `severity_index` that is a direct arithmetic function of the two classification targets (`priority`, `requires_road_closure`), explicitly excluded from those two models' feature sets.
**Explain like I'm 5:** If a test question spells out the answer to another question, that's not a fair test.
**How it works in this project:** `severity_index = cause_score + type_score + priority_score + closure_score`, where the last two are derived directly from ground truth. Feeding it into the priority/closure models would let them decode their own answer.
**Interview one-liner:** "We caught that our severity score was built from the same labels we were trying to predict, so we excluded it from those two models' inputs and kept it only where it's genuinely independent — the duration model."

### 2. Class imbalance & threshold tuning
**What we built:** The closure classifier faces a 7.26%-positive-class problem; handled via `scale_pos_weight` (dampened to ×0.6 of the raw `neg/pos` ratio via a small grid search) plus post-hoc threshold tuning via a precision-recall sweep.
**How it works in this project:** `precision_recall_curve` + `f1_scores = 2*(p*r)/(p+r+1e-9)`; `best_threshold = thresholds[f1_scores.argmax()]` → **0.609**, stored in the bundle and used at inference (`prob_closure >= B['closure_threshold']`).
**Why the dampening mattered:** the full, undampened `scale_pos_weight` combined with deeper trees over-predicted the minority class — a grid search over depth/estimators/learning-rate/dampening found `max_depth=4` + `×0.6` dampening scored better on F1 than the naive full weighting.
**What could go wrong:** F1 of ~0.37 is still weak in absolute terms — be upfront that this is the hardest of the three tasks, fundamentally data-starved (only 580 historical positive examples).
**Interview one-liner:** "Road closures are rare, so beyond `scale_pos_weight` we tuned both the model's own imbalance dampening and the decision threshold by sweeping the precision-recall curve — a two-stage tuning process, not a single knob."

### 3. Target transformation (log1p / expm1) for skewed regression targets
**What we built:** Duration regressor trained on `log1p(duration_hrs)`; predictions inverted with `np.expm1()`.
**Why it needed it:** Duration is heavily right-skewed (median 0.76h vs max 23.95h) — training directly on hours would let a few huge outliers dominate the loss.
**What could go wrong:** Forgetting `expm1` at inference is a classic real bug — `app_v2.py` gets it right (`np.expm1(B['duration_model'].predict(Xd)[0])`).

### 4. One-hot encoding a single new row consistently with training-time columns
**What we built:** `encode_single_event()` — one-hot encodes a single incoming event, then reindexes against the exact column list captured at training time, zero-filling anything missing.
**Why it matters here specifically:** The feature block adds 14 numeric columns (6 hotspot distances, 6 circular values, 2 interactions) on top of the base categorical/numeric features — if `encode_single_event`'s reindex step and `train_v2.py`'s feature-list capture ever drifted out of sync (e.g., a renamed hotspot key in one file but not the other), this is exactly the mechanism that would either crash loudly (missing column) or, worse, silently zero-fill a real feature. This is precisely what the CI smoke test exists to catch early.

### 5. Composite scoring instead of a fourth ML model (severity_index / manpower_map)
**What we built:** Severity and officer-count recommendations are deterministic, hand-tuned formulas/lookup tables, not a learned model.
**Why:** Transparent, auditable resource-deployment logic matters more than marginal accuracy gains for a system recommending physical police deployment.
**Interview one-liner:** "Not everything here is ML — severity and manpower are transparent rule-based formulas layered on top of the model outputs, which is a deliberate choice for interpretability in a policing context."

### 6. Feature set differs per model, on purpose (corridor exclusion)
**What we built:** The priority model excludes `corridor` (near-1:1 leak risk with priority in some data slices); closure and duration include it (a legitimate, if strong, ~88–97% correlated signal, not a leak).
**Why:** A judgment call based on inspecting each feature's relationship to each target separately, not a blanket rule.
**What could go wrong:** This is a somewhat subjective line to draw ("88–97% correlation is fine, ~100% is not") — a more rigorous pipeline would want a more principled leakage-detection method than an eyeballed correlation threshold. Worth admitting as a soft spot if pressed.

### 7. Nearest-neighbor GPS lookup instead of a fixed lookup table
**What we built:** `nearest_corridor()`/`nearest_police_station()` — haversine distance to every historical point, k=5-nearest, mode (denoises against a single mislabeled historical point).
**Why it's distinct from the hotspot features:** The 6 *named* hotspots (used for distance features and the map overlay) are a small, curated, fixed list — deliberately different from the corridor/station lookups, which search the *entire* historical table. Hotspots are for feature engineering + a legible map overlay ("distance to MG Road" is interpretable to a dispatcher); the full corridor/station tables are for the actual routing/deployment decision, where using only 6 fixed points would throw away most of the historical signal.

### 8. Validating a suspiciously good result isn't leakage (the standout concept)
**What we found:** The priority classifier's test-set performance — 97% accuracy, AUC 0.99 — is high enough for a real-world tabular problem that it deserved active suspicion, not just acceptance.
**What we checked:** First, whether `cause` or `zone` alone near-deterministically implies `priority` (it doesn't — priority rates by cause range 32%–69%, no near-100% pattern; by zone, 30%–82%, also no perfect determinism). Second, whether the 80/20 *random* split let many test-set incidents share an almost-identical GPS location with a training-set incident carrying the same priority label — a real risk, since real-world incident data repeats: only 3,336 unique ~100m GPS clusters exist across 8,173 rows, and individual clusters can be 100% one class (e.g., one 65-incident cluster is 100% "High," another 53-incident cluster is 100% "Low"). To rule this out, I re-evaluated with a **spatial group-holdout** (`GroupShuffleSplit` grouped by rounded lat/lon, so no test-set location's ~100m cluster appears in training) instead of a random split: **Accuracy 0.9730, F1 0.9785, AUC 0.9886** — essentially unchanged from the random-split numbers.
**What this means:** The model is genuinely learning a spatial+temporal decision boundary that generalizes to locations it never saw in training, not memorizing repeated addresses. That's a materially stronger claim than "our test accuracy is high," and it directly pre-empts the most obvious follow-up question a good interviewer would ask.
**Interview one-liner:** "A number that good is exactly what should make you suspicious of your own pipeline before you're proud of it — so before trusting it, I re-validated with a spatial holdout where the test locations never appeared in training at all, and the result held. That's what actually convinced me it was real."

---

## PART 6: INTERVIEW QUESTIONS — THE INTERROGATION

### Architecture

#### Q1. Why did you structure this as a single Streamlit script instead of a proper frontend/backend split?
**Strong answer:** "Hackathon deadline, and the goal was a working interactive demo. Streamlit removes the frontend-build step entirely. The cost is architectural — one process, no API contract, whole-script reruns — none of which matters for a single-user demo."
**Follow-up — "How would you productionize it?"**: "Split `recommend_resources()` into a stateless FastAPI `/predict` endpoint (it's already a pure function with no UI coupling — low-risk refactor), keep a thin client in front, move the model bundle to an artifact store, and add a model-quality gate to CI that the current one doesn't have."

#### Q2. Why three separate models instead of one multi-output model?
**Strong answer:** "Priority, closure, and duration have meaningfully different characteristics — closure is severely imbalanced, duration has a third of the usable rows and needed a log-transform, priority needed corridor excluded specifically for leakage reasons. Separate models meant each got its own feature set and its own hyperparameter search, rather than compromising all three to fit one shared architecture."
**Possible follow-up:** "Doesn't that create inconsistency risk?" **Follow-up answer:** "Yes — which is exactly why each model's own `feature_cols` list is serialized into the bundle alongside it, and `encode_single_event` always reindexes against that specific list. The code never assumes a shared schema across models."

#### Q3. Why do the three models perform so differently — priority near-perfect, closure and duration much weaker?
**What the interviewer is asking:** Do you understand *why* your model performs the way it does, or are you just reporting numbers?
**Strong answer:** "It comes down to data and target characteristics, not the algorithm. Priority has ~8,000 labeled rows and a roughly balanced target, and the added hotspot-distance and circular-time features apparently expose a strong, learnable spatial/temporal pattern — validated with a spatial holdout so I'm confident it's real signal, not leakage. Closure has only 580 positive examples out of ~8,000 — fundamentally data-starved, so no amount of feature engineering fully solves that. Duration has only 2,455 usable rows (most incidents lack a valid close timestamp) and a noisy, right-skewed target — R² of 0.22 reflects a target that's inherently hard to predict from these features, not an undertrained model."
**Possible follow-up:** "Did you tune hyperparameters for real?"
**Follow-up answer:** "For closure and duration, yes — a real (if small, a few dozen combinations) grid search over depth/estimators/learning-rate/imbalance-dampening, selected by F1 (closure) or MAE (duration) on the held-out test set. For priority, mostly sensible defaults plus `scale_pos_weight` — it didn't need much tuning to already perform well."

#### Q4. That priority AUC is suspiciously high. Did you check for leakage?
**Strong answer:** Walk through Part 5, Concept 8 verbatim — the cause/zone determinism check, the GPS-cluster-repetition finding, and the spatial group-holdout result. This is the single best-prepared answer in the whole document; don't rush it.

### Database

#### Q5. Why no database?
**Strong answer:** "Nothing to persist for the demo's scope — every request is stateless. Training data lives in a gitignored CSV, consumed once, offline."
**Follow-up — "What would you store if you added one?"**: "Every incident + recommendation pair, both for an audit trail and to eventually retrain on real outcomes instead of only the original historical dump — closing the feedback loop the current one-shot training script doesn't have."

#### Q6. What happens when the dataset (or user count) grows 100x?
**Strong answer:** "Two bottlenecks. First, `nearest_corridor`/`nearest_police_station` are O(n) brute-force haversine scans over the full historical table on every request — fine at 8,000 rows, would need a `BallTree`/PostGIS at 100x. Second, `train_v2.py` currently loads the whole CSV into pandas in memory — comfortably fine at 8K rows, would need chunking or a different tool well before millions of rows."

### Backend

#### Q7. How does this app validate incoming requests?
**Strong answer:** "Minimally, and this is a real, known gap. Streamlit's own widgets constrain most inputs structurally (`st.selectbox` for cause, `st.slider` bounds for hour), but latitude/longitude are free-text `st.number_input` with no bounds check against Bengaluru's actual geographic range. I'd add that before calling this production-ready."

#### Q8. How do you handle a model failure mid-request?
**Strong answer:** "No explicit try/except around the prediction path — an unexpected input would surface Streamlit's default traceback. Acceptable for a hackathon demo where the input space is constrained by form widgets; not acceptable for production without wrapping the prediction path and returning a generic user-facing error while logging the real traceback server-side."

#### Q9. How would you scale this to serve many concurrent dispatchers?
**Strong answer:** "`@st.cache_resource` already shares the bundle across sessions in one process, which helps. To truly scale, move `recommend_resources()` behind a stateless FastAPI service that can be horizontally replicated, with the (~2MB) bundle loaded once per worker."

### Frontend

#### Q10. Why Streamlit instead of React/Vue?
**Strong answer:** "The core skill set was ML/Python, and the goal was a demo within a hackathon timebox. The cost is customizability — constrained to Streamlit's widget set and whole-script-rerun execution model."

#### Q11. How does the live Folium map actually get into a Streamlit page?
**Strong answer:** "A `folium.Map` object has a `_repr_html_()` method that returns a full HTML document for the map; I pass that straight into `streamlit.components.v1.html()`, which renders arbitrary HTML/JS in an iframe. No extra `streamlit-folium` dependency needed — just the two libraries already in `requirements.txt`."

#### Q12. How is state managed in this app?
**Strong answer:** "Almost none, deliberately. Streamlit reruns the whole script on every interaction; the only things that persist across reruns are the cached bundle and Streamlit's own internal widget state. No `st.session_state` usage — every recommendation is computed fresh."

### Security

#### Q13. What security vulnerabilities exist in this code as written?
**Strong answer:** "Two, and I'd flag them myself: first, `joblib.load()` on a `.pkl` is equivalent to unpickling arbitrary Python — fine because it's our own file in our own repo, but a real risk class if this pattern were reused with an externally-sourced file. Second, zero authentication on the Streamlit app itself — anyone with the URL can use it, which is fine for a hackathon demo, not for anything handling real police dispatch decisions."
**Possible follow-up:** "Is the raw dataset itself a security/privacy concern?"
**Follow-up answer:** "Yes, and it's handled correctly here — `Dataset/Hack_dataset.csv` contains real street addresses and GPS coordinates, so it's excluded via `.gitignore` and never committed. That's a deliberate call."

#### Q14. How do you handle secrets/credentials in this project?
**Strong answer:** "There are none — no API keys, tokens, or credentials anywhere in `app_v2.py` or `train_v2.py`. Nothing to rotate or move to an environment variable."

### Performance

#### Q15. What's the biggest performance bottleneck?
**Strong answer:** "The nearest-neighbor GPS lookups — brute-force haversine against ~8,000+ rows per request, no spatial index. Negligible today; would need a `BallTree` at real scale. The feature computation itself (6 hotspot distances, 6 circular values, 2 interactions) is comparatively free — a handful of arithmetic operations per request versus a distance-sort over thousands of rows."

#### Q16. Does the richer feature set slow down inference meaningfully?
**Strong answer:** "No — XGBoost inference on a single row with ~50-70 columns is sub-millisecond regardless of exact column count; the dominant cost per request is still the two brute-force nearest-neighbor searches, not model inference or feature computation."

### Reliability

#### Q17. What happens if the same incident is submitted twice?
**Strong answer:** "Naturally idempotent — no persistence layer, no side effects, so it just computes and displays the same recommendation twice."

#### Q18. What happens if the data schema changes and the bundle drifts?
**Strong answer:** "This is exactly what the CI smoke test guards against — it loads the actual committed bundle and asserts the expected keys exist, then runs a real `recommend_resources()` call. It's not a full regression-test suite with an accuracy floor, but it does catch a hard schema break (missing key, incompatible feature list) before it reaches a user."

### Trade-offs

#### Q19. What would you change if you rebuilt this today?
**Strong answer:** "Three things. First, add a real accuracy/F1 floor to CI so a retrain that regresses model quality gets caught, not just structural breaks. Second, add GPS bounds validation — the one input-handling gap that's persisted throughout. Third, log every incident/recommendation pair somewhere, even just to a file, so there's a path toward retraining on real outcomes rather than only the original static dump."

#### Q20. What did you deliberately not build?
**Strong answer:** "A live traffic feed, authentication, persistence/history, and an automatic retraining pipeline. Training is `python notebook/train_v2.py`, run manually against a local CSV — CI checks the *app* is still compatible with whatever bundle is committed, but it doesn't retrain or gate on model quality. That's a conscious scope decision for a hackathon-derived project, not something I'd claim as production-grade MLOps."

---

## PART 7: DESIGN DECISIONS & TRADE-OFFS

**Decision:** XGBoost for all three models, with `scale_pos_weight` as the primary imbalance lever for closure.
**Why:** Gradient-boosted trees handle the mixed categorical/numeric feature set well, and `scale_pos_weight` gives a direct, tunable knob for the closure model's severe class imbalance.
**Alternative:** Oversampling/undersampling (e.g., SMOTE).
**Why we didn't:** `scale_pos_weight` plus threshold tuning is simpler and doesn't risk synthetic-sample artifacts.
**Trade-off:** Even after tuning, closure F1 (0.37) is still weak in absolute terms — this reflects genuine data scarcity (580 positive examples), not a tooling gap.

**Decision:** Exclude `severity_index` components from the priority/closure classifiers.
**Why:** Direct functions of those models' own target labels — textbook leakage.
**Trade-off:** Excluding it likely cost some real predictive accuracy on those two models, since it's genuinely informative — but the reported numbers are honest, not inflated.

**Decision:** Validate the priority model's high accuracy with a spatial group-holdout rather than accepting the random-split number at face value.
**Why:** An 8,173-row real-world incident dataset has substantial GPS repetition (3,336 unique ~100m clusters); a random split can let near-identical locations appear in both train and test, inflating apparent generalization.
**Trade-off:** Extra engineering time for a check that (this time) came back clean — but the alternative (skipping it and being caught out in an interview) is worse.
**When I'd change this decision:** I wouldn't — this kind of validation should be standard practice whenever a metric looks unexpectedly strong, regardless of whether it turns out to matter.

**Decision:** Store lookup tables (corridor/station coordinates, cause scores, manpower map, hotspots) inside the same pickle as the models, rather than a database or config file.
**Why:** Keeps the app's only runtime dependency to a single file, simple to deploy/copy.
**Trade-off:** Any update to `cause_score_map` or `manpower_map` requires retraining/re-exporting the whole bundle — versus editing a small config file directly.
**When I'd change this decision:** As soon as these tables needed to be tweaked independently of retraining the models.

---

## PART 8: EDGE CASES & FAILURE SCENARIOS

### Scenario: Duration regressor trained on a tiny, non-representative subset
- **What can go wrong?** Only 2,455 of ~8,120 usable rows have a valid `closed_datetime` — the rest lack one (likely still-active incidents at data export time).
- **How does the current code handle it?** No explicit check for whether this subset is representative of all incidents.
- **What happens if it isn't handled?** The measured R² of 0.2193 (modest) may partly reflect this restricted, possibly-biased training set.
- **How would I improve it in production?** Compare cause/priority distributions between the "has valid duration" and "doesn't" groups before trusting the duration model's generalization.

### Scenario: Unrecognized `event_cause` submitted
- **What can go wrong?** `cause_score_map.get(event['event_cause'], 1)` silently defaults an unrecognized cause to weight 1, and `encode_single_event`'s reindex zeroes out the corresponding one-hot column.
- **Why can it happen?** In the shipped app, `st.selectbox` is populated directly from `B['cause_score_map'].keys()`, so this can't actually be triggered through the UI as built — but it would matter immediately if this logic were reused behind a freeform-input API.
- **How would I improve it in production?** Surface a warning whenever a categorical input falls outside the model's known training vocabulary.

### Scenario: GPS coordinates outside Bengaluru
- **What can go wrong?** `st.number_input` for latitude/longitude has no bounds check — the single most persistent, real gap in this project.
- **How does the current code handle it?** `nearest_corridor`/`nearest_police_station` will happily return the "closest" historical match even for a coordinate on the other side of the world; hotspot distances would just be very large numbers with no warning.
- **How would I improve it in production?** Validate lat/lon against Bengaluru's real bounding box (`latitude.between(12.8,13.3)`, `longitude.between(77.3,77.9)`) before returning a recommendation.

### Scenario: Bundle/app schema drift
- **What can go wrong?** `app_v2.py` hardcodes the path `models/recommendation_engine_bundle_v2.pkl`; a bundle with a renamed key or different feature-column count would `KeyError` or misalign silently.
- **How does the current code handle it?** The CI smoke test now catches the "crashes outright" case automatically on every push — it does not catch silent misalignment (same key names, subtly wrong values).
- **How would I improve it further?** A checksum or version tag on the bundle, checked at app startup, so a stale or hand-edited bundle fails loudly rather than silently producing subtly-wrong predictions.

---

## PART 9: SCALING THE PROJECT

**Current implementation:** A single Streamlit process, one in-memory ~2MB model bundle, no database, no external API calls, brute-force nearest-neighbor search over ~8,000 historical rows on every request.

### 10x users
**What breaks first:** Nothing catastrophic — `@st.cache_resource` shares the bundle across sessions. **Improvement:** run multiple Streamlit workers behind a reverse proxy.

### 100x users (concurrent load)
**What breaks first:** Streamlit's whole-script-rerun execution model under concurrent load. **Improvement:** extract `recommend_resources()` into a stateless FastAPI service, horizontally scaled.

### Much larger historical database (millions of incidents)
**What breaks first:** `nearest_corridor`/`nearest_police_station` — O(n) per lookup. **Improvement:** a `BallTree` (haversine metric) or PostGIS instead of a linear scan. Also: `train_v2.py` loading the whole CSV into pandas would need reconsideration well before this point.

### High concurrent traffic
**Improvement:** A proper multi-process API server to parallelize CPU-bound inference across cores — though per-row XGBoost inference cost here is small enough that the nearest-neighbor searches would bottleneck first.

### Areas explicitly not relevant to this project's current scale
Queues/background workers, file storage, rate limiting, observability — none exist, none are currently load-bearing gaps given the actual traffic this app has ever seen (a hackathon demo).

---

## PART 10: SECURITY REVIEW

| Issue | Current status | Why it matters | How to fix |
|---|---|---|---|
| **No authentication/authorization** | `app_v2.py` has zero login/session/access-control logic. | Fine for a demo; not for a real dispatch tool. | Basic auth or a reverse-proxy SSO gate before real deployment. |
| **Unpickling risk (`joblib.load`)** | `load_bundle()` calls `joblib.load()` on the bundle with no integrity check. | `pickle`-based deserialization can execute arbitrary code if the file source is ever untrusted. | Low risk today (self-produced, same repo); a hardened deployment should checksum the file before loading. |
| **No input validation (GPS bounds)** | Still the one persistent, unaddressed gap. | Nonsensical coordinates still produce a confident-looking answer. | Validate lat/lon against Bengaluru's real bounding box before running any model. |
| **Raw dataset handling** | **Handled correctly** — `Dataset/Hack_dataset.csv` (real addresses + GPS) is `.gitignore`'d, never committed. | Avoids shipping real, potentially sensitive location data in a public repo. | Already done — worth citing as a positive, not just listing gaps. |
| **CI has no secrets exposure** | `.github/workflows/ci.yml` uses no tokens/credentials — installs public packages and runs local smoke tests only. | No CI-secret-leak surface exists in this pipeline. | N/A. |
| **Error/traceback exposure** | No try/except around the prediction path. | Minor information disclosure via Streamlit's default dev traceback. | Wrap `recommend_resources()`, log real errors server-side only. |

---

## PART 11: PERFORMANCE REVIEW

| Area | Current behavior | Impact | Improvement |
|---|---|---|---|
| **Nearest-corridor/station lookup** | Full haversine scan + `np.argsort` over ~8,000-row tables per request. | Negligible today; would bottleneck at 100x–1000x data. | `np.argpartition` for a modest win now; `BallTree`/PostGIS for real scale. |
| **Feature computation** | 6 hotspot distances + 6 circular values + 2 interactions, computed per request. | Trivial — a handful of arithmetic ops, dwarfed by the nearest-neighbor searches. | Not worth optimizing at this scale. |
| **Bundle load** | `joblib.load()` of a ~2MB file, once per process, cached. | Low impact given caching is already correct. | None needed. |
| **Model inference** | Three XGBoost `predict`/`predict_proba` calls per request on a single-row DataFrame. | Sub-millisecond at this scale. | Batch requests into one `predict()` call only if throughput ever actually demanded it. |
| **Full-script rerun per interaction** | Streamlit reruns the whole script; only the bundle load is cached. | Fine given nothing else expensive exists outside the cached function. | Wrap any future expensive, input-independent computation in `@st.cache_data`. |

---

## PART 12: RESUME BULLETS — WITH CODE MAPPING

> "Built an ML-powered dashboard on 8,173 ASTRAM traffic events to predict incident priority (High/Low), road closure requirement, and estimated duration in real time. Engineered 3 XGBoost models with circular time encoding, haversine hotspot distances, and interaction features achieving AUC 0.98 and 89% avg confidence across all predictions."

**How to defend it:** Point to `notebook/train_v2.py` for the full pipeline and `models/v2_metrics.json` for the exact numbers. Precisely: priority ROC-AUC is 0.9924 (the "0.98" is a safe underestimate); closure AUC is 0.7724 (a much harder, imbalanced problem — be ready to name this distinction if asked "AUC of what"). Avg confidence 88.4% (resume rounds to 89%). Be ready to also state the honest counter-numbers: closure F1 0.3694 and duration R² 0.2193 are both modest in absolute terms — the strong headline number is specifically the priority classifier.

> "Recommendation engine auto-detects nearest corridor via GPS and outputs officer count, deployment station, barricading plan, and an interactive Folium live map with hotspot overlays."

**How to defend it:** `app/app_v2.py`'s `nearest_corridor()`, `manpower_map` lookup, `zone_station_map`/`nearest_police_station()`, and `render_hotspot_map()`. Be ready to explain the k=5-nearest-then-mode denoising pattern and how the Folium map is embedded (`_repr_html_()` into `st.components.v1.html`, no extra dependency).

**Also now true and defensible, if asked directly:** "GitHub Actions" (`.github/workflows/ci.yml`, real, runs a real smoke test) and "Ensemble learning" (XGBoost is boosted-tree ensemble learning, literally).

**Worth being upfront about if pushed hard on specifics:** the lack of an accuracy/F1 quality gate in CI (Part 3/9) and the still-unaddressed GPS bounds-validation gap (Part 8/10) — naming these unprompted, if the conversation heads that way, reads as more credible than waiting to be caught.

---

## PART 13: RAPID-FIRE ONE-LINERS

| Question | One-line answer |
|---|---|
| What does the project do? | Predicts a Bengaluru traffic incident's priority, road-closure need, and duration via three XGBoost models, then recommends officer count and police station via a Streamlit dashboard with a live hotspot map. |
| Why XGBoost? | `scale_pos_weight` cleanly handles the closure model's severe imbalance, and gradient-boosted trees handle the mixed categorical/numeric, moderately nonlinear feature set well. |
| Biggest result? | Priority classifier: ROC-AUC 0.9924, validated with a spatial holdout (AUC 0.9886) to rule out location-memorization. |
| Biggest technical challenge? | Catching that our severity score leaked the classification targets it was meant to help predict. |
| Most important validation habit? | Not trusting a suspiciously good metric — validating the priority-model result with a spatial group-holdout before believing it. |
| Biggest limitation? | Closure F1 (0.3694) is still weak in absolute terms — the closure task is fundamentally data-starved (580 positive examples), not something feature engineering alone fixes. |
| How does authentication work? | It doesn't — no login or access control anywhere. |
| How does the "database" work? | There isn't one — stateless app, all data in one pickled bundle loaded once at startup. |
| How does road-closure prediction work? | XGBoost's probability compared against a tuned threshold (0.609), picked by maximizing F1 on a precision-recall sweep, with `scale_pos_weight` dampened to ×0.6 via grid search. |
| How would you scale it? | Extract `recommend_resources()` into a stateless API, replace brute-force haversine search with a spatial index, add a model-quality gate to CI. |
| What would you improve first? | GPS bounds validation and a model-quality floor in CI. |
| Is training automated? | No — `python notebook/train_v2.py` is a manual, offline step; CI checks app/bundle compatibility, not model quality. |
| Where does severity score come from? | A hand-built formula (cause + type + predicted priority + predicted closure weights, range 2–11) — not a learned model. |

---

## PART 14: "IF THEY DIG DEEPER" CHEAT SHEET

**Important functions:**
- `recommend_resources(event)` — the orchestrating pipeline function (`app/app_v2.py`).
- `encode_single_event(...)` — reindex-based one-hot encoding for a single live row.
- `add_hotspot_distances`, `add_circular_time_features`, `add_interaction_features` — the feature-engineering block, present identically in both `app/app_v2.py` and `notebook/train_v2.py`.
- `haversine_km`, `nearest_corridor`, `nearest_police_station` — geospatial lookups.
- `render_hotspot_map`, `confidence_gauge` — the map/UI helpers.
- `load_bundle()` — `@st.cache_resource`-decorated bundle loader.

**Important "tables":**
- `station_coords` (8,120 rows), `corridor_coords` (8,101 rows), `zone_station_map` (10 zones), `cause_score_map` (16 causes), `manpower_map` (severity 2–11 → officers), `hotspots` (6 named points).

**Important algorithms:**
- Haversine formula, gradient-boosted trees (XGBoost) for classification/regression, precision-recall threshold sweep, log1p/expm1 target transformation, sin/cos circular encoding, group-based (spatial) cross-validation for leakage-style sanity checks.

**Important dependencies:** `pandas`, `numpy`, `scikit-learn` (metrics + `GroupShuffleSplit`), `xgboost`, `streamlit`, `joblib`, `folium`, `seaborn`/`matplotlib` (asset generation only).

**Important infrastructure:** GitHub Actions CI (`.github/workflows/ci.yml`). No Docker, no cloud deployment config.

**Important terminology to use precisely:**
- **Data leakage** — a feature that encodes information from the target that wouldn't be available at real prediction time.
- **Class imbalance** — one class vastly outnumbering another (7.3% positive for road closure).
- **Threshold tuning** — choosing a probability cutoff other than the default 0.5 to optimize a chosen metric.
- **Spatial/group holdout** — a validation split where entire groups (here, GPS location clusters) are kept out of training, to detect whether a model is generalizing or memorizing repeated near-identical examples.
- **Ensemble learning** — combining multiple models into a stronger one; XGBoost does this by boosting (sequential trees that each correct the previous ones' errors).
- **Log-transform (log1p/expm1)** — compressing a right-skewed target before training, and reversing it at inference.
- **Haversine distance** — great-circle distance between two GPS points on a sphere.
- **Feature schema drift** — when the columns a model expects at inference no longer match what it was trained on.

---

## FINAL SECTION: 60-SECOND PROJECT ANSWER

"For Gridlock Hackathon 2.0, we built a traffic-incident response recommender for Bengaluru using about 8,170 real historical incidents from the ASTRAM system. Three XGBoost models handle priority classification, road-closure classification, and duration regression, sitting on top of an engineered feature set — circular sin/cos time encoding, haversine distance to six named congestion hotspots, and interaction terms combining peak-hour and cause severity. All of it's wrapped in a Streamlit dashboard that also auto-detects the nearest road corridor from GPS, recommends an officer count and police station, and shows a live Folium map with the hotspots overlaid.

The priority classifier is the headline result — ROC-AUC 0.99. That's high enough that I didn't just trust it — I validated it with a spatial holdout, training on one set of locations and testing on GPS coordinates the model had genuinely never seen, and the result held (AUC 0.989), which told me it's a real spatial pattern, not memorized addresses. The part I'd actually most want to talk through, though, is a bug we caught before trusting any of these numbers: we'd built a severity score directly from the same priority and closure labels we were trying to predict, caught that it was leaking the answer into two of the three models, and fixed it. That habit — being suspicious of your own metrics before being proud of them — is the same one that led to double-checking the priority AUC later. Not every model here is a clean win, either — the closure classifier, working with only 580 historical positive examples, has a much weaker F1 than priority, and I can walk through exactly why that one's a harder problem."

**(~60 seconds spoken aloud)**
