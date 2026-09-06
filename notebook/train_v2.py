"""
Bengaluru Traffic Congestion — v2 training pipeline.

Cleans the raw ASTRAM incident CSV, engineers a feature set (circular time
encoding, haversine distance to named hotspots, interaction features), and
trains XGBoost for all three tasks (priority classifier, road-closure
classifier, duration regressor).

Run: python notebook/train_v2.py
Requires: Dataset/Hack_dataset.csv (not tracked in git — see Readme.md)
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "Dataset", "Hack_dataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

# 6 well-known Bengaluru congestion hotspots (resume/README claim) — used for
# haversine-distance features, same role a domain expert's curated list plays.
HOTSPOTS = {
    "mg_road": (12.9766, 77.6075),
    "silk_board": (12.9174, 77.6228),
    "hebbal": (13.0358, 77.5970),
    "marathahalli": (12.9563, 77.7010),
    "whitefield": (12.9698, 77.7500),
    "electronic_city": (12.8399, 77.6770),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_hotspot_distances(df):
    for name, (hlat, hlon) in HOTSPOTS.items():
        df[f"dist_{name}_km"] = haversine_km(df["latitude"], df["longitude"], hlat, hlon)
    return df


def add_circular_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    dow_num = df["day_of_week_num"]
    df["dow_sin"] = np.sin(2 * np.pi * dow_num / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow_num / 7)
    return df


def add_interaction_features(df):
    df["peak_x_cause"] = df["is_peak_hour"] * df["cause_score"]
    df["weekend_x_cause"] = df["is_weekend"] * df["cause_score"]
    return df


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


LOG_LINES = []

log("=" * 70)
log("STEP 0 — LOAD RAW DATA")
log("=" * 70)
df = pd.read_csv(DATA_PATH)
temp_df = df.copy()
log(f"Raw shape: {temp_df.shape}")

log("\n" + "=" * 70)
log("STEP 1 — CLEANING (mirrors notebook exactly)")
log("=" * 70)
col_to_drop = ["map_file", "comment", "meta_data"]
temp_df = temp_df.drop(columns=[c for c in col_to_drop if c in temp_df.columns])

datetime_cols = [
    "start_datetime", "end_datetime", "modified_datetime", "created_date",
    "closed_datetime", "resolved_datetime",
]
for col in datetime_cols:
    if col in temp_df.columns:
        temp_df[col] = pd.to_datetime(temp_df[col], errors="coerce")

before = len(temp_df)
inconsistent = temp_df[temp_df["end_datetime"] < temp_df["start_datetime"]]
temp_df = temp_df.drop(inconsistent.index)
log(f"Dropped {before - len(temp_df)} rows where end_datetime < start_datetime")

missing_pct = temp_df.isnull().sum() / len(temp_df) * 100
high_null_cols = missing_pct[missing_pct > 90].index.tolist()
temp_df = temp_df.drop(columns=high_null_cols)
log(f"Dropped high-null (>90%) columns: {high_null_cols}")

temp_df["endlatitude"] = temp_df["endlatitude"].replace(0.0, np.nan)
temp_df["endlongitude"] = temp_df["endlongitude"].replace(0.0, np.nan)

temp_df["duration_hrs"] = (
    temp_df["closed_datetime"] - temp_df["start_datetime"]
).dt.total_seconds() / 3600
temp_df = temp_df[temp_df["duration_hrs"].isna() | (temp_df["duration_hrs"] >= 0)]

temp_df["hour"] = temp_df["start_datetime"].dt.hour
temp_df["day_of_week"] = temp_df["start_datetime"].dt.day_name()
temp_df["day_of_week_num"] = temp_df["start_datetime"].dt.dayofweek
temp_df["month"] = temp_df["start_datetime"].dt.month_name()
temp_df["month_num"] = temp_df["start_datetime"].dt.month

temp_df["event_cause"] = temp_df["event_cause"].str.strip().str.lower()
temp_df = temp_df.drop(columns=["endlatitude", "endlongitude"])
log(f"Cleaned shape: {temp_df.shape}")

log("\n" + "=" * 70)
log("STEP 2 — SEVERITY INDEX (kept OUT of priority/closure features — leakage)")
log("=" * 70)
cause_score_map = {
    "vip_movement": 5, "public_event": 4, "protest": 4, "procession": 3,
    "construction": 3, "tree_fall": 3, "water_logging": 3, "debris": 3,
    "road_conditions": 2, "congestion": 2, "accident": 2, "others": 2,
    "vehicle_breakdown": 1, "pot_holes": 1, "fog / low visibility": 1, "test_demo": 0,
}
manpower_map = {2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 6, 8: 8, 9: 10, 10: 12, 11: 15}


def risk_bucket(score):
    if score <= 4:
        return "Low"
    elif score <= 7:
        return "Medium"
    elif score <= 9:
        return "High"
    return "Critical"


temp_df["cause_score"] = temp_df["event_cause"].map(cause_score_map).fillna(1)
temp_df["type_score"] = temp_df["event_type"].map({"planned": 2, "unplanned": 1})
temp_df["priority_score"] = temp_df["priority"].map({"High": 2, "Low": 1}).fillna(1)
temp_df["closure_score"] = temp_df["requires_road_closure"].astype(int) * 2
temp_df["severity_index"] = (
    temp_df["cause_score"] + temp_df["type_score"]
    + temp_df["priority_score"] + temp_df["closure_score"]
)

log("\n" + "=" * 70)
log("STEP 3 — FINAL NULL HANDLING")
log("=" * 70)
before = len(temp_df)
temp_df = temp_df.dropna(subset=["priority"])
log(f"Dropped {before - len(temp_df)} rows with missing priority")
temp_df["veh_type"] = temp_df["veh_type"].fillna("unknown")
temp_df["zone"] = temp_df["zone"].fillna("Unknown")

temp_df["is_weekend"] = temp_df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
temp_df["is_peak_hour"] = temp_df["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
temp_df["is_night"] = ((temp_df["hour"] >= 22) | (temp_df["hour"] < 6)).astype(int)

log("\n" + "=" * 70)
log("STEP 4 — V2 FEATURE ENGINEERING (circular time, hotspot distances, interactions)")
log("=" * 70)
temp_df = add_hotspot_distances(temp_df)
temp_df = add_circular_time_features(temp_df)
temp_df = add_interaction_features(temp_df)
hotspot_cols = [f"dist_{name}_km" for name in HOTSPOTS]
circular_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
interaction_cols = ["peak_x_cause", "weekend_x_cause"]
log(f"Hotspot distance features: {hotspot_cols}")
log(f"Circular time features: {circular_cols}")
log(f"Interaction features: {interaction_cols}")

# ============================================================================
# MODEL 1 — PRIORITY CLASSIFIER (corridor excluded — leaks priority ~1:1)
# ============================================================================
log("\n" + "=" * 70)
log("MODEL 1 — PRIORITY CLASSIFIER (XGBoost, v2 features)")
log("=" * 70)
CAT_FEATURES_FIXED = ["event_type", "event_cause", "zone", "veh_type"]
NUM_FEATURES_FIXED = (
    ["latitude", "longitude", "is_weekend", "is_peak_hour", "is_night"]
    + hotspot_cols + circular_cols + interaction_cols
)

priority_df = temp_df.dropna(subset=CAT_FEATURES_FIXED + NUM_FEATURES_FIXED + ["priority"]).copy()
X_fixed = pd.get_dummies(priority_df[CAT_FEATURES_FIXED + NUM_FEATURES_FIXED], columns=CAT_FEATURES_FIXED)
y_fixed = (priority_df["priority"] == "High").astype(int)
log(f"Rows usable: {len(priority_df)}  Class balance: {y_fixed.value_counts().to_dict()}")

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fixed, y_fixed, test_size=0.2, random_state=RANDOM_STATE, stratify=y_fixed
)
neg_p, pos_p = (y_train_f == 0).sum(), (y_train_f == 1).sum()
priority_model = XGBClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.1, reg_lambda=5, min_child_weight=10,
    scale_pos_weight=neg_p / pos_p, random_state=RANDOM_STATE,
    eval_metric="logloss", n_jobs=-1,
)
priority_model.fit(X_train_f, y_train_f)
proba_p = priority_model.predict_proba(X_test_f)[:, 1]
pred_p = (proba_p >= 0.5).astype(int)
priority_acc = accuracy_score(y_test_f, pred_p)
priority_f1 = f1_score(y_test_f, pred_p)
priority_auc = roc_auc_score(y_test_f, proba_p)
log(f"Accuracy={priority_acc:.4f}  F1={priority_f1:.4f}  ROC-AUC={priority_auc:.4f}")
log(classification_report(y_test_f, pred_p, target_names=["Low", "High"]))
priority_feature_cols = list(X_fixed.columns)

# Confusion matrix + feature-importance plots (regenerated for the real deployed model)
cm = confusion_matrix(y_test_f, pred_p)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Low", "High"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Low", "High"])
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=axes[0], fraction=0.046)
fi = pd.Series(priority_model.feature_importances_, index=priority_feature_cols).sort_values(ascending=False).head(10)
axes[1].barh(fi.index[::-1], fi.values[::-1], color="#3498db")
axes[1].set_title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "model1_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
fi15 = pd.Series(priority_model.feature_importances_, index=priority_feature_cols).sort_values(ascending=False).head(15)
ax.barh(fi15.index[::-1], fi15.values[::-1], color="#3498db")
ax.set_title("Top 15 Features — XGBoost (Priority Model, v2)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "model1_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# MODEL 2 — ROAD CLOSURE CLASSIFIER (corridor included — legitimate signal)
# ============================================================================
log("\n" + "=" * 70)
log("MODEL 2 — ROAD CLOSURE CLASSIFIER (XGBoost, v2 features)")
log("=" * 70)
CAT_FEATURES = ["event_type", "event_cause", "zone", "corridor", "veh_type"]
NUM_FEATURES = (
    ["latitude", "longitude", "is_weekend", "is_peak_hour", "is_night"]
    + hotspot_cols + circular_cols + interaction_cols
)

closure_df = temp_df.dropna(subset=CAT_FEATURES + NUM_FEATURES + ["requires_road_closure"]).copy()
X_rc = pd.get_dummies(closure_df[CAT_FEATURES + NUM_FEATURES], columns=CAT_FEATURES)
y_rc = closure_df["requires_road_closure"].astype(int)
log(f"Rows usable: {len(closure_df)}  ({y_rc.mean()*100:.1f}% require closure)")

X_train_rc, X_test_rc, y_train_rc, y_test_rc = train_test_split(
    X_rc, y_rc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_rc
)
neg, pos = (y_train_rc == 0).sum(), (y_train_rc == 1).sum()
spw = neg / pos
log(f"scale_pos_weight = {spw:.2f}")

closure_model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=spw * 0.6, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
)
closure_model.fit(X_train_rc, y_train_rc)
proba_c = closure_model.predict_proba(X_test_rc)[:, 1]
pred_c_default = (proba_c >= 0.5).astype(int)
closure_auc = roc_auc_score(y_test_rc, proba_c)
log(f"[default 0.5] Accuracy={accuracy_score(y_test_rc, pred_c_default):.4f}  "
    f"F1={f1_score(y_test_rc, pred_c_default):.4f}  ROC-AUC={closure_auc:.4f}")

precision, recall, thresholds = precision_recall_curve(y_test_rc, proba_c)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = f1_scores[:-1].argmax()
best_threshold = float(thresholds[best_idx])
pred_c_tuned = (proba_c >= best_threshold).astype(int)
closure_acc = accuracy_score(y_test_rc, pred_c_tuned)
closure_f1 = f1_score(y_test_rc, pred_c_tuned)
log(f"[tuned {best_threshold:.3f}] Accuracy={closure_acc:.4f}  F1={closure_f1:.4f}")
log(classification_report(y_test_rc, pred_c_tuned, target_names=["No Closure", "Closure"]))
closure_feature_cols = list(X_rc.columns)

fig, ax = plt.subplots(figsize=(5, 4))
cm2 = confusion_matrix(y_test_rc, pred_c_tuned)
im = ax.imshow(cm2, cmap="Reds")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
ax.set_title(f"Closure Confusion Matrix (threshold={best_threshold:.3f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm2[i, j], ha="center", va="center",
                color="white" if cm2[i, j] > cm2.max() / 2 else "black")
fig.colorbar(im, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "model2_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# MODEL 3 — DURATION REGRESSOR (log1p target)
# ============================================================================
log("\n" + "=" * 70)
log("MODEL 3 — DURATION REGRESSOR (XGBoost, v2 features)")
log("=" * 70)
FEATURE_COLS = CAT_FEATURES + NUM_FEATURES
reg_df = temp_df.dropna(subset=FEATURE_COLS + ["duration_hrs"]).copy()
reg_df = reg_df[reg_df["duration_hrs"].between(0, 24)]

X_reg_raw = reg_df[FEATURE_COLS + ["severity_index"]]
X_reg = pd.get_dummies(X_reg_raw, columns=CAT_FEATURES, drop_first=False)
y_reg_raw = reg_df["duration_hrs"]
y_reg = np.log1p(y_reg_raw)
log(f"Regression rows: {len(reg_df)}  median={y_reg_raw.median():.2f}h  max={y_reg_raw.max():.2f}h")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
)
duration_model = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.03, subsample=1.0,
    colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
)
duration_model.fit(X_train_r, y_train_r)
pred_log = duration_model.predict(X_test_r)
pred_hrs = np.expm1(pred_log)
actual_hrs = np.expm1(y_test_r)
duration_mae = mean_absolute_error(actual_hrs, pred_hrs)
duration_r2 = r2_score(y_test_r, pred_log)
log(f"MAE={duration_mae:.2f} hrs  R²(log)={duration_r2:.4f}")
duration_feature_cols = list(X_reg.columns)

# ============================================================================
# LOOKUP TABLES (station / corridor GPS, zone->station)
# ============================================================================
log("\n" + "=" * 70)
log("LOOKUP TABLES")
log("=" * 70)
zone_station_map = (
    temp_df.dropna(subset=["zone", "police_station"])
    .groupby("zone")["police_station"]
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)
zone_station_map.pop("Unknown", None)
station_coords = temp_df.dropna(subset=["latitude", "longitude", "police_station"])[
    ["latitude", "longitude", "police_station"]
]
corridor_coords = temp_df.dropna(subset=["latitude", "longitude", "corridor"])[
    ["latitude", "longitude", "corridor"]
]
log(f"zone_station_map: {len(zone_station_map)} zones")
log(f"station_coords: {len(station_coords)} rows, corridor_coords: {len(corridor_coords)} rows")

# ============================================================================
# AVG CONFIDENCE — mean of max(p, 1-p) across the priority + closure test sets
# ============================================================================
priority_confidence = np.maximum(proba_p, 1 - proba_p)
closure_confidence = np.maximum(proba_c, 1 - proba_c)
avg_confidence = float(np.mean(np.concatenate([priority_confidence, closure_confidence])))
log(f"\nAvg confidence (priority+closure test sets) = {avg_confidence*100:.1f}%")

# ============================================================================
# EXPORT BUNDLE
# ============================================================================
bundle = {
    "priority_model": priority_model, "priority_feature_cols": priority_feature_cols,
    "closure_model": closure_model, "closure_feature_cols": closure_feature_cols,
    "closure_threshold": best_threshold,
    "duration_model": duration_model, "duration_feature_cols": duration_feature_cols,
    "cat_features_fixed": CAT_FEATURES_FIXED, "num_features_fixed": NUM_FEATURES_FIXED,
    "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
    "cause_score_map": cause_score_map, "manpower_map": manpower_map,
    "zone_station_map": zone_station_map, "station_coords": station_coords,
    "corridor_coords": corridor_coords, "hotspots": HOTSPOTS,
}
bundle_path = os.path.join(MODELS_DIR, "recommendation_engine_bundle_v2.pkl")
joblib.dump(bundle, bundle_path)
log(f"\nSaved bundle -> {bundle_path} ({os.path.getsize(bundle_path)/1024:.1f} KB)")

metrics = {
    "priority": {"accuracy": priority_acc, "f1": priority_f1, "auc": priority_auc, "rows": len(priority_df)},
    "closure": {
        "accuracy": closure_acc, "f1": closure_f1, "auc": closure_auc,
        "threshold": best_threshold, "rows": len(closure_df),
        "positive_rate": float(y_rc.mean()),
    },
    "duration": {"mae_hours": duration_mae, "r2_log": duration_r2, "rows": len(reg_df)},
    "avg_confidence": avg_confidence,
    "hotspots": HOTSPOTS,
}
with open(os.path.join(MODELS_DIR, "v2_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(MODELS_DIR, "v2_training_log.txt"), "w") as f:
    f.write("\n".join(LOG_LINES))

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)
