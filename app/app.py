import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bengaluru Event-Driven Congestion", page_icon="🚦", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load('/content/drive/MyDrive/gridlock_models/recommendation_engine_bundle.pkl')

B = load_bundle()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def nearest_police_station(lat, lon, k=5):
    sc = B['station_coords']
    d = haversine_km(lat, lon, sc['latitude'].values, sc['longitude'].values)
    idx = np.argsort(d)[:k]
    return sc.iloc[idx]['police_station'].mode()[0], round(d[idx].mean(), 2)

def nearest_corridor(lat, lon, k=5):
    cc = B['corridor_coords']
    d = haversine_km(lat, lon, cc['latitude'].values, cc['longitude'].values)
    idx = np.argsort(d)[:k]
    return cc.iloc[idx]['corridor'].mode()[0]

def risk_bucket(score):
    if score <= 4: return 'Low'
    elif score <= 7: return 'Medium'
    elif score <= 9: return 'High'
    else: return 'Critical'

def encode_single_event(event_dict, cat_cols, num_cols, target_columns, extra_numeric=None):
    row = {c: event_dict.get(c) for c in cat_cols + num_cols}
    if extra_numeric: row.update(extra_numeric)
    row_df = pd.DataFrame([row])
    encoded = pd.get_dummies(row_df, columns=cat_cols)
    return encoded.reindex(columns=target_columns, fill_value=0)

def recommend_resources(event):
    event = event.copy()
    event.setdefault('zone', 'Unknown')
    event.setdefault('veh_type', 'unknown')
    event.setdefault('is_weekend', 1 if event.get('day_of_week', 0) in [5, 6] else 0)
    event.setdefault('is_peak_hour', 1 if event.get('hour') in [7,8,9,17,18,19,20] else 0)
    event.setdefault('is_night', 1 if (event.get('hour', 12) >= 22 or event.get('hour', 12) < 6) else 0)

    detected_corridor = nearest_corridor(event['latitude'], event['longitude'])
    event['corridor'] = detected_corridor

    Xp = encode_single_event(event, B['cat_features_fixed'], B['num_features_fixed'], B['priority_feature_cols'])
    prob_high = B['priority_model'].predict_proba(Xp)[0][1]
    pred_priority = 'High' if prob_high >= 0.5 else 'Low'

    Xc = encode_single_event(event, B['cat_features'], B['num_features'], B['closure_feature_cols'])
    prob_closure = B['closure_model'].predict_proba(Xc)[0][1]
    pred_closure = bool(prob_closure >= B['closure_threshold'])

    cause_w = B['cause_score_map'].get(event['event_cause'], 1)
    type_w = 2 if event['event_type'] == 'planned' else 1
    priority_w = 2 if pred_priority == 'High' else 1
    closure_w = 2 if pred_closure else 0
    est_severity = cause_w + type_w + priority_w + closure_w

    Xd = encode_single_event(event, B['cat_features'], B['num_features'], B['duration_feature_cols'],
                              extra_numeric={'severity_index': est_severity})
    pred_dur_hrs = np.expm1(B['duration_model'].predict(Xd)[0])

    officers = B['manpower_map'].get(est_severity, max(1, est_severity - 1))
    risk = risk_bucket(est_severity)

    if event['zone'] != 'Unknown' and event['zone'] in B['zone_station_map']:
        station = B['zone_station_map'][event['zone']]
        method = f"historical mode for {event['zone']}"
    else:
        station, dist = nearest_police_station(event['latitude'], event['longitude'])
        method = f"nearest by GPS (~{dist} km)"

    return {
        'predicted_priority': pred_priority,
        'priority_confidence': round(prob_high if pred_priority == 'High' else 1 - prob_high, 3),
        'predicted_road_closure': pred_closure,
        'closure_probability': round(prob_closure, 3),
        'predicted_duration_hours': round(pred_dur_hrs, 2),
        'estimated_severity_score': est_severity,
        'risk_level': risk,
        'recommended_officers': officers,
        'recommend_barricading': pred_closure,
        'recommended_police_station': station,
        'station_assignment_method': method,
        'detected_corridor': detected_corridor,
    }

RISK_COLORS = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c', 'Critical': '#922b21'}

def badge(text, color):
    return f'<span style="background:{color};color:white;padding:4px 14px;border-radius:14px;font-weight:600;">{text}</span>'

# ---------------- UI ----------------
st.title("🚦 Bengaluru Event-Driven Congestion — Response Recommender")
st.caption("Gridlock Hackathon 2.0 · Theme 2 · Predicts incident impact and recommends manpower, barricading, and station deployment")

col_form, col_result = st.columns([1, 1.4])

with col_form:
    st.subheader("Incoming Event")
    event_type = st.radio("Event Type", ["unplanned", "planned"], horizontal=True)
    event_cause = st.selectbox("Cause", list(B['cause_score_map'].keys()))
    lat = st.number_input("Latitude", value=12.9716, format="%.6f")
    lon = st.number_input("Longitude", value=77.5946, format="%.6f")
    c1, c2, c3 = st.columns(3)
    hour = c1.slider("Hour", 0, 23, 9)
    day_of_week = c2.selectbox("Day", list(range(7)),
                                format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    month_num = c3.selectbox("Month", list(range(1,13)), index=2)
    zone = st.selectbox("Zone (optional — leave Unknown to auto-detect)",
                         ['Unknown'] + sorted(B['zone_station_map'].keys()))
    submit = st.button("🔍 Get Recommendation", type="primary", use_container_width=True)

with col_result:
    if submit:
        event = {'event_type': event_type, 'event_cause': event_cause,
                  'latitude': lat, 'longitude': lon, 'hour': hour,
                  'day_of_week': day_of_week, 'month_num': month_num, 'zone': zone}
        r = recommend_resources(event)

        st.subheader("Recommendation")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown("**Priority**<br>" + badge(r['predicted_priority'],
                    '#e74c3c' if r['predicted_priority']=='High' else '#3498db'), unsafe_allow_html=True)
        m2.markdown("**Road Closure**<br>" + badge('YES' if r['predicted_road_closure'] else 'No',
                    '#e74c3c' if r['predicted_road_closure'] else '#2ecc71'), unsafe_allow_html=True)
        m3.markdown("**Risk Level**<br>" + badge(r['risk_level'], RISK_COLORS[r['risk_level']]), unsafe_allow_html=True)
        m4.metric("Est. Duration", f"{r['predicted_duration_hours']} hrs")

        st.write("")
        d1, d2, d3 = st.columns(3)
        d1.metric("Recommended Officers", r['recommended_officers'])
        d2.metric("Barricading Needed", "Yes" if r['recommend_barricading'] else "No")
        d3.metric("Priority Confidence", f"{r['priority_confidence']*100:.0f}%")

        st.write("")
        st.markdown(f"**Deploy from:** {r['recommended_police_station']}  \n"
                    f"*({r['station_assignment_method']})*")
        st.markdown(f"**Detected corridor:** {r['detected_corridor']}")
        st.markdown(f"**Closure probability:** {r['closure_probability']*100:.1f}%  ·  "
                    f"**Severity score:** {r['estimated_severity_score']}/11")

        st.map(pd.DataFrame({'lat':[lat], 'lon':[lon]}), zoom=12)
    else:
        st.info("Fill in the event details and click **Get Recommendation**.")
