from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from catboost import CatBoostClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────
ART_DIR    = Path(__file__).parent / "artifacts"
MODEL_PATH = ART_DIR / "catboost_dbp.cbm"
META_PATH  = ART_DIR / "dbp_meta.json"

st.set_page_config(
    page_title="Digital Wellbeing Classifier",
    layout="wide",
    page_icon="🎯"
)
st.title("🎯 Digital Wellbeing Classifier")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Missing `catboost_dbp.cbm` or `dbp_meta.json` in `artifacts/`.")
    st.stop()

@st.cache_resource
def load_model_and_meta():
    m = CatBoostClassifier()
    m.load_model(str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text())
    return m, meta

model, meta      = load_model_and_meta()
thresholds       = np.array(meta["thresholds"])
feature_columns  = meta["features"]
CLASS_LABELS     = {0: "Low", 1: "Medium", 2: "High"}
ADVICE_BY_CLASS  = {
    "Low":    "• Limit screen time below 6 h/day  \n• Add 30 min exercise  \n• Mute notifications before sleep",
    "Medium": "• Stay balanced with app timers  \n• Track mood & sleep weekly",
    "High":   "Great balance! Keep up the good habits ✨"
}

# ── USER INPUTS ──────────────────────────────────────────────────────────
st.subheader("Enter Your Daily Metrics")
sm_time   = st.number_input("Daily Social Media Time (hrs)",   0.0, 24.0, 2.0, 0.25)
ent_time  = st.number_input("Daily Entertainment Time (hrs)",  0.0, 24.0, 1.0, 0.25)
screen    = st.number_input("Screen Time (hrs)",               0.0, 24.0, 4.0, 0.25)
sleep     = st.number_input("Average Sleep Time (hrs)",       0.0, 24.0, 7.0, 0.25)
activity  = st.number_input("Physical Activity Time (hrs)",    0.0, 24.0, 1.0, 0.25)
notifications = st.number_input("Notifications Received Daily", 0, 1000, 50, 1)

# ── OVERUSE FLAG ─────────────────────────────────────────────────────────
overuse = screen > 8
flag_txt = "🚩 Overuse" if overuse else "✅ Normal"
st.markdown(f"**Screen-time Flag:** {flag_txt}")

# ── PREDICTION ───────────────────────────────────────────────────────────
if st.button("🔍 Predict Well-Being Class"):
    # build DF in the exact column order your model needs
    user_df = pd.DataFrame([{
        "Daily Social Media Time (hrs)": sm_time,
        "Daily Entertainment Time (hrs)": ent_time,
        "Screen Time (hrs)":               screen,
        "Average Sleep Time (hrs)":        sleep,
        "Physical Activity Time (hrs)":     activity,
        "Notifications Received Daily":     notifications
    }], columns=feature_columns)

    probs = model.predict_proba(user_df)
    # apply per-class thresholds
    pred = np.where(
        (probs < thresholds).all(axis=1),
        probs.argmax(axis=1),
        (probs >= thresholds).argmax(axis=1)
    )[0]

    label = CLASS_LABELS.get(pred, "Unknown")
    st.subheader(f"Predicted Class: **{label}**")

    # advice based on overuse
    if overuse:
        st.warning("Your screen time is high—consider limiting it to under 8 h/day.")
    else:
        st.success("Your screen time is within a healthy range. 👍")

    # advice based on predicted class
    st.markdown(f"**Advice:**  \n{ADVICE_BY_CLASS.get(label,'')}")
