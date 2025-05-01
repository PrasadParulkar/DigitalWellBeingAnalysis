"""
Digital Balance — Manual 3‑Level Classifier (Derived Features)
=============================================================
This **Streamlit** app aligns with Phase 1’s derived columns:
* Four granular usage inputs (Social, Video, Gaming, Messaging) → **Total Digital Activity**
* **Digital Overuse** flag = "Overuse" when total > 8 h, else "Normal"

The model still expects six columns, so we auto‑compute **Screen Time** (as the sum) plus the derived fields, then feed those to `phase2_model.pkl`.

Folder
------
```
src/phase4/
│   app.py
│   requirements.txt
└── artifacts/
    └── phase2_model.pkl
```
Run:
```bash
cd src/phase4
pip install -r requirements.txt
streamlit run app.py
```
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

# ── Config & model ─────────────────────────────────────────────
ART_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ART_DIR / "phase2_model.pkl"

st.set_page_config(page_title="Digital Balance Classifier", layout="wide", page_icon="🎯")
st.title("🎯 Digital Balance — 3‑Level Well‑Being Classifier")

if not MODEL_PATH.exists():
    st.error("phase2_model.pkl missing — add it under src/phase4/artifacts/")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

pipe = load_model(MODEL_PATH)

# ── Helper maps ────────────────────────────────────────────────
CLASS_MAP = {
    "Low":    (1, 4,  "#e45756"),   # 1.0 ≤ score  < 4.0
    "Medium": (4, 8,  "#f1ab00"),   # 4.0 ≤ score  < 8.0
    "High":   (8, 10.1, "#4caf50")  # 8.0 ≤ score ≤ 10.0
}

def score_to_class(score: float) -> str:
    for label, (lo, hi, _) in CLASS_MAP.items():
        if lo <= score < hi:        #  ←  use < hi   (not ≤)
            return label
    return "Unknown"

def advice(label: str) -> str:
    return {
        "Low":    "* Limit screen time below 6 h / day\n* Add 30 min exercise\n* Mute notifications before sleep",
        "Medium": "* Stay balanced with app timers\n* Track mood & sleep weekly",
        "High":   "Great balance! Keep up the good habits ✨"
    }.get(label, "")

# ── Manual input form ──────────────────────────────────────────
st.subheader("Enter today’s digital‑use metrics (hrs)")
col1, col2 = st.columns(2)
with col1:
    sm_time   = st.number_input("Social Media",   0.0, 24.0, 2.0, 0.25)
    video     = st.number_input("Video Content",  0.0, 24.0, 1.0, 0.25)
with col2:
    gaming    = st.number_input("Gaming",         0.0, 24.0, 0.5, 0.25)
    messaging = st.number_input("Messaging",      0.0, 24.0, 2.0, 0.25)

activity = st.number_input("Physical Activity (hrs)", 0.0, 10.0, 1.0, 0.25, key="act")
fatigue  = st.slider("Social Media Fatigue (1‑10)", 1, 10, 5)

# Derived features
total_hours = sm_time + video + gaming + messaging
screen_time = total_hours  # treat combined usage as screen time
overuse_flag_num = 1 if total_hours > 8 else 0
overuse_text = "Overuse" if overuse_flag_num else "Normal"

st.markdown(f"**Total Digital Activity:** `{total_hours:.2f} h`  —  Flag: **{overuse_text}**")

if st.button("🔍 Predict Well‑Being Class", type="primary"):
    # Construct the exact feature set expected by the model
    user_df = pd.DataFrame({
        "Daily Social Media Time (hrs)": [sm_time],
        "Screen Time (hrs)":               [screen_time],
        "Physical Activity Time (hrs)":    [activity],
        "Social Media Fatigue Level (scale 1-10)": [fatigue],
        "Total Digital Activity (hrs)":    [total_hours],
        "Digital Overuse":                 [overuse_flag_num]
    })

    pred_score = float(pipe.predict(user_df)[0] + 1)
    score = max(1.0, min(10.0, pred_score))
    label = score_to_class(score)
    colour = CLASS_MAP.get(label, (None, None, "#808080"))[2]

    st.markdown(f"### Predicted Score : **{score:.1f} / 10**")
    st.markdown(f"<span style='font-size:32px; color:{colour};'>Class : {label}</span>", unsafe_allow_html=True)
    st.info(advice(label))

st.divider()
