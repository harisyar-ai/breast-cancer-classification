import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OnchoScan · AI Breast Cancer Classifier",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Global Neon Styles ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --neon-pink:   #FF2D78;
    --neon-cyan:   #00F5FF;
    --neon-green:  #39FF14;
    --neon-purple: #BF5FFF;
    --bg-dark:     #080810;
    --bg-card:     #0D0D1A;
    --bg-card2:    #12122A;
    --border:      rgba(0,245,255,0.13);
    --text-main:   #E8E8FF;
    --text-muted:  #6B6B9A;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp {
    background: var(--bg-dark) !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid rgba(0,245,255,0.1) !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Sidebar nav buttons (inactive) ── */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent !important;
    border: 1px solid rgba(0,245,255,0.1) !important;
    border-radius: 10px !important;
    color: #9090C0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1rem !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,245,255,0.07) !important;
    border-color: rgba(0,245,255,0.35) !important;
    color: #00F5FF !important;
}

/* ── Main predict button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple)) !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    box-shadow: 0 0 22px rgba(255,45,120,0.45) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 38px rgba(255,45,120,0.75) !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--neon-cyan) !important;
    box-shadow: 0 0 10px var(--neon-cyan) !important;
}

hr { border-color: rgba(0,245,255,0.1) !important; }

.neon-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.page-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    font-weight: 900;
    color: var(--neon-cyan);
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.divider {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, #00F5FF, transparent);
    margin: 14px 0 22px;
}
.stat-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(0,245,255,0.07);
}
.stat-icon { font-size: 1.1rem; }
.stat-label { font-size: 0.68rem; color: #6B6B9A; text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-size: 0.82rem; color: #00F5FF; font-weight: 600; }
.result-malignant {
    background: linear-gradient(135deg, rgba(255,45,120,0.15), rgba(191,95,255,0.1));
    border: 1px solid rgba(255,45,120,0.4);
    border-radius: 14px; padding: 2rem; text-align: center;
    box-shadow: 0 0 50px rgba(255,45,120,0.12);
}
.result-benign {
    background: linear-gradient(135deg, rgba(57,255,20,0.1), rgba(0,245,255,0.07));
    border: 1px solid rgba(57,255,20,0.35);
    border-radius: 14px; padding: 2rem; text-align: center;
    box-shadow: 0 0 50px rgba(57,255,20,0.08);
}
.result-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.7rem; font-weight: 900; letter-spacing: 0.1em; margin: 8px 0 4px;
}
.result-malignant .result-title { color: #FF2D78; text-shadow: 0 0 22px rgba(255,45,120,0.7); }
.result-benign   .result-title { color: #39FF14; text-shadow: 0 0 22px rgba(57,255,20,0.7); }
.result-conf { font-size: 0.88rem; color: #E8E8FF; opacity: 0.85; margin-top: 4px; }
.result-note { font-size: 0.75rem; color: #6B6B9A; margin-top: 10px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Classify"


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def get_default_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    median_texture = df['mean texture'].median()
    means = df.mean().to_dict()
    return means, median_texture


@st.cache_resource
def load_model():
    # model = joblib.load("F:\Coding\PYTHON\ML_Projects\Breast_Cancer_Classification\model.pkl")   # ← update path if needed
    model = joblib.load("model.pkl")   # ← update path if needed
    return model


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style='text-align:center; padding:22px 0 18px;'>
        <div style='font-size:2.2rem; filter:drop-shadow(0 0 12px #FF2D78);'>🩺</div>
        <div style='font-family:Orbitron,sans-serif; font-size:1.05rem; font-weight:900;
                    color:#00F5FF; letter-spacing:0.18em; margin-top:8px;'>
            ONCO<span style='color:#FF2D78;'>SCAN</span>
        </div>
        <div style='font-size:0.6rem; color:#6B6B9A; letter-spacing:0.22em;
                    text-transform:uppercase; margin-top:4px;'>
            AI Diagnostic Tool
        </div>
        <div style='width:50px; height:1px;
                    background:linear-gradient(90deg,transparent,#00F5FF,transparent);
                    margin:14px auto 0;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-bottom:10px; font-size:0.62rem; color:#6B6B9A;"
        "letter-spacing:0.2em; text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True
    )

    nav_items = [
        ("Classify",        "🩺"),
        ("About the Study", "📋"),
        ("About Us",        "👤"),
    ]

    active_index = [label for label, _ in nav_items].index(st.session_state.page) + 1
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] .stButton:nth-of-type({active_index}) > button {{
        background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(191,95,255,0.12)) !important;
        border: 1px solid rgba(0,245,255,0.45) !important;
        color: #00F5FF !important;
        font-weight: 600 !important;
        box-shadow: 0 0 14px rgba(0,245,255,0.12) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    for label, icon in nav_items:
        is_active = st.session_state.page == label
        marker    = "▶  " if is_active else "    "
        if st.button(f"{icon}  {marker}{label}", key=f"nav_{label}"):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div style='margin-top:3rem; border-top:1px solid rgba(0,245,255,0.08);
                padding-top:1rem; text-align:center; font-size:0.6rem;
                color:#6B6B9A; letter-spacing:0.13em; text-transform:uppercase;'>
        © 2025 Muhammad Haris Afridi
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — CLASSIFY
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Classify":

    st.markdown("""
    <div style='text-align:center; padding:1.8rem 0 0.2rem;'>
        <div style='font-size:2.8rem;'>🎗️</div>
        <div style='font-family:Orbitron,sans-serif; font-size:1.7rem; font-weight:900;
                    background:linear-gradient(90deg,#00F5FF,#FF2D78,#BF5FFF);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:0.06em; margin-top:10px;'>
            Breast Cancer Classifier
        </div>
        <div style='color:#6B6B9A; font-size:0.83rem; margin-top:8px;
                    max-width:480px; margin-inline:auto; line-height:1.6;'>
            Adjust the tumor measurements below — the model will predict whether
            the tumor is <span style='color:#39FF14; font-weight:600;'>Benign</span> or
            <span style='color:#FF2D78; font-weight:600;'>Malignant</span>.
        </div>
        <div style='width:70px; height:2px;
                    background:linear-gradient(90deg,transparent,#00F5FF,transparent);
                    margin:16px auto 28px;'></div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    means, median_texture = get_default_data()

    # ── Sliders only — no cards, no table ────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        mean_radius     = st.slider("Mean Radius",     6.0,   30.0,    14.0,  step=0.1)
        mean_perimeter  = st.slider("Mean Perimeter",  43.0,  190.0,   91.0,  step=0.5)
        mean_smoothness = st.slider("Mean Smoothness", 0.05,  0.17,    0.096, step=0.001, format="%.3f")
    with col2:
        mean_texture = st.slider("Mean Texture", 9.0,   40.0,   19.0,  step=0.1)
        mean_area    = st.slider("Mean Area",    143.0, 2501.0, 654.0, step=1.0)

    # Build input dataframe
    user_data = means.copy()
    user_data['mean radius']     = mean_radius
    user_data['mean texture']    = mean_texture
    user_data['mean perimeter']  = mean_perimeter
    user_data['mean area']       = mean_area
    user_data['mean smoothness'] = mean_smoothness
    user_data['tumor_size_category'] = (
        "Small" if mean_radius <= 12 else "Medium" if mean_radius <= 18 else "Large"
    )
    user_data['texture_type'] = "Rough" if mean_texture > median_texture else "Smooth"
    input_df = pd.DataFrame(user_data, index=[0])

    # ── Run Diagnostic button ─────────────────────────────────────────────────
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        predict_clicked = st.button("⚡ Run Diagnostic", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing biomarkers..."):
            prediction = model.predict(input_df)
            try:
                prob = model.predict_proba(input_df)[0]
                conf_str = f"Model confidence: <strong>{np.max(prob)*100:.1f}%</strong>"
            except AttributeError:
                conf_str = ""

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        if prediction[0] == 0:
            st.markdown(f"""
            <div class='result-malignant'>
                <div style='font-size:2rem;'>🚨</div>
                <div class='result-title'>MALIGNANT</div>
                <div class='result-conf'>{conf_str}</div>
                <div class='result-note'>
                    For informational purposes only.<br>
                    Please consult a qualified medical professional immediately.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-benign'>
                <div style='font-size:2rem;'>✅</div>
                <div class='result-title'>BENIGN</div>
                <div class='result-conf'>{conf_str}</div>
                <div class='result-note'>
                    Measurements suggest benign characteristics.<br>
                    Regular medical check-ups are still recommended.
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:4rem; text-align:center; font-size:0.62rem; color:#6B6B9A;
                letter-spacing:0.14em; text-transform:uppercase;
                border-top:1px solid rgba(0,245,255,0.07); padding-top:1.2rem;'>
        © 2026 Muhammad Haris Afridi | Powered by &nbsp;
        <span style='color:#00F5FF;'>Scikit-Learn</span> ·
        <span style='color:#FF2D78;'>Lasso Regression</span> ·
        <span style='color:#BF5FFF;'>Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — ABOUT THE STUDY
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About the Study":

    st.markdown('<div class="page-title">About the Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6B6B9A; font-size:0.85rem; line-height:1.6; margin-bottom:1.8rem;'>"
        "Everything you need to know about how this AI classifier was built and why it matters.</p>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='neon-card'>
        <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#00F5FF;
                    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:12px;'>
            Why It Matters
        </div>
        <p style='color:#E8E8FF; font-size:0.9rem; line-height:1.85; margin:0;'>
            Breast cancer is one of the most prevalent cancers globally, affecting millions each year.
            <strong style='color:#00F5FF;'>Early detection</strong> is the single greatest factor in
            improving survival rates — yet diagnostic access remains unequal worldwide. This tool shows
            how machine learning can assist clinicians by providing fast, data-driven tumor classification
            from a handful of measurable biomarkers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='neon-card'>
            <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#BF5FFF;
                        letter-spacing:0.18em; text-transform:uppercase; margin-bottom:14px;'>
                Dataset
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🗂️</span>
                <div><div class='stat-label'>Source</div>
                     <div class='stat-value'>Breast Cancer Wisconsin</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🔢</span>
                <div><div class='stat-label'>Samples</div>
                     <div class='stat-value'>569 patient records</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>📐</span>
                <div><div class='stat-label'>Features</div>
                     <div class='stat-value'>30 numeric biomarkers</div></div>
            </div>
            <div class='stat-row' style='border:none;'>
                <span class='stat-icon'>🏷️</span>
                <div><div class='stat-label'>Classes</div>
                     <div class='stat-value'>Malignant · Benign</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='neon-card'>
            <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#FF2D78;
                        letter-spacing:0.18em; text-transform:uppercase; margin-bottom:14px;'>
                Model Pipeline
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🔬</span>
                <div><div class='stat-label'>Algorithm</div>
                     <div class='stat-value'>Lasso Logistic Regression</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>⚖️</span>
                <div><div class='stat-label'>Preprocessing</div>
                     <div class='stat-value'>StandardScaler + Encoding</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🧪</span>
                <div><div class='stat-label'>Validation</div>
                     <div class='stat-value'>Train / Test Split</div></div>
            </div>
            <div class='stat-row' style='border:none;'>
                <span class='stat-icon'>📦</span>
                <div><div class='stat-label'>Serialisation</div>
                     <div class='stat-value'>joblib (.pkl)</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='neon-card' style='margin-top:0.2rem;'>
        <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#39FF14;
                    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:18px;'>
            How It Works
        </div>
        <div style='display:flex; flex-direction:column; gap:18px;'>
            <div style='display:flex; gap:16px; align-items:flex-start;'>
                <div style='background:rgba(0,245,255,0.12); border-radius:50%; width:34px; height:34px; flex-shrink:0;
                            display:flex; align-items:center; justify-content:center;
                            font-family:Orbitron,sans-serif; font-size:0.78rem; color:#00F5FF; font-weight:700;'>1</div>
                <div>
                    <div style='color:#E8E8FF; font-size:0.88rem; font-weight:600; margin-bottom:4px;'>Input Measurements</div>
                    <div style='color:#6B6B9A; font-size:0.82rem; line-height:1.6;'>Five key tumor measurements are collected: radius, texture, perimeter, area, and smoothness. Two engineered features are derived automatically.</div>
                </div>
            </div>
            <div style='display:flex; gap:16px; align-items:flex-start;'>
                <div style='background:rgba(191,95,255,0.12); border-radius:50%; width:34px; height:34px; flex-shrink:0;
                            display:flex; align-items:center; justify-content:center;
                            font-family:Orbitron,sans-serif; font-size:0.78rem; color:#BF5FFF; font-weight:700;'>2</div>
                <div>
                    <div style='color:#E8E8FF; font-size:0.88rem; font-weight:600; margin-bottom:4px;'>Feature Scaling</div>
                    <div style='color:#6B6B9A; font-size:0.82rem; line-height:1.6;'>The pipeline standardises numerical features using StandardScaler and one-hot encodes categorical columns — matching exactly the preprocessing applied during training.</div>
                </div>
            </div>
            <div style='display:flex; gap:16px; align-items:flex-start;'>
                <div style='background:rgba(255,45,120,0.12); border-radius:50%; width:34px; height:34px; flex-shrink:0;
                            display:flex; align-items:center; justify-content:center;
                            font-family:Orbitron,sans-serif; font-size:0.78rem; color:#FF2D78; font-weight:700;'>3</div>
                <div>
                 
