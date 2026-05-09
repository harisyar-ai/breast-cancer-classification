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

#MainMenu, footer { visibility: hidden; }
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

    st.markdown("""
    <style>
    /* Radio label text */
    [data-testid="stSidebar"] .stRadio label {
        color: #9090C0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        padding: 4px 0 !important;
        cursor: pointer !important;
        transition: color 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #00F5FF !important;
    }
    /* Radio dot — unselected */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {
        border-color: rgba(0,245,255,0.35) !important;
        background: transparent !important;
    }
    /* Radio dot — selected fill */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"][aria-checked="true"] div:first-child {
        border-color: #00F5FF !important;
        background: #00F5FF !important;
        box-shadow: 0 0 8px rgba(0,245,255,0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-bottom:10px; font-size:0.62rem; color:#6B6B9A;"
        "letter-spacing:0.2em; text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True
    )

    nav_labels = ["🩺  Classify", "📋  About the Study", "👤  About Us"]
    nav_keys   = ["Classify", "About the Study", "About Us"]

    current_label = nav_labels[nav_keys.index(st.session_state.page)]
    selected = st.radio("Navigation", nav_labels, index=nav_labels.index(current_label), label_visibility="hidden")
    selected_page = nav_keys[nav_labels.index(selected)]
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
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
                    <div style='color:#E8E8FF; font-size:0.88rem; font-weight:600; margin-bottom:4px;'>Lasso Classification</div>
                    <div style='color:#6B6B9A; font-size:0.82rem; line-height:1.6;'>The L1-regularised logistic regression model assigns a probability to each class, keeping the model sparse and reducing overfitting.</div>
                </div>
            </div>
            <div style='display:flex; gap:16px; align-items:flex-start;'>
                <div style='background:rgba(57,255,20,0.1); border-radius:50%; width:34px; height:34px; flex-shrink:0;
                            display:flex; align-items:center; justify-content:center;
                            font-family:Orbitron,sans-serif; font-size:0.78rem; color:#39FF14; font-weight:700;'>4</div>
                <div>
                    <div style='color:#E8E8FF; font-size:0.88rem; font-weight:600; margin-bottom:4px;'>Result + Confidence</div>
                    <div style='color:#6B6B9A; font-size:0.82rem; line-height:1.6;'>The prediction (Benign or Malignant) is returned along with a confidence score for a clear, interpretable output.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding:14px 18px; background:rgba(255,224,0,0.06);
                border:1px solid rgba(255,224,0,0.2); border-radius:10px;
                font-size:0.8rem; color:#CCCC80; line-height:1.7;'>
        ⚠️ <strong style='color:#FFE000;'>Disclaimer:</strong>
        This tool is for educational and research purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — ABOUT US
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About Us":

    st.markdown('<div class="page-title">About the Developer</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class='neon-card'>
            <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#00F5FF;
                        letter-spacing:0.18em; text-transform:uppercase; margin-bottom:14px;'>
                Muhammad Haris Afridi
            </div>
            <p style='color:#E8E8FF; font-size:0.9rem; line-height:1.85; margin-bottom:12px;'>
                I'm a passionate, self-taught
                <strong style='color:#00F5FF;'>AI Engineer</strong> and
                <strong style='color:#BF5FFF;'>Full-Stack Developer</strong> from Peshawar, Pakistan.
                I love turning raw ideas into real-world tools — whether it's a machine learning model,
                a government web platform, or an intelligent recommender system.
            </p>
            <p style='color:#6B6B9A; font-size:0.85rem; line-height:1.8; margin:0;'>
                I'm driven by the challenge of solving problems that actually matter —
                from predicting car prices for Pakistani buyers to building government MIS platforms
                that transform how districts plan development projects.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='neon-card'>
            <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#BF5FFF;
                        letter-spacing:0.18em; text-transform:uppercase; margin-bottom:14px;'>
                Current Position
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>🏛️</span>
                <div><div class='stat-label'>University</div>
                     <div class='stat-value'>Agriculture University, Peshawar</div></div>
            </div>
            <div class='stat-row'>
                <span class='stat-icon'>📅</span>
                <div><div class='stat-label'>Degree</div>
                     <div class='stat-value'>BS Artificial Intelligence · 2023–2027</div></div>
            </div>
            <div class='stat-row' style='border:none;'>
                <span class='stat-icon'>🔭</span>
                <div><div class='stat-label'>Research</div>
                     <div class='stat-value'>DIP Lab · Islamia College Peshawar</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tech stack tags
    st.markdown("""
    <div class='neon-card'>
        <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#39FF14;
                    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:14px;'>
            Tech Stack Used in This Project
        </div>
        <div style='display:flex; flex-wrap:wrap; gap:10px;'>
            <span style='background:rgba(0,245,255,0.1); border:1px solid rgba(0,245,255,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#00F5FF;'>Python</span>
            <span style='background:rgba(191,95,255,0.1); border:1px solid rgba(191,95,255,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#BF5FFF;'>Scikit-Learn</span>
            <span style='background:rgba(255,45,120,0.1); border:1px solid rgba(255,45,120,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#FF2D78;'>Lasso Regression</span>
            <span style='background:rgba(57,255,20,0.08); border:1px solid rgba(57,255,20,0.2);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#39FF14;'>Pandas · NumPy</span>
            <span style='background:rgba(0,245,255,0.1); border:1px solid rgba(0,245,255,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#00F5FF;'>Streamlit</span>
            <span style='background:rgba(191,95,255,0.1); border:1px solid rgba(191,95,255,0.25);
                         border-radius:20px; padding:5px 14px; font-size:0.78rem; color:#BF5FFF;'>joblib</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Connect — SVG icons inline, no external image requests
    st.markdown("""
    <div class='neon-card'>
        <div style='font-family:Orbitron,sans-serif; font-size:0.72rem; color:#FF2D78;
                    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:14px;'>
            Connect With Me
        </div>
        <div style='font-size:0.88rem; color:#E8E8FF; line-height:2;'>
            Click here for &nbsp;
            <a href='https://www.linkedin.com/in/harisyar-ai' target='_blank'
               style='display:inline-flex; align-items:center; gap:6px; text-decoration:none; color:#0A66C2; font-weight:600;'>
                <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='#0A66C2' style='vertical-align:middle;'>
                    <path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/>
                </svg>
                LinkedIn
            </a>
            &nbsp;<span style='color:#6B6B9A;'>|</span>&nbsp;
            <a href='https://github.com/harisyar-ai' target='_blank'
               style='display:inline-flex; align-items:center; gap:6px; text-decoration:none; color:#BF5FFF; font-weight:600;'>
                <svg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='#BF5FFF' style='vertical-align:middle;'>
                    <path d='M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12'/>
                </svg>
                GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
