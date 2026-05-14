import os
import warnings
import logging

# =========================
# 🔥 ENV SETTINGS (MUST BE FIRST)
# =========================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel("ERROR")

# =========================
# 🔥 CORE LIBRARIES
# =========================
import json
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

# =========================
# 🔥 TENSORFLOW (LOAD ONCE ONLY)
# =========================
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# =========================
# 🔥 REPORTLAB (PDF GENERATION)
# =========================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# 🔥 STREAMLIT COMPONENTS
# =========================
import streamlit.components.v1 as components

# =========================
# PATHS
# =========================




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIX: models folder is in SAME directory as app.py
MODEL_DIR = os.path.join(BASE_DIR, "models")

USER_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "user_history.json")

print("MODEL_DIR:", MODEL_DIR)
# =========================
# LOAD MODELS (FIXED PATHS)
# =========================
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

dl_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "dl_model.keras"),
    compile=False
)

# =========================
# FEATURE MAP
# =========================
feature_map = {
    "heart_rate": "Heart Rate (BPM)",
    "hrv": "Heart Rate Variability",
    "eda": "Electrodermal Activity",
    "skin_temp": "Skin Temperature",
    "respiration_rate": "Respiration Rate",
    "spo2": "Oxygen Saturation (SpO2)",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "cortisol": "Cortisol Hormone Level",
    "motion_level": "Body Movement Level",
    "sleep_quality": "Sleep Quality Index",
    "activity_level": "Activity Level",
    "bmi": "Body Mass Index",
    "hydration_level": "Hydration Level",
    "blood_glucose": "Blood Glucose Level"
}

# =========================
# USER DB
# =========================
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        users_db = json.load(f)
else:
    users_db = {}

def save_users():
    with open(USER_FILE, "w") as f:
        json.dump(users_db, f)

# =========================
# SESSION INIT
# =========================
defaults = {
    "page": "Home",
    "logged_in": False,
    "user": None,
    "pred": None,
    "final_prob": None,
    "risk_score": None,
    "labels": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Stress System", layout="wide")


# 🔥 ADD CSS HERE
st.markdown("""
<style>



/* Cards */
.card {
    background: linear-gradient(135deg, #6dd5ed, #2193b0);
    padding: 20px;
    border-radius: 15px;
    color: red;
    text-align: center;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


st.title("🏥 AI Hospital Stress Monitoring System")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"

with c2:
    if st.button("ℹ️ About"):
        st.session_state.page = "About"

with c3:
    if st.button("📝 Register"):
        st.session_state.page = "Register"

with c4:
    if st.button("🔐 Login"):
        st.session_state.page = "Login"

with c5:
    if st.button("🩺 Dashboard"):
        st.session_state.page = "Stress"

st.markdown("---")

page = st.session_state.page



# =========================
# HOME (FULL SaaS UI)
# =========================
if page == "Home":

    # 🔥 GLOBAL CSS (same as About)
    st.markdown("""
    <style>
    .info-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        transition: 0.3s;
        text-align: center;
    }

    .info-card:hover {
        transform: translateY(-8px);
        box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
    }

    .stat {
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # 🔥 HERO BANNER
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #36d1dc, #5b86e5);
        padding: 35px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    ">
        <h1>🏥 AI Hospital Stress Monitoring</h1>
        <p>Smart Healthcare System powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # 📊 STATS CARDS
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown('<div class="info-card"><div class="stat">98%</div><p>Model Accuracy</p></div>', unsafe_allow_html=True)

    with s2:
        st.markdown('<div class="info-card"><div class="stat">< 2s</div><p>Prediction Time</p></div>', unsafe_allow_html=True)

    with s3:
        st.markdown('<div class="info-card"><div class="stat">15+</div><p>Health Parameters</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 🔥 TOP FEATURES (CARDS)
    st.markdown("<h2 style='text-align: center;'>🚀 Why Choose Our System?</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h4>⚡ Fast AI Prediction</h4>
            Real-time analysis of patient vitals using advanced models.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h4>🧠 Smart Diagnosis</h4>
            Detects stress levels with high accuracy using hybrid AI.
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Live Reports</h4>
            Generates instant visual insights and PDF reports.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 🔥 CENTER IMAGE (CLEAN SaaS STYLE)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            "https://healthindustrytrends.com/wp-content/uploads/2024/11/revolutionizing-remote-patient-care-with-iot-technology.png",
            
            width='stretch'
        )

    st.markdown("---")

    # 🔥 SYSTEM FEATURES (CARDS)
    st.markdown("<h2 style='text-align: center;'>🏥 System Features</h2>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="info-card">
            <h4>🧠 AI Analysis Engine</h4>
            Processes HR, HRV, SpO2 and predicts stress accurately.
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="info-card">
            <h4>📡 Real-Time Monitoring</h4>
            Live dashboard with continuous patient tracking.
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="info-card">
            <h4>📄 Smart Reports</h4>
            Automated PDF reports with insights and recommendations.
        </div>
        """, unsafe_allow_html=True)

    #st.markdown("---")

   
    

# =========================
# ABOUT (FULL SaaS UI)
# =========================
elif page == "About":

    # 🔥 GLOBAL CARD + HOVER CSS
    st.markdown("""
    <style>
    .info-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        transition: 0.3s;
    }

    .info-card:hover {
        transform: translateY(-8px);
        box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # 🔥 HEADER BANNER
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #36d1dc, #5b86e5);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h2>🏥 AI Stress Monitoring System</h2>
        <p>Smart Healthcare powered by Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # 1️⃣ ABOUT SYSTEM
    st.markdown("""
    <div class="info-card">
        <h4 style="display:flex;align-items:center;gap:10px;">🏥 About System</h4>
        AI-powered stress detection platform that analyzes patient physiological data 
        using Machine Learning and Deep Learning models to predict stress levels accurately.
    </div>
    """, unsafe_allow_html=True)


     # 🔥 BOTTOM IMAGE CARDS (CONSISTENT STYLE)
    st.markdown("<h2 style='text-align: center;'>🧩 Application Highlights</h2>", unsafe_allow_html=True)

    # 🔥 BOTTOM IMAGES ROW (INTERACTIVE)
    b1, b2, b3 = st.columns(3)


    # 🔥 INTERACTIVE CLICKABLE IMAGE CARDS (JS BASED)

    components.html("""
    <style>
    .container {
        display: flex;
        gap: 20px;
        justify-content: center;
    }

    .card {
        width: 300px;
        cursor: pointer;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: 0.3s;
    }

    .card:hover {
        transform: scale(1.05);
    }

    .card img {
        width: 100%;
        height: 180px;
        object-fit: cover;
    }

    .info {
        display: none;
        padding: 10px;
        background: #f5f5f5;
        color: #333;
        font-size: 14px;
    }
    </style>

    <div class="container">

        <div class="card" onclick="toggleInfo('info1')">
            <img src="https://img.freepik.com/premium-photo/happy-woman-doctor-tablet-employees-management-hospital-workflow-clinic-staff_1203138-75607.jpg"/>
            <div id="info1" class="info">
                <b>AI Assisted Diagnosis</b><br>
                ML-based stress detection, faster clinical decisions.
            </div>
        </div>

        <div class="card" onclick="toggleInfo('info2')">
            <img src="https://healthindustrytrends.com/wp-content/uploads/2024/11/revolutionizing-remote-patient-care-with-iot-technology.png"/>
            <div id="info2" class="info">
                <b>Smart Healthcare AI</b><br>
                Real-time insights and AI-driven monitoring.
            </div>
        </div>

        <div class="card" onclick="toggleInfo('info3')">
            <img src="https://integrio.net/static/8e824f4e387101e186548e849a414203/patient-health-1.png"/>
            <div id="info3" class="info">
                <b>Patient Monitoring</b><br>
                Live dashboards and continuous tracking.
            </div>
        </div>

    </div>

    <script>
    function toggleInfo(id) {
        var el = document.getElementById(id);
        el.style.display = (el.style.display === "block") ? "none" : "block";
    }
    </script>
    """, height=420)

    

    # 2️⃣ PURPOSE
    st.markdown("""
    <div class="info-card">
        <h4 style="display:flex;align-items:center;gap:10px;">🎯 Purpose</h4>
        Helps doctors detect stress early, improve monitoring, and make faster 
        data-driven clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    # 3️⃣ HOW IT WORKS
    st.markdown("### ⚙️ How It Works")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown("""
        <div class="info-card">
            <h4>📥 Input</h4>
            Patient vitals like Heart Rate, HRV, SpO2 are collected.
        </div>
        """, unsafe_allow_html=True)

    with h2:
        st.markdown("""
        <div class="info-card">
            <h4>🤖 AI Processing</h4>
            Random Forest + Deep Learning models analyze the data.
        </div>
        """, unsafe_allow_html=True)

    with h3:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Output</h4>
            Stress level (Low / Moderate / High) with score and insights.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4️⃣ FEATURES
    st.markdown("### 🚀 Features")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="info-card">
            <h4>🧠 AI Prediction</h4>
            Hybrid ML + DL model for accurate stress detection.
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Visualization</h4>
            Interactive charts and real-time analytics.
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="info-card">
            <h4>📄 Reports</h4>
            Instant PDF report generation.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 5️⃣ BENEFITS
    st.markdown("### ✅ Benefits")

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown('<div class="info-card">⚡ Early stress detection</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="info-card">🧑‍⚕️ Faster medical decisions</div>', unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="info-card">📉 Reduced manual effort</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 6️⃣ USE CASES
    st.markdown("### 🏥 Use Cases")

    u1, u2, u3 = st.columns(3)

    with u1:
        st.markdown('<div class="info-card">🏥 Hospitals & Clinics</div>', unsafe_allow_html=True)

    with u2:
        st.markdown('<div class="info-card">🌐 Telemedicine</div>', unsafe_allow_html=True)

    with u3:
        st.markdown('<div class="info-card">📡 Remote Monitoring</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 7️⃣ DISCLAIMER
    st.markdown("### ⚠️ Disclaimer")

    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #ff758c, #ff7eb3);">
        This system supports healthcare professionals but does not replace medical advice.
    </div>
    """, unsafe_allow_html=True)
    
# =========================
# REGISTER
# =========================
elif page == "Register":

    st.subheader("📝 Register")

    with st.form("register_form", clear_on_submit=True):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        cp = st.text_input("Confirm Password", type="password")

        submit = st.form_submit_button("Create Account")

        if submit:
            u, p, cp = (u or "").strip(), (p or "").strip(), (cp or "").strip()

            if not u or not p or not cp:
                st.error("All fields required")
            elif p != cp:
                st.error("Passwords do not match")
            elif u in users_db:
                st.error("User already exists")
            else:
                users_db[u] = p
                save_users()
                st.success("Registration successful 🎉")

# =========================
# LOGIN
# =========================
elif page == "Login":

    st.subheader("🔐 Login")

    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        submit = st.form_submit_button("Login")

        if submit:
            u, p = (u or "").strip(), (p or "").strip()

            if u in users_db and users_db[u] == p:
                st.session_state.logged_in = True
                st.session_state.user = u
                st.success("Login successful")
            else:
                st.error("Invalid credentials")

# =========================
# DASHBOARD (FINAL SaaS UI)
# =========================
elif page == "Stress":

    # 🔥 DASHBOARD STYLING
    st.markdown("""
    <style>

    .section-card {
        background: linear-gradient(135deg, #ffffff, #f3f6ff);
        padding: 18px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        transition: 0.3s;
    }

    .section-card:hover {
        transform: translateY(-5px);
    }

    .indicator-card {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        background: linear-gradient(135deg, #36d1dc, #5b86e5);
    }

    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.warning("Please login first")
        st.stop()

    # 🔥 HEADER
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;">
        <h2>🏥 Patient Stress Analysis Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input("👤 Patient Name for Report")

    left, mid, right = st.columns([1.2, 1.5, 1.3])
    data_dict = {}


# =========================
# LEFT → INPUTS (NO SLIDERS)
# =========================
    with left:
        st.markdown("### 🧾 Patient Vitals")

        ranges = {
            "heart_rate": (40, 140, 72),
            "hrv": (10, 100, 60),
            "eda": (0, 10, 2),
            "skin_temp": (30, 40, 36.5),
            "respiration_rate": (8, 30, 16),
            "spo2": (85, 100, 98),
            "systolic_bp": (90, 180, 120),
            "diastolic_bp": (60, 120, 80),
            "cortisol": (0, 10, 3),
            "motion_level": (0, 10, 2),
            "sleep_quality": (0, 100, 75),
            "activity_level": (0, 100, 50),
            "bmi": (15, 40, 22),
            "hydration_level": (0, 100, 70),
            "blood_glucose": (70, 200, 100)
        }

        data_dict = {}

        for f in cols:
            min_v, max_v, default = ranges.get(f, (0, 100, 50))

            data_dict[f] = st.number_input(
                label=feature_map.get(f, f),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=1.0
            )
    data = np.array([[data_dict[c] for c in cols]])
    labels = ["Low Risk 😊", "Moderate Risk 😐", "High Risk 😟"]

    # =========================
    # MIDDLE → ANALYSIS + RESULTS
    # =========================
    with mid:

        
        if st.button("🔬 Analyze Patient"):
            with st.spinner("Analyzing..."):

                rf_prob = rf.predict_proba(data)
                dl_prob = dl_model.predict(scaler.transform(data))

                final_prob = (rf_prob + dl_prob) / 2
                final_prob = final_prob[0]

                risk_score = float(np.max(final_prob) * 100)

                # ✅ RULE-BASED LABEL (FINAL FIX)
                if risk_score < 55:
                    pred = 0
                elif risk_score <= 70:
                    pred = 1
                else:
                    pred = 2

                st.session_state.final_prob = final_prob
                st.session_state.pred = pred
                st.session_state.risk_score = risk_score

        # =========================
        # 🔥 DISPLAY JUST BELOW BUTTON
        # =========================
        if st.session_state.pred is not None:

            risk_score = st.session_state.risk_score
            pred = st.session_state.pred

            # 🔥 RESULT CARD
            # 🎯 Dynamic UI based on risk
            if risk_score < 55:
                color = "linear-gradient(135deg, #00c853, #64dd17)"   # green
                emoji = "😊"
                label = "Low Risk"
            elif risk_score <= 70:
                color = "linear-gradient(135deg, #ff9800, #ffc107)"   # orange
                emoji = "😐"
                label = "Moderate Risk"
            else:
                color = "linear-gradient(135deg, #d32f2f, #f44336)"   # red
                emoji = "😟"
                label = "High Risk"

            # 🔥 Result Card (Color + Emoji)
            st.markdown(f"""
            <div style="
                background: {color};
                padding: 20px;
                border-radius: 15px;
                color: white;
                text-align: center;
                font-weight: bold;
                box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
            ">
                <h2>{emoji} {label}</h2>
                <p style="font-size:18px;">Stress Score: {risk_score:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # 🔥 GAUGE
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 55], 'color': "green"},
                        {'range': [55, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"},
                    ]
                }
            ))
            
            st.plotly_chart(fig, width='stretch')

            # 🔥 STRESS BAR (DIRECTLY BELOW GAUGE)
            

            categories = ["Low", "Moderate", "High"]

            # Initialize all to 0
            values = [0, 0, 0]

            # Set only one based on prediction
            if pred == 0:
                values[0] = risk_score
            elif pred == 1:
                values[1] = risk_score
            else:
                values[2] = risk_score

            # Color mapping
            colors = ["green", "orange", "red"]

            fig2, ax = plt.subplots()
            ax.bar(categories, values, color=colors)

            ax.set_ylim(0, 100)
            ax.set_ylabel("Stress Level (%)")
            ax.set_title("Stress Category")

            # Show value only on active bar
            for i, v in enumerate(values):
                if v > 0:
                    ax.text(i, v + 2, f"{v:.1f}%", ha='center')

            st.pyplot(fig2)
            


    # =========================
    # RIGHT → INSIGHTS
    # =========================
    with right:

        if st.session_state.pred is not None:

            final_prob = st.session_state.final_prob
            pred = st.session_state.pred
            risk_score = st.session_state.risk_score

            st.markdown("### 🧠 Key Clinical Indicators")

            importances = rf.feature_importances_
            top = np.argsort(importances)[-5:]

            for i in reversed(top):
                key = cols[i]
                importance = float(importances[i]) * 100

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
                    padding: 10px;
                    border-radius: 8px;
                    margin-bottom: 8px;">
                    <b style="color:#000;">{feature_map.get(key, key)}</b><br>
                    <span style="color:#1a237e; font-weight:600;">
                        Importance: {importance:.2f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(importances[i]))

            # =========================
            # AI EXPLANATION (NEW 🔥)
            # =========================
            st.markdown("### 🤖 Why this prediction?")

            reasons = []

            if data_dict["heart_rate"] > 90:
                reasons.append("High Heart Rate detected")
            if data_dict["hrv"] < 40:
                reasons.append("Low HRV (stress indicator)")
            if data_dict["eda"] > 4:
                reasons.append("High skin conductance (EDA)")
            if data_dict["cortisol"] > 5:
                reasons.append("Elevated cortisol level")
            if data_dict["sleep_quality"] < 50:
                reasons.append("Poor sleep quality")
            if data_dict["spo2"] < 95:
                reasons.append("Low oxygen saturation")

            if reasons:
                for r in reasons:
                    st.warning("⚠️ " + r)
            else:
                st.success("✅ All vitals are within healthy range")

            # =========================
            # RECOMMENDATIONS
            # =========================
            st.markdown("### 💡 Recommendations")

            if pred == 0:
                recs = ["Maintain routine", "Exercise daily", "Sleep well"]
            elif pred == 1:
                recs = ["Take breaks", "Practice breathing exercises"]
            else:
                recs = ["Consult doctor", "Reduce workload", "Monitor vitals"]

            for r in recs:
                st.success(f"✔ {r}")

            # =========================
            # PDF REPORT
            # =========================
            if patient_name:

                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                content = []

                content.append(Paragraph("AI based Medical Stress Report", styles["Title"]))
                content.append(Spacer(1, 10))
                content.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
                content.append(Paragraph(f"Doctor: {st.session_state.user}", styles["Normal"]))
                content.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))
                content.append(Paragraph(f"Result: {labels[pred]}", styles["Normal"]))
                content.append(Paragraph(f"Stress Score: {risk_score:.2f}%", styles["Normal"]))

                content.append(Spacer(1, 10))
                content.append(Paragraph("Patient Inputs:", styles["Heading2"]))

                for k, v in data_dict.items():
                    content.append(Paragraph(f"{feature_map.get(k,k)}: {v}", styles["Normal"]))

                content.append(Spacer(1, 10))
                content.append(Paragraph("Recommendations:", styles["Heading2"]))

                for r in recs:
                    content.append(Paragraph(f"• {r}", styles["Normal"]))

                doc.build(content)
                buffer.seek(0)

                st.download_button(
                    "📄 Download Report",
                    data=buffer,
                    file_name=f"{patient_name}_stress_report.pdf",
                    mime="application/pdf"
                )