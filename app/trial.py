import os
import uuid
import urllib.request
import logging
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from io import BytesIO
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from fpdf import FPDF

# === Font Bootstrap Helpers ===
FONT_DIR  = "fonts"
FONT_NAME = "DejaVuSans.ttf"
FONT_PATH = os.path.join(FONT_DIR, FONT_NAME)
RAW_URL   = "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf/DejaVuSans.ttf"

def _is_valid_ttf(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as fh:
        tag = fh.read(4)
    return tag in (b"\x00\x01\x00\x00", b"true")

def bootstrap_font() -> str:
    os.makedirs(FONT_DIR, exist_ok=True)
    if not _is_valid_ttf(FONT_PATH):
        logging.info("Downloading DejaVuSans.ttf …")
        urllib.request.urlretrieve(RAW_URL, FONT_PATH)
        if not _is_valid_ttf(FONT_PATH):
            raise RuntimeError("Downloaded file is not a valid TrueType font.")
    return FONT_PATH

# === Generate PDF Report ===
def generate_pdf_report(health_summary: str, ai_response: str) -> str:
    try:
        font_path = bootstrap_font()
    except Exception as font_boot_exc:
        st.error(f"❌ Unicode font bootstrap failed: {font_boot_exc}")
        return ""

    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception:
        pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary.strip())
    pdf.ln()

    try:
        pdf.set_font("DejaVu", "B", size=12)
    except:
        pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)

    try:
        pdf.set_font("DejaVu", size=12)
    except:
        pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ai_response.strip())

    os.makedirs("data", exist_ok=True)
    filename = os.path.join("data", f"health_report_{uuid.uuid4().hex}.pdf")
    pdf.output(filename)
    return filename

# === Load API Key ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=api_key)

# === Streamlit Setup ===
st.set_page_config(page_title="JakeAI Healthcare Advisor", layout="wide", page_icon="🧠")

def format_currency(amount, symbol):
    abs_amt = abs(amount)
    if symbol == "₹":
        if abs_amt >= 1e7:
            return f"{symbol}{abs_amt / 1e7:.1f}Cr"
        elif abs_amt >= 1e5:
            return f"{symbol}{abs_amt / 1e5:.1f}L"
    elif abs_amt >= 1e9:
        return f"{symbol}{abs_amt / 1e9:.1f}B"
    elif abs_amt >= 1e6:
        return f"{symbol}{abs_amt / 1e6:.1f}M"
    elif abs_amt >= 1e3:
        return f"{symbol}{abs_amt / 1e3:.1f}K"
    return f"{symbol}{amount:,}"

# === Load Model & Data ===
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "logistic_regression_pipeline.pkl")
    if not os.path.isfile(model_path):
        st.error(f"❌ Model not found at '{model_path}'.")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_data():
    base_path = "/mnt/data/cleaned_blood_data.csv"
    if not os.path.isfile(base_path):
        st.error(f"❌ Base data missing at '{base_path}'.")
        return pd.DataFrame()
    base_df = pd.read_csv(base_path)

    synth_path = "/mnt/data/synthetic_patient_dataset.csv"
    if os.path.isfile(synth_path):
        synth_df = pd.read_csv(synth_path)
        common_cols = [c for c in synth_df.columns if c in base_df.columns]
        synth_df = synth_df[common_cols]
        combined = pd.concat([base_df, synth_df], ignore_index=True)
        st.success(f"✅ Loaded and appended {len(synth_df)} synthetic records.")
        return combined
    else:
        st.warning(f"⚠️ Could not find '{synth_path}'. Using base data only.")
        return base_df

model = load_model()
df    = load_data()
if model is None or df.empty:
    st.warning("⚠️ Missing essential resources (model/data). Please verify setup.")
    st.stop()

# === SHAP Helper ===
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

def generate_recommendation(pred_label: int) -> str:
    return {
        0: "✅ **Low Risk**\nMaintain your current healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ **Medium Risk**\nIncrease physical activity, monitor diet. Schedule a medical consultation.",
        2: "🚨 **High Risk**\nImmediate medical attention advised. Begin treatment under supervision.",
    }.get(pred_label, "❓ No recommendation available.")

# === Sidebar Inputs ===
st.sidebar.header("📝 Patient Profile")
frequency     = st.sidebar.slider("📅 Visit Frequency (visits/year)", 0, 50, 5)
age           = st.sidebar.number_input("🎂 Age", min_value=1, max_value=120, value=30)
gender        = st.sidebar.selectbox("👥 Gender", ["Male", "Female", "Other"])
bp            = st.sidebar.slider("🩸 Blood Pressure (mmHg)", 80, 200, 120)
hr            = st.sidebar.slider("💓 Heart Rate (bpm)", 40, 180, 80)
cholesterol   = st.sidebar.slider("🧪 Cholesterol (mg/dL)", 100, 400, 200)
hemoglobin    = st.sidebar.slider("🩺 Hemoglobin (g/dL)", 8.0, 20.0, 14.0, step=0.1)
smoking_status= st.sidebar.selectbox("🚬 Smoking Status", ["Non-smoker","Smoker"])
exercise_level= st.sidebar.selectbox("🏃 Exercise Level", ["Low","Moderate","High"])
currency_opts = {"$":{"min":1000,"max":200000,"step":1000},"€":{"min":1000,"max":180000,"step":1000},"₹":{"min":5000,"max":2000000,"step":5000},"£":{"min":1000,"max":150000,"step":1000}}
cur_sym       = st.sidebar.selectbox("💱 Choose Currency", list(currency_opts.keys()))
rng           = currency_opts[cur_sym]
monetary      = st.sidebar.slider(f"💸 Annual Healthcare Spending ({cur_sym})",rng["min"],rng["max"],(rng["min"]+rng["max"])//2,step=rng["step"])
time_since    = st.sidebar.slider("⏳ Time Since Last Visit (months)",0,60,12)

# === Main Layout ===
st.title("🧠 Jake-Driven Personalized Healthcare Advisor")
st.markdown("Empowering health decisions through machine intelligence.")

tabs = st.tabs(["🧬 Recommendation Engine","📊 Data Intelligence","🔍 Model Insights","🤖 AI Chat Assistant","ℹ️ About"])

# === Tab 3: Model Insights ===
with tabs[2]:
    st.subheader("📈 Model Performance & Insights")

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_mask = corr.mask(mask)

    fig_corr_plotly = px.imshow(
        corr_mask,
        text_auto=".3f",
        aspect="equal",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation"
    )
    fig_corr_plotly.update_layout(xaxis_side="bottom")
    st.plotly_chart(fig_corr_plotly, use_container_width=True)

    fig_corr_mat, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink":0.9},
        ax=ax
    )
    ax.set_title("Correlation Matrix of Health Features", fontsize=16)
    st.pyplot(fig_corr_mat)

    buf = BytesIO()
    fig_corr_mat.savefig(buf, format="png")
    st.download_button(
        label="📥 Download Heatmap as PNG",
        data=buf.getvalue(),
        file_name="correlation_heatmap.png",
        mime="image/png"
    )

    # ... rest of your tabs and logic unchanged ...
