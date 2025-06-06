import os
import uuid
import urllib.request
import logging

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
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv
from fpdf import FPDF

# === Font Bootstrap Helpers ===
FONT_DIR  = "fonts"
FONT_NAME = "DejaVuSans.ttf"
FONT_PATH = os.path.join(FONT_DIR, FONT_NAME)

RAW_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf/DejaVuSans.ttf"

def _is_valid_ttf(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as fh:
        tag = fh.read(4)
    return tag in (b"\x00\x01\x00\x00", b"true")

def bootstrap_font() -> str:
    os.makedirs(FONT_DIR, exist_ok=True)
    if not _is_valid_ttf(FONT_PATH):
        try:
            logging.info("Downloading DejaVuSans.ttf …")
            urllib.request.urlretrieve(RAW_URL, FONT_PATH)
        except Exception as download_exc:
            raise RuntimeError(f"Font download failed → {download_exc}") from download_exc
        if not _is_valid_ttf(FONT_PATH):
            raise RuntimeError("Downloaded file is not a valid TrueType font.")
    return FONT_PATH

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
    except Exception as font_exc:
        st.warning(f"⚠️ Could not add DejaVu TTF font: {font_exc}\nFalling back to Arial.")
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
    unique_id = str(uuid.uuid4())
    output_path = os.path.join("data", f"health_report_{unique_id}.pdf")
    try:
        pdf.output(output_path)
    except Exception as export_exc:
        st.error(f"❌ Unable to save PDF report: {export_exc}")
        return ""

    return output_path

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=api_key)

st.set_page_config(page_title="JakeAI Healthcare Advisor", layout="wide", page_icon="🧠")

@st.cache_resource
def load_model():
    model_path = "models/logistic_regression_pipeline.pkl"
    if not os.path.isfile(model_path):
        st.error(f"❌ Model not found at '{model_path}'.")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = "data/cleaned_blood_data.csv"
    if not os.path.isfile(data_path):
        st.error(f"❌ Data file missing at '{data_path}'.")
        return pd.DataFrame()
    return pd.read_csv(data_path)

model = load_model()
df = load_data()

if model is None or df.empty:
    st.warning("⚠️ Missing essential resources (model/data). Please verify setup.")
    st.stop()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

def generate_recommendation(pred_label: int) -> str:
    return {
        0: "✅ **Low Risk**\nMaintain your current healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ **Medium Risk**\nIncrease physical activity, monitor diet. Schedule a medical consultation.",
        2: "🚨 **High Risk**\nImmediate medical attention advised. Begin treatment under supervision.",
    }.get(pred_label, "❓ No recommendation available.")

# === Inserted: Gender vs Diabetes Count ===
st.markdown("#### 🧍 Gender-wise Diabetes Distribution")
fig_gen, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="gender", hue="diabetes", ax=ax)
ax.set_title("Diabetes Distribution by Gender")
st.pyplot(fig_gen)

# === Inserted: Correlation Heatmap ===
st.markdown("#### 🔥 Feature Correlation Heatmap")
fig_corr, ax = plt.subplots(figsize=(14, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket_r", linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix of Key Health Features")
st.pyplot(fig_corr)
