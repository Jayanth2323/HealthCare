import os
import sys
import streamlit as st
from dotenv import load_dotenv

# === Ensure root directory is available in path for local module imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Custom Module Imports ===
from modules.loaders import load_model, load_data
from modules.ui_tabs import render_tabs
from modules.utils import configure_genai

# === Load and Configure Gemini API ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
configure_genai(api_key)

# === Streamlit App Config ===
st.set_page_config(page_title="AI Healthcare Advisor", layout="wide", page_icon="🧠")
st.title("🧠 AI-Driven Personalized Healthcare Advisor")
st.markdown("Empowering health decisions through machine intelligence.")

# === Load ML Model and Patient Data ===
model = load_model("models/logistic_regression_pipeline.pkl")
df = load_data("data/cleaned_blood_data.csv")

if df.empty:
    st.error("❌ Data not found or empty. Please verify your CSV path.")
    st.stop()

# === Launch Full Dashboard Tabs ===
render_tabs(model, df)
