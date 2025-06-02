import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"❌ Model not found at {path}.")
        return None

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"❌ Data file missing at {path}.")
        return pd.DataFrame()
