import os
import uuid
import urllib.request
import logging
import numpy as np
# from sklearn.pipeline import Pipeline
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
from PyPDF2 import PdfReader
from docx import Document
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from fpdf import FPDF
from babel.numbers import format_currency  # for currency formatting

# === Font Bootstrap Helpers ===
FONT_DIR = "fonts"
FONT_NAME = "DejaVuSans.ttf"
FONT_PATH = os.path.join(FONT_DIR, FONT_NAME)

# CORRECTED RAW_URL: use raw.githubusercontent.com instead of github.com/raw
# Updated RAW_URL as the previous one was returning 404
RAW_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/main/ttf/DejaVuSans.ttf"


def _is_valid_ttf(path: str) -> bool:
    """
    Quick sanity-check: TrueType files typically begin with 0x00010000 (uint32) or b'true'.
    Returns True if the file exists and starts with a valid TTF header.
    """
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as fh:
        tag = fh.read(4)
    return tag in (b"\x00\x01\x00\x00", b"true")


def bootstrap_font() -> str:
    """
    Ensure that FONT_PATH points to a valid DejaVuSans.ttf.
    If missing or invalid, download from RAW_URL.
    Raises RuntimeError if download/validation fails.
    Returns the path to a valid TTF file.
    """
    os.makedirs(FONT_DIR, exist_ok=True)

    # If the file doesn’t exist or fails the header check, attempt to download
    if not _is_valid_ttf(FONT_PATH):
        try:
            logging.info("Downloading DejaVuSans.ttf …")
            urllib.request.urlretrieve(RAW_URL, FONT_PATH)
        except Exception as download_exc:
            raise RuntimeError(
                f"Font download failed → {download_exc}"
            ) from download_exc

        # Re-check after download
        if not _is_valid_ttf(FONT_PATH):
            raise RuntimeError("Downloaded file is not a valid TrueType font.")

    return FONT_PATH


# === Generate PDF Report ===
def generate_pdf_report(health_summary: str, ai_response: str) -> str:
    """
    Generates a PDF report containing the provided health_summary and AI recommendations.
    Uses DejaVuSans.ttf for Unicode support. Returns the path to the generated PDF file,
    or an empty string if generation failed.
    """
    try:
        font_path = bootstrap_font()
    except Exception as font_boot_exc:
        st.error(f"❌ Unicode font bootstrap failed: {font_boot_exc}")
        return ""

    pdf = FPDF()
    pdf.add_page()

    # Add DejaVu font (Unicode-capable)
    try:
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception as font_exc:
        st.warning(
            f"⚠️ Could not add DejaVu TTF font: {font_exc}\nFalling back to Arial."
        )
        pdf.set_font("Arial", size=12)

    # Title & Summary
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary.strip())
    pdf.ln()

    # Gemini Recommendations
    try:
        pdf.set_font("DejaVu", "B", size=12)
    except BaseException:
        pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)

    try:
        pdf.set_font("DejaVu", size=12)
    except BaseException:
        pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ai_response.strip())

    # Persist PDF with a unique filename
    os.makedirs("data", exist_ok=True)
    filename = os.path.join("data", f"health_report_{uuid.uuid4().hex}.pdf")

    pdf.output(filename)
    return filename
    unique_id = str(uuid.uuid4())
    output_path = os.path.join("data", f"health_report_{unique_id}.pdf")
    try:
        pdf.output(output_path)
    except Exception as export_exc:
        st.error(f"❌ Unable to save PDF report: {export_exc}")
        return ""

    return output_path


# === Load API Key ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file."
    )
    st.stop()
genai.configure(api_key=api_key)

# === Streamlit Setup ===
st.set_page_config(
    page_title="JakeAI Healthcare Advisor", layout="wide", page_icon="🧠"
)


# === Currency Formatter ===
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
    model_path = os.path.join(
        "models", "logistic_regression_pipeline.pkl"
    )  # "models/logistic_regression_pipeline.pkl"
    if not os.path.isfile(model_path):
        st.error(f"❌ Model not found at '{model_path}'.")
        return None
    return joblib.load(model_path)


@st.cache_data
def load_data():
    # 1) Load the "real" dataset
    base_path = "data/cleaned_blood_data.csv"
    base_df = pd.DataFrame()
    if not os.path.isfile(base_path):
        st.error(f"❌ Base data missing at '{base_path}'.")
        return pd.DataFrame()

    else:
        base_df = pd.read_csv(base_path)

    # 2) Load the synthetic dataset if present
    synthetic_path = "data/synthetic_patient_dataset.csv"
    synthetic_df = pd.DataFrame()

    if os.path.isfile(synthetic_path):
        try:
            # Load synthetic data
            synthetic_df = pd.read_csv(synthetic_path)

            # Normalize column names to lowercase
            synthetic_df.columns = synthetic_df.columns.str.lower()

            # Don't filter by base_df columns - keep all synthetic columns
            # Only filter if we need to combine datasets (which we're not
            # using)
        except Exception as e:
            st.error(f"❌ Error loading synthetic data: {e}")
    else:
        st.warning(
            f"⚠️ Synthetic data missing at '{synthetic_path}'. Using base data only."
        )
        synthetic_df = pd.DataFrame()

    return base_df, synthetic_df  # Return full synthetic dataset


model = load_model()
base_df, synthetic_df = load_data()

# Remove the combined_df creation since we're not using it
# if model is None or base_df.empty:
#     st.warning("⚠️ Missing essential resources (model/data). Please verify setup.")
#     st.stop()

# Create combined dataset for visualizations
# if synthetic_df.empty:
#     combined_df = base_df.copy()
# else:
#     common_cols = base_df.columns.intersection(synthetic_df.columns)
#     combined_df = pd.concat([
#         base_df[common_cols],
#         synthetic_df[common_cols]
#     ], ignore_index=True)

# if model is None or base_df.empty:
#     st.warning("⚠️ Missing essential resources (model/data). Please verify setup.")
#     st.stop()

# Prepare combined data
# common_cols = base_df.columns
# if not synthetic_df.empty:
#     common_cols = common_cols.intersection(synthetic_df.columns)
# combined_df = pd.concat([base_df[common_cols], synthetic_df.get(common_cols, pd.DataFrame())], ignore_index=True)


# === SHAP Visualization Helper ===
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)


# === Recommendation Mapping ===
def generate_recommendation(pred_label: int) -> str:
    return {
        0: "✅ **Low Risk**\nMaintain your current healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ **Medium Risk**\nIncrease physical activity, monitor diet. Schedule a medical consultation.",
        2: "🚨 **High Risk**\nImmediate medical attention advised. Begin treatment under supervision.",
    }.get(pred_label, "❓ No recommendation available.")


# === Sidebar Inputs ===
st.sidebar.header("📝 Patient Profile")
frequency = st.sidebar.slider("📅 Visit Frequency (visits/year)", 0, 50, 5)
age = st.sidebar.number_input("🎂 Age", min_value=1, max_value=120, value=22)
gender = st.sidebar.selectbox("👥 Gender", ["Male", "Female", "Other"])
blood_pressure = st.sidebar.slider("🩸 Blood Pressure (mmHg)", 80, 200, 120)
heart_rate = st.sidebar.slider("💓 Heart Rate (bpm)", 40, 180, 80)
cholesterol = st.sidebar.slider("🧪 Cholesterol (mg/dL)", 100, 400, 200)
hemoglobin = st.sidebar.slider(
    "🩺 Hemoglobin (g/dL)", 8.0, 20.0, 14.0, step=0.1)
smoking_status = st.sidebar.selectbox(
    "🚬 Smoking Status", ["Non-smoker", "Smoker"])
exercise_level = st.sidebar.selectbox(
    "🏃 Exercise Level", [
        "Low", "Moderate", "High"])

# Currency and default ranges based on economic context
currency_options = {
    "$": {"label": "USD", "min": 1000, "max": 200000, "step": 1000},
    "€": {"label": "Euro", "min": 1000, "max": 180000, "step": 1000},
    "₹": {"label": "INR", "min": 5000, "max": 2000000, "step": 5000},
    "£": {"label": "GBP", "min": 1000, "max": 150000, "step": 1000},
    "¥": {"label": "JPY", "min": 100000, "max": 10000000, "step": 100000},
    "₩": {"label": "KRW", "min": 1000000, "max": 50000000, "step": 1000000},
    "₽": {"label": "RUB", "min": 10000, "max": 2000000, "step": 10000},
    "₺": {"label": "TRY", "min": 1000, "max": 300000, "step": 1000},
    "AED": {"label": "Dirham", "min": 2000, "max": 500000, "step": 2000},
}

# Sidebar currency selection
currency_symbol = st.sidebar.selectbox(
    "💱 Choose Currency", list(currency_options.keys())
)
currency_range = currency_options[currency_symbol]
monetary = st.sidebar.slider(
    f"💸 Annual Healthcare Spending ({currency_symbol})",
    min_value=currency_range["min"],
    max_value=currency_range["max"],
    value=(currency_range["min"] + currency_range["max"]) // 2,
    step=currency_range["step"],
)

try:
    formatted_spending = format_currency(monetary, currency_symbol)
except BaseException:
    formatted_spending = f"{currency_symbol}{monetary}"

time = st.sidebar.slider("⏳ Time Since Last Visit (months)", 0, 60, 12)
formatted_spending = format_currency(monetary, currency_symbol)

# === Main Page ===
st.title("🧠 Jake-Driven Personalized Healthcare Advisor")
st.markdown("Empowering health decisions through machine intelligence.")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧬 Recommendation Engine",
        "📊 Data Intelligence",
        "🔍 Model Insights",
        "🤖 AI Chat Assistant",
        "ℹ️ About",
    ]
)

# === Tab 1: Recommendation Engine ===
with tab1:
    st.subheader("Your Personalized Health Recommendation")

    # Upload patient dataset for batch prediction
    uploaded_file = st.file_uploader(
        "📁 Upload a file for processing (CSV/TXT/PDF/DOCX)", type=None
    )

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_ext == ".csv":
                batch_data = pd.read_csv(uploaded_file)
                st.success("✅ CSV file uploaded successfully.")

                # Check if required columns exist
                required_cols = ["Frequency", "Monetary", "Time"]
                if all(col in batch_data.columns for col in required_cols):
                    batch_data["Prediction"] = model.predict(
                        batch_data[required_cols])
                    st.success("✅ CSV processed.")
                    st.dataframe(batch_data)
                    st.download_button(
                        "📥 Download Predictions",
                        batch_data.to_csv(index=False),
                        "batch_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    missing = [
                        col for col in required_cols if col not in batch_data.columns
                    ]
                    st.error(
                        f"❌ Missing required columns: {', '.join(missing)}")

            elif file_ext in [".txt", ".md"]:
                content = uploaded_file.read().decode("utf-8")
                st.text_area("📄 File Content", content, height=300)
            elif file_ext == ".pdf":
                reader = PdfReader(uploaded_file)
                text = "".join(
                    [page.extract_text() or "" for page in reader.pages])
                st.text_area(
                    "📑 Extracted PDF Text", text or "No text found.", height=300
                )

            elif file_ext == ".docx":
                doc = Document(uploaded_file)
                doc_text = "\n".join([para.text for para in doc.paragraphs])
                st.text_area("📝 Word Document Content", doc_text, height=300)

            else:
                st.warning(
                    f"⚠️ File type '{file_ext}' not supported for processing. "
                    "Please upload a supported format."
                )
        except Exception as e:
            st.error(f"❌ Word document processing failed: {e}")

        # except Exception as e:
        #     st.error(f"❌ PDF extraction failed: {e}")

    # Individual Recommendation
    if st.sidebar.button("💡 Generate Recommendation"):
        # 1) Build DataFrame for prediction
        input_df = pd.DataFrame(
            {
                "Frequency": [frequency],
                "Monetary": [monetary],
                "Time": [time],
            }
        )

        # 2) Predict
        try:
            prediction = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            confidence = f"{probs[prediction] * 100:.2f}%"
            recommendation = generate_recommendation(prediction)
        except Exception as e:
            st.error(f"❌ Model prediction failed: {e}")
            st.stop()

        # 3) Build health summary
        rec_clean = (
            str(recommendation)
            .replace("**", "")
            .replace("✅", "")
            .replace("⚠️", "")
            .replace("🚨", "")
        )
        health_summary = (
            "Patient Health Summary:\n"
            f"- Risk Level: {['Low', 'Medium', 'High'][prediction]}\n"
            f"- Confidence: {confidence}\n"
            f"- Recommendation: {rec_clean}\n"
            f"- Visit Frequency: {frequency}\n"
            f"- Healthcare Spending: {format_currency(monetary, currency_symbol)}\n"
            f"- Time Since Last Visit: {time} months\n"
        )
        st.session_state["health_summary"] = health_summary

        # 4) Gemini AI Treatment Suggestion
        with st.spinner("🔬 Analyzing treatment options using Jake..."):
            try:
                chat_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                chat = chat_model.start_chat(history=[])
                prompt = (
                    f"{health_summary.strip()}\n\n"
                    "Based on the patient's complete profile and identified health risks, provide the following details:\n"
                    "1. Likely cause(s) of the health risk.\n"
                    "2. Recommended lifestyle, dietary, or behavioral changes.\n"
                    "3. A detailed and specific treatment plan tailored to the condition.\n"
                    "4. Exact **medication names (generic or brand)** that are commonly prescribed for this condition globally,\n"
                    "   including the dosage form (e.g., tablet, capsule, injection), standard dosage range (if available),\n"
                    "   and any important administration guidelines.\n"
                    "5. Specialist doctor recommendations (if any).\n"
                    "6. Rationale behind the treatment and medicine choice.\n"
                    "7. Include warnings or contraindications for the mentioned medications.\n\n"
                    "Only suggest medicines that are clinically approved and widely used. If no medicine is applicable, state clearly."
                )
                ai_response = chat.send_message(prompt)
                response_text = ai_response.text or ""
                st.markdown("### 🧠 Jake's Auto Analysis")
                st.success("✅ Jake's AI-driven treatment suggestions:")
                try:
                    st.markdown(response_text)
                except Exception:
                    st.code(response_text)

                    st.warning(
                        "⚠️ Disclaimer: This is an AI-generated suggestion. Always consult a certified medical professional before starting any medication."
                    )
            except Exception as e:
                st.error("❌ Jake AI failed to process the treatment plan.")
                st.exception(e)
                response_text = ""

        # 5) PDF Export
        if response_text:
            pdf_file = generate_pdf_report(health_summary, response_text)
            if pdf_file and os.path.isfile(pdf_file):
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="📄 Download Full PDF Report",
                        data=f,
                        file_name=os.path.basename(pdf_file),
                        mime="application/pdf",
                    )

            st.download_button(
                label="📥 Download Treatment Plan",
                data=response_text,
                file_name="treatment_plan.txt",
                mime="text/plain",
            )

        # 6) Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Risk Level", [
                    "Low", "Medium", "High"][prediction])
            st.metric("Prediction Confidence", confidence)
            st.markdown(recommendation)

            # Feature Importance
            try:
                classifier_step = next(
                    step
                    for step in model.named_steps
                    if hasattr(model.named_steps[step], "coef_")
                )
                classifier = model.named_steps[classifier_step]
                importance = classifier.coef_[0]
            except Exception:
                importance = [0, 0, 0]

            fig_imp = px.bar(
                x=["Frequency", "Monetary", "Time"],
                y=importance,
                labels={"x": "Features", "y": "Importance"},
                title="Feature Contribution to Risk",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # # === SHAP Explainability (dynamic step detection with diagnostics & proper background) ===
            # st.markdown("#### 🧠 Model Explainability (SHAP)")
            # with st.expander("Show SHAP values"):
            #     try:
            #         # 1) Identify classifier and preprocessor from pipeline
            #         if isinstance(model, Pipeline):
            #             classifier = model.steps[-1][1]
            #             preprocessor = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
            #         else:
            #             if hasattr(model, "predict_proba"):
            #                 classifier, preprocessor = model, None
            #             else:
            #                 raise ValueError("Model is not a Pipeline and has no predict_proba.")

            #         if not hasattr(classifier, "predict_proba"):
            #             raise ValueError("Detected classifier has no predict_proba method.")

            #         # 2) Inspect input_df
            #         st.write("🔍 input_df shape:", input_df.shape)
            #         st.write("🔍 input_df columns:", input_df.columns.tolist())

            #         # Transform the single-row input
            #         if preprocessor is not None:
            #             X_pre = preprocessor.transform(input_df)
            #             st.write("🔍 Transformed input X_pre shape:", X_pre.shape)
            #         else:
            #             X_pre = input_df.values
            #             st.write("🔍 Raw input array X_pre shape:", X_pre.shape)

            #         if X_pre.ndim != 2:
            #             raise ValueError(f"Expected transformed input to be 2D, but got shape: {X_pre.shape}")

            #         # 3) Prepare background from training data (base_df)
            #         if "base_df" in globals() and not base_df.empty:
            #         # sample up to 100 rows
            #             bg_df = base_df[input_df.columns].sample(n=min(100, len(base_df)), random_state=42)
            #             if preprocessor is not None:
            #                 background = preprocessor.transform(bg_df)
            #             else:
            #                 background = bg_df.values
            #             st.write("🔍 Background after transform shape:", background.shape)
            #         else:
            #             raise ValueError("base_df is missing or empty; please supply a proper background dataset for SHAP.")

            #         if background.ndim != 2:
            #             raise ValueError(f"Expected background array to be 2D, but got shape: {background.shape}")

            #         # 4) Create explainer & 5) compute SHAP values
            #         explainer = shap.KernelExplainer(classifier.predict_proba, background)
            #         shap_vals = explainer.shap_values(X_pre, nsamples=100)

            #         # 6) Pick the right class slice
            #         if isinstance(shap_vals, list):
            #         # binary or multiclass
            #             if len(shap_vals) == 2:
            #                 shap_array = shap_vals[1]
            #                 st.write("🔍 Explaining positive-class SHAP values (index 1)")
            #             else:
            #                 idx = int(classifier.predict(X_pre)[0])
            #                 shap_array = shap_vals[idx]
            #                 st.write(f"🔍 Explaining class index {idx} SHAP values")
            #         elif isinstance(shap_vals, np.ndarray):
            #             if shap_vals.ndim == 2:
            #                 shap_array = shap_vals
            #                 st.write("🔍 shap_vals is 2D ndarray, using directly")
            #             elif shap_vals.ndim == 3:
            #                 n_s, n_f, n_c = shap_vals.shape
            #                 if n_c == 2:
            #                     shap_array = shap_vals[:, :, 1]
            #                     st.write("🔍 shap_vals is 3D ndarray, picking positive-class axis=2 index 1")
            #                 else:
            #                     idx = int(classifier.predict(X_pre)[0])
            #                     shap_array = shap_vals[:, :, idx]
            #                     st.write(f"🔍 shap_vals is 3D ndarray, picking predicted-class index {idx}")
            #             else:
            #                 raise ValueError(f"Unexpected shap_vals ndarray ndim={shap_vals.ndim}")
            #         else:
            #             raise ValueError(f"Unexpected shap_vals type: {type(shap_vals)}")

            #         st.write("🔍 shap_array shape:", shap_array.shape)
            #         if shap_array.ndim != 2:
            #             raise ValueError(f"Expected shap_array to be 2D, but got: {shap_array.shape}")

            #         # 7) Uniformity / plotting
            #         n_samples, n_features = shap_array.shape
            #         feature_names = input_df.columns.tolist()

            #         if n_samples > 1:
            # # multiple inputs → compare rows
            #             if np.allclose(shap_array, shap_array[0], atol=1e-8):
            #                 st.info("ℹ️ SHAP values are uniform across the samples — no variation in feature influence.")
            #                 st.dataframe(pd.DataFrame(shap_array, columns=feature_names))
            #             else:
            #                 fig, ax = plt.subplots()
            #                 shap.summary_plot(
            #                 shap_array,
            #                 features=X_pre,
            #                 feature_names=feature_names,
            #                 plot_type="bar",
            #                 show=False,
            #                 alpha=0.8
            #             )
            #             ax.set_title("SHAP Feature Contribution")
            #             st.pyplot(fig)

            #         else:
            #             # single input → compare feature values
            #             row = shap_array[0]
            #             if np.allclose(row, np.full_like(row, row[0]), atol=1e-8):
            #                 st.info("ℹ️ SHAP values for all features are nearly identical for this input.")
            #                 st.dataframe(pd.DataFrame([row], columns=feature_names))
            #             else:
            #                 fig, ax = plt.subplots()
            #                 shap.summary_plot(
            #                 row.reshape(1, -1),
            #                 features=X_pre,
            #                 feature_names=feature_names,
            #                 plot_type="bar",
            #                 show=False,
            #                 alpha=0.8
            #             )
            #             ax.set_title("SHAP Feature Contribution")
            #             st.pyplot(fig)

            #     except Exception as e:
            #         st.warning(f"⚠️ SHAP could not be generated: {e}")


        with col2:
            st.markdown("#### 📊 Your Health Snapshot")
            patient_df = pd.DataFrame(
                {
                    "Metric": [
                        "Visit Frequency",
                        f"Spending ({currency_symbol})",
                        "Time Since Last Visit",
                    ],
                    "Value": [frequency, monetary, time],
                }
            )
            avg = [
                base_df["Frequency"].mean(),
                base_df["Monetary"].mean(),
                base_df["Time"].mean(),
            ]

            fig_patient = go.Figure()
            fig_patient.add_trace(
                go.Bar(
                    x=patient_df["Metric"],
                    y=patient_df["Value"],
                    name="You")
            )
            fig_patient.add_trace(
                go.Scatter(
                    x=patient_df["Metric"],
                    y=avg,
                    mode="lines+markers",
                    name="Population Avg",
                )
            )
            fig_patient.update_layout(
                title="Your Metrics vs Population Average")
            st.plotly_chart(fig_patient, use_container_width=True)

    else:
        st.info("👈 Adjust sidebar inputs and click 'Generate Recommendation'.")

# === Tab 2: Data Intelligence ===
with tab2:
    st.subheader("📂 Dataset Overview")

    # === 🔹 ROW 1: Dataset Preview + Risk Class ===
    st.markdown("### 🔍 Data Snapshot")
    row1_col1, row1_col2 = st.columns([1.5, 1])  # Wider data table

    with row1_col1:
        st.markdown("#### 🧾 Sample Data")
        st.dataframe(base_df.head(), use_container_width=True)

        st.markdown("#### 📊 Summary Statistics")
        st.dataframe(base_df.describe(), use_container_width=True)

        st.markdown("#### 🧬 Dataset Fields")
        st.markdown("""
        - Hemoglobin | Blood Pressure | Heart Rate
        - Cholesterol | White Blood Cell Count | Glucose
        - Gender | Age | Smoking Status
        - Exercise Level | BMI | Blood Type
        """)

    with row1_col2:
        st.markdown("#### 🩺 Risk Class Distribution")
        if "Class" in base_df.columns:
            fig_class = px.pie(
                base_df, names="Class", title="Risk Class Distribution", hole=0.3
            )
            fig_class.update_layout(height=300)
            st.plotly_chart(fig_class, use_container_width=True)
        else:
            st.warning("⚠️ 'Class' column not available")
        st.markdown("")

    st.divider()

    # === 🔹 ROW 2: Synthetic Gender & Diabetes ===
    st.markdown("### 🧠 Synthetic Data Insights")
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### 👥 Gender Distribution")
        if synthetic_df.empty:
            st.warning("⚠️ Synthetic dataset is empty.")
        elif "gender" not in synthetic_df.columns:
            st.warning("⚠️ 'gender' column not found.")
            st.code(synthetic_df.columns.tolist())
        else:
            synthetic_df["gender_label"] = synthetic_df["gender"].apply(
                lambda x: "Male" if x >= 0.5 else "Female"
            )
            gender_counts = synthetic_df["gender_label"].value_counts()

            if not gender_counts.empty:
                fig_gender = px.pie(
                    base_df,
                    names=gender_counts.index,
                    values=gender_counts.values,
                    title="Gender Distribution",
                    hole=0.3,
                )
                fig_gender.update_layout(height=300)
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.warning("⚠️ No gender data after processing")
            st.markdown("**🔢 Gender Summary**")
            st.dataframe(
                synthetic_df["gender"].describe().to_frame(), use_container_width=True
            )

    with row2_col2:
        st.markdown("#### 🧪 Diabetes Distribution")
        if synthetic_df.empty:
            st.warning("⚠️ Synthetic dataset is empty.")
        elif "diabetes" not in synthetic_df.columns:
            st.warning("⚠️ 'diabetes' column not found.")
            st.code(synthetic_df.columns.tolist())
        else:
            synthetic_df["diabetes_label"] = synthetic_df["diabetes"].apply(
                lambda x: "Diabetic" if x >= 1.5 else "Non-Diabetic"
            )
            diabetes_counts = synthetic_df["diabetes_label"].value_counts()

            if not diabetes_counts.empty:
                fig_diabetes = px.pie(
                    names=diabetes_counts.index,
                    values=diabetes_counts.values,
                    title="Diabetes Distribution",
                    hole=0.3,
                )
                fig_diabetes.update_layout(height=300)
                st.plotly_chart(fig_diabetes, use_container_width=True)
            else:
                st.warning("⚠️ No diabetes data after processing")

            st.markdown("**🔢 Diabetes Summary**")
            st.dataframe(
                synthetic_df["diabetes"].describe().to_frame(), use_container_width=True
            )

    # 🔎 Debug Info (optional toggle)
    with st.expander("🛠️ Synthetic Data Debug Info"):
        st.code(f"Shape: {synthetic_df.shape}")
        st.code(f"Columns: {synthetic_df.columns.tolist()}")
        if not synthetic_df.empty:
            st.dataframe(synthetic_df.head(3), use_container_width=True)

    # === 🔹 ROW 3 : Time Since Last Visit ===
    st.markdown("### 🕰️ Time Since Last Visit")
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        if "Time" in base_df.columns:
            st.markdown("#### Time Since Last Visit Distribution")
            fig_time = px.histogram(
                base_df, x="Time", title="Time Since Last Visit (months)", nbins=20
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("⚠️ 'Time' column not available for distribution plot.")

with tab3:
    st.subheader("📈 Model Performance & Insights")

    # Load the dataset
    base_df = pd.read_csv("./data/synthetic_patient_dataset.csv")

    # Normalize column names
    base_df.columns = base_df.columns.str.lower()
    if "class" in base_df.columns:
        base_df.rename(columns={"class": "diabetes"}, inplace=True)

    # 1️⃣ Correlation heatmap
    corr = base_df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig_corr, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.9},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Health Features", fontsize=16)

    corr_masked = corr.where(~mask)  # upper-triangle → NaN

    # Build interactive Plotly heatmap
    fig = px.imshow(
        corr_masked,
        text_auto=".3f",
        # rocket_r
        color_continuous_scale=px.colors.sequential.Inferno_r[::-1],
        zmin=-1,
        zmax=1,
        labels=dict(x="", y="", color="corr"),
        width=1400,
        height=900,
    )
    fig.update_layout(
        title="Correlation Matrix of Health Features",
        xaxis_side="bottom",
        font=dict(size=14),
        margin=dict(l=100, r=100, t=100, b=100),
    )
    # Rotate x-axis labels to match your style
    fig.update_xaxes(tickangle=45, tickfont=dict(size=12))
    fig.update_yaxes(tickfont=dict(size=12), autorange="reversed")

    # Show interactive figure
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Download button for the heatmap
    buf = BytesIO()
    # generate the static one for PNG download
    fig_static, ax = plt.subplots(figsize=(24, 14))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.9},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Health Features", fontsize=16)
    fig_static.savefig(buf, format="png", bbox_inches="tight")
    st.download_button(
        label="📥 Download Heatmap as PNG",
        data=buf.getvalue(),
        file_name="correlation_heatmap.png",
        mime="image/png",
    )

    # 2️⃣ Live Seaborn countplot for Gender vs Diabetes

    # Load and preprocess data
    base_df = pd.read_csv("./data/synthetic_patient_dataset.csv")
    base_df_o = pd.read_csv("./data/cleaned_blood_data.csv")

    # 1️⃣ Categorize gender (Assuming values near 0 = Female, 1 = Male)
    base_df["gender_cat"] = base_df["gender"].apply(
        lambda x: "Male" if x >= 0.5 else "Female"
    )

    # 2️⃣ Categorize diabetes (Assuming a threshold for diagnosis, e.g., >=1.5
    # is diabetic)
    base_df["diabetes_cat"] = base_df["diabetes"].apply(
        lambda x: "Diabetic" if x >= 1.5 else "Non-Diabetic"
    )

    # 3️⃣ Count Plot for Gender vs Diabetes
    st.markdown("#### 👥 Gender Distribution by Diabetes Class")

    fig_gender, ax_gender = plt.subplots(figsize=(8, 6))

    sns.countplot(
        data=base_df,
        x="gender_cat",
        hue="diabetes_cat",
        palette=sns.color_palette("Set2", 2),
        ax=ax_gender,
    )

    # Enhance visuals
    ax_gender.set_xlabel("Gender", fontsize=12)
    ax_gender.set_ylabel("Count", fontsize=12)
    ax_gender.set_title("Diabetes Distribution by Gender", fontsize=14)
    ax_gender.legend(title="Diabetes", title_fontsize=11, fontsize=10)
    ax_gender.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    # Show the updated figure
    st.pyplot(fig_gender)

    # BMI vs Age colored by Diabetes
    if all(c in base_df.columns for c in ["age", "bmi", "diabetes"]):
        st.markdown("#### 📉 BMI vs Age by Diabetes Class")
        fig_bmi_age = px.scatter(
            base_df,
            x="age",
            y="bmi",
            color="diabetes",
            labels={"diabetes": "Diabetes Class"},
            title="BMI vs Age Colored by Diabetes Risk",
        )
        st.plotly_chart(fig_bmi_age, use_container_width=True)
    else:
        st.info(
            "⚠️ Could not plot BMI vs Age: missing 'bmi', 'age' or 'diabetes' column."
        )

    st.markdown("#### 🧬 PCA: Patient Clusters")
    # PCA
    numeric_df = base_df.select_dtypes(include="number").dropna()
    if "diabetes" in base_df.columns and not numeric_df.empty:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaled)
        reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        reduced_df["diabetes"] = base_df.loc[numeric_df.index, "diabetes"].values

        fig_pca = px.scatter(
            reduced_df,
            x="PC1",
            y="PC2",
            color="diabetes",
            title="PCA Projection of Patient Features",
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("⚠️ Cannot show PCA plot because required data is missing.")

    st.markdown("#### Feature Distributions by Risk Class")
    if "Class" in base_df_o.columns:
        feature_options = [
            col for col in ["Frequency", "Monetary", "Time"] if col in base_df_o.columns
        ]

        if feature_options:
            feat = st.selectbox("Choose a feature:", feature_options)
            fig_box = px.box(
                base_df_o,
                x="Class",
                y=feat,
                color="Class",
                title=f"{feat} Distribution by Risk Class",
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No suitable features available for distribution analysis")
    else:
        st.warning("'Class' column missing for distribution analysis")

    st.markdown("#### 🧬 Feature Relationships")
    st.image("./images/pairplot.png", caption="Pairwise Feature Distribution")

    st.markdown("#### Model Evaluation Report")
    # Check for required columns including 'Class'
    # required_cols = ]
    if all(
        col in base_df_o.columns for col in ["Frequency", "Monetary", "Time", "Class"]
    ):
        try:
            X = base_df_o[["Frequency", "Monetary", "Time"]]
            y_true = base_df_o["Class"]
            y_pred = model.predict(X)

            st.text("Classification Report:")
            report = classification_report(y_true, y_pred)
            st.text(report)
        except Exception as e:
            st.warning(f"Could not compute model evaluation metrics: {e}")
    else:
        st.warning("Required columns for evaluation missing in dataset")

    st.markdown("#### Confusion Matrix")
    if y_true is not None and y_pred is not None:
        try:
            conf_matrix = confusion_matrix(y_true, y_pred)
            fig_conf = px.imshow(
                conf_matrix,
                text_auto=True,
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render confusion matrix: {e}")
    else:
        st.warning("Confusion matrix unavailable - evaluation data missing")

# === Tab 4: AI Chat Assistant ===
with tab4:
    st.subheader("🤖 Jake's Chat Assistant")
    st.markdown(
        "Ask personalized health questions and get dynamic responses from Jake AI."
    )

    user_input = st.text_area("💬 Enter your question:")
    if st.button("🚀 Ask Jake AI"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        else:
            with st.spinner("Jake is thinking..."):
                try:
                    chat_model = genai.GenerativeModel(
                        "gemini-1.5-flash-latest")
                    chat = chat_model.start_chat(history=[])
                    context = st.session_state.get("health_summary", "")
                    prompt = f"{context}\n\nPatient's Question: {user_input.strip()}"
                    response = chat.send_message(prompt)
                    st.markdown(response.text or "")
                except Exception as e:
                    st.error("❌ Jake AI failed to process the question.")
                    st.exception(e)

# === Tab 5: About ===
with tab5:
    st.subheader("ℹ️ About This App")
    st.markdown("""
    A cutting-edge healthcare recommendation system combining:
    - 🧠 Machine Learning (Logistic Regression)
    - 💬 Jake AI (Gemini)
    - 📊 Interactive Dashboards
    - 📄 SHAP Explainability + PDF Export

    **Developed by:** Jayanth | Full Stack Developer & AI Enthusiast
    🔗 [GitHub Repository](https://github.com/Jayanth2323/HealthCare)
    """)
