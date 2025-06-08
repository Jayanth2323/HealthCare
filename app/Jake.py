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
            raise RuntimeError(f"Font download failed → {download_exc}") from download_exc

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
        st.warning(f"⚠️ Could not add DejaVu TTF font: {font_exc}\nFalling back to Arial.")
        pdf.set_font("Arial", size=12)

    # Title & Summary
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary.strip())
    pdf.ln()

    # Gemini Recommendations
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
    model_path = os.path.join("models", "logistic_regression_pipeline.pkl")  # "models/logistic_regression_pipeline.pkl"
    if not os.path.isfile(model_path):
        st.error(f"❌ Model not found at '{model_path}'.")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_data():
    import os
    import pandas as pd

    # 1) Load the “real” dataset
    data_path = "data/cleaned_blood_data.csv"
    if not os.path.isfile(data_path):
        st.error(f"❌ Base data missing at '{data_path}'.")
        return pd.DataFrame()
    base_df = pd.read_csv(data_path)

    # 2) Try to load & append the synthetic CSV
    synthetic_path = "data/synthetic_patient_dataset.csv"
    if os.path.isfile(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        # only keep columns that both frames share
        common_cols = [c for c in synthetic_df.columns if c in base_df.columns]
        synthetic_df = synthetic_df[common_cols]
        combined = pd.concat([base_df, synthetic_df], ignore_index=True)
        st.success(f"✅ Loaded and appended {len(synthetic_df)} synthetic records.")
        return combined
    else:
        st.warning(f"⚠️ Could not find '{synthetic_path}'. Using base data only.")
        return base_df

model = load_model()
df = load_data()

if model is None or df.empty:
    st.warning("⚠️ Missing essential resources (model/data). Please verify setup.")
    st.stop()


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
age = st.sidebar.number_input("🎂 Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("👥 Gender", ["Male", "Female", "Other"])
blood_pressure = st.sidebar.slider("🩸 Blood Pressure (mmHg)", 80, 200, 120)
heart_rate = st.sidebar.slider("💓 Heart Rate (bpm)", 40, 180, 80)
cholesterol = st.sidebar.slider("🧪 Cholesterol (mg/dL)", 100, 400, 200)
hemoglobin = st.sidebar.slider("🩺 Hemoglobin (g/dL)", 8.0, 20.0, 14.0, step=0.1)
smoking_status = st.sidebar.selectbox("🚬 Smoking Status", ["Non-smoker", "Smoker"])
exercise_level = st.sidebar.selectbox("🏃 Exercise Level", ["Low", "Moderate", "High"])

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
currency_symbol = st.sidebar.selectbox("💱 Choose Currency", list(currency_options.keys()))
currency_range = currency_options[currency_symbol]
monetary = st.sidebar.slider(
    f"💸 Annual Healthcare Spending ({currency_symbol})",
    min_value=currency_range["min"],
    max_value=currency_range["max"],
    value=(currency_range["min"] + currency_range["max"]) // 2,
    step=currency_range["step"],
)

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
        file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

        if file_ext == ".csv":
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success("✅ CSV file uploaded successfully.")
                batch_data["Prediction"] = model.predict(batch_data)
                st.dataframe(batch_data)
                st.download_button(
                    "📥 Download Predictions",
                    batch_data.to_csv(index=False),
                    "batch_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"❌ Failed to process CSV: {e}")

        elif file_ext in [".txt", ".md"]:
            try:
                content = uploaded_file.read().decode("utf-8")
                st.text_area("📄 File Content", content, height=300)
            except Exception as e:
                st.error(f"❌ Could not read text file: {e}")

        elif file_ext == ".pdf":
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() or "" for page in reader.pages])
                st.text_area("📑 Extracted PDF Text", text or "No text found.", height=300)
            except Exception as e:
                st.error(f"❌ PDF extraction failed: {e}")

        elif file_ext == ".docx":
            try:
                from docx import Document

                doc = Document(uploaded_file)
                doc_text = "\n".join([para.text for para in doc.paragraphs])
                st.text_area("📝 Word Document Content", doc_text, height=300)
            except Exception as e:
                st.error(f"❌ Word document processing failed: {e}")

        else:
            st.warning(
                f"⚠️ File type '{file_ext}' not supported for processing. "
                "Please upload a supported format."
            )

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
                    "Based on the patient's profile, suggest:\n"
                    "- Likely cause of risk\n"
                    "- Recommended treatment or lifestyle changes\n"
                    "- Specialist referrals\n"
                    "- Rationale for the recommendation"
                )
                ai_response = chat.send_message(prompt)
                response_text = ai_response.text or ""
                st.markdown("### 🧠 Jake's Auto Analysis")
                st.success("✅ Jake's AI-driven treatment suggestions:")
                try:
                    st.markdown(response_text)
                except Exception:
                    st.code(response_text)
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
            st.metric("Predicted Risk Level", ["Low", "Medium", "High"][prediction])
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

            # SHAP Explainability
            st.markdown("#### 🧠 Model Explainability (SHAP)")
            with st.expander("Show SHAP values"):
                try:
                    preprocessed_input = model[:-1].transform(input_df)
                    explainer = shap.LinearExplainer(classifier, preprocessed_input)
                    shap_values = explainer(preprocessed_input)
                    st_shap(shap.plots.waterfall(shap_values[0]))
                except Exception as e:
                    st.warning(f"SHAP could not be generated: {e}")

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
                df["Frequency"].mean(),
                df["Monetary"].mean(),
                df["Time"].mean(),
            ]

            fig_patient = go.Figure()
            fig_patient.add_trace(
                go.Bar(x=patient_df["Metric"], y=patient_df["Value"], name="You")
            )
            fig_patient.add_trace(
                go.Scatter(
                    x=patient_df["Metric"],
                    y=avg,
                    mode="lines+markers",
                    name="Population Avg",
                )
            )
            fig_patient.update_layout(title="Your Metrics vs Population Average")
            st.plotly_chart(fig_patient, use_container_width=True)

    else:
        st.info("👈 Adjust sidebar inputs and click 'Generate Recommendation'.")

# === Tab 2: Data Intelligence ===
with tab2:
    st.subheader("📂 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sample Data")
        st.dataframe(df.head())
        st.markdown("#### Summary Statistics")
        st.dataframe(df.describe())

        st.markdown(
            """
        **Dataset Fields Overview:**
        - Hemoglobin
        - Blood Pressure
        - Heart Rate
        - Cholesterol
        - White Blood Cell Count
        - Glucose
        - Gender
        - Age
        - Smoking Status
        - Exercise Level
        - BMI
        - Blood Type
        """
        )

    with col2:
        st.markdown("#### Risk Class Distribution")
        fig_class = px.pie(
            df, names="Class", title="Risk Class Distribution", hole=0.3
        )
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("#### Missing Data Overview")
        st.dataframe(df.isnull().sum().to_frame("Missing Count"))

with tab3:
    st.subheader("📈 Model Performance & Insights")

    # Compute correlation matrix (numeric only)
    corr = df.corr(numeric_only=True)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_mask = corr.mask(mask)
    fig_corr = px.imshow(
        corr_mask,
        text_auto=".3f",
        aspect="equal",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation"
        )
    fig_corr.update_layout(xaxis_side="bottom")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # - Matplotlib heatmap (for download/export)
    fig_corr, ax = plt.subplots(figsize=(14, 10))
    
    # BMI vs Age
    if all(c in df.columns for c in ["age","bmi","diabetes"]):
        fig_sc = px.scatter(df, x="age", y="bmi", color="diabetes",
                            labels={"diabetes":"Diabetes"},
                            title="BMI vs Age by Diabetes")
        st.plotly_chart(fig_sc, use_container_width=True)

    # PCA
    num_df = df.select_dtypes(include="number").dropna()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(num_df))
    pca_df = pd.DataFrame(coords, columns=["PC1","PC2"])
    if "diabetes" in df.columns:
        pca_df["diabetes"] = df.loc[num_df.index, "diabetes"].values
        fig_p = px.scatter(pca_df, x="PC1", y="PC2", color="diabetes",
                        title="PCA Projection")
        st.plotly_chart(fig_p, use_container_width=True)
    
    # Create the heatmap (lower triangle only)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",  # same as your image
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.9},
        ax=ax
    )

    ax.set_title("Correlation Matrix of Health Features", fontsize=16)
    st.pyplot(fig_corr)

    # Optional: Export heatmap as PNG
    buf = BytesIO()
    fig_corr.savefig(buf, format="png")
    st.download_button(
        label="📥 Download Heatmap as PNG",
        data=buf.getvalue(),
        file_name="correlation_heatmap.png",
        mime="image/png"
    )

    # BMI vs Age by Diabetes Class
    st.markdown("#### 📉 BMI vs Age by Diabetes Class")
    if "bmi" in df.columns and "age" in df.columns and "diabetes" in df.columns:
        fig_bmi_age = px.scatter(
            df,
            x="age",
            y="bmi",
            color="diabetes",
            labels={"diabetes": "Diabetes Class"},
            title="BMI vs Age Colored by Diabetes Risk",
        )
        st.plotly_chart(fig_bmi_age, use_container_width=True)
    else:
        st.info("⚠️ Could not plot BMI vs Age: missing 'bmi', 'age' or 'diabetes' column.")

    # PCA Visualization
    st.markdown("#### 🧬 PCA: Patient Clusters")
    numeric_df = df.select_dtypes(include="number").dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)
    reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])

    if "diabetes" in df.columns:
        reduced_df["diabetes"] = df.loc[numeric_df.index, "diabetes"].values
        fig_pca = px.scatter(
            reduced_df,
            x="PC1",
            y="PC2",
            color="diabetes",
            title="PCA Projection of Patient Features"
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("⚠️ Cannot show PCA plot because the 'diabetes' column is missing.")

    st.markdown("#### Feature Distributions")
    feat = st.selectbox("Choose a feature:", ["Frequency", "Monetary", "Time"])
    fig_feat = px.box(
        df, x="Class", y=feat, color="Class", title=f"{feat} by Risk Class"
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("#### Pairwise Feature Plot")
    st.image("images/pairplot.png", caption="Pairwise Feature Analysis")

    st.markdown("#### Model Evaluation Report")
    try:
        y_true = df["Class"]
        y_pred = model.predict(df[["Frequency", "Monetary", "Time"]])
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred))
    except Exception as e:
        st.warning(f"Could not compute model evaluation metrics: {e}")

    st.markdown("#### Confusion Matrix")
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
                    chat_model = genai.GenerativeModel("gemini-1.5-flash-latest")
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
    st.markdown(
        """
    A cutting-edge healthcare recommendation system combining:
    - 🧠 Machine Learning (Logistic Regression)
    - 💬 Jake AI (Gemini)
    - 📊 Interactive Dashboards
    - 📄 SHAP Explainability + PDF Export

    **Developed by:** Jayanth | Full Stack Developer & AI Enthusiast  
    🔗 [GitHub Repository](https://github.com/Jayanth2323/HealthCare)
    """
    )
