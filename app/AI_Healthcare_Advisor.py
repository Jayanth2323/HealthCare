import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import shap
import streamlit.components.v1 as components
from dotenv import load_dotenv
from fpdf import FPDF

# === Load API Key ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=api_key)

# === Streamlit Setup ===
st.set_page_config(page_title="AI Healthcare Advisor", layout="wide", page_icon="🧠")

# === Load Model & Data ===
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/logistic_regression_pipeline.pkl")
    except FileNotFoundError:
        st.error("❌ Model not found at 'models/logistic_regression_pipeline.pkl'.")
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/cleaned_blood_data.csv")
    except FileNotFoundError:
        st.error("❌ Data file missing at 'data/cleaned_blood_data.csv'.")
        return pd.DataFrame()

model = load_model()
df = load_data()
if df.empty:
    st.warning("⚠️ No data loaded. Please check your dataset.")
    st.stop()

# === SHAP Visualization Helper ===
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

# === PDF Generator ===
def generate_pdf_report(health_summary, ai_response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary)
    pdf.ln()
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ai_response)
    output_path = "./data/health_report.pdf"
    pdf.output(output_path)
    return output_path

# === Recommendation Mapping ===
def generate_recommendation(pred_label):
    return {
        0: "✅ **Low Risk**\nMaintain your current healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ **Medium Risk**\nIncrease physical activity, monitor diet. Schedule a medical consultation.",
        2: "🚨 **High Risk**\nImmediate medical attention advised. Begin treatment under supervision.",
    }.get(pred_label, "❓ No recommendation available.")

# === Sidebar Inputs ===
st.sidebar.header("📝 Patient Profile")
frequency = st.sidebar.slider("📅 Visit Frequency (visits/year)", 0, 50, 5)
monetary = st.sidebar.slider("💸 Annual Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("⏳ Time Since Last Visit (months)", 0, 60, 12)

# === Main Page ===
st.title("🧠 AI-Driven Personalized Healthcare Advisor")
st.markdown("Empowering health decisions through machine intelligence.")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Recommendation Engine",
    "📊 Data Intelligence",
    "🔍 Model Insights",
    "🤖 AI Chat Assistant",
    "ℹ️ About"
])

# === Tab 1: Recommendation Engine ===
with tab1:
    st.subheader("Your Personalized Health Recommendation")

    # Upload patient dataset for batch prediction
    uploaded_file = st.file_uploader("📁 Upload CSV for batch processing:", type=["csv"])
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        batch_data["Prediction"] = model.predict(batch_data)
        st.dataframe(batch_data)
        st.download_button("📥 Download Predictions", batch_data.to_csv(index=False), "batch_predictions.csv")

    # Individual Recommendation
    if st.sidebar.button("💡 Generate Recommendation"):
        input_df = pd.DataFrame({"Frequency": [frequency], "Monetary": [monetary], "Time": [time]})
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = f"{probs[prediction]*100:.2f}%"
        recommendation = generate_recommendation(prediction)

        # Build health summary
        health_summary = f"""
        Patient Health Summary:
        - Risk Level: {['Low', 'Medium', 'High'][prediction]}
        - Confidence: {confidence}
        - Recommendation: {recommendation.replace('**', '').replace('✅', '').replace('⚠️', '').replace('🚨', '')}
        - Visit Frequency: {frequency}
        - Healthcare Spending: ${monetary}
        - Time Since Last Visit: {time} months
        """
        st.session_state["health_summary"] = health_summary

        # Gemini AI Treatment Suggestion
        with st.spinner("🔬 Analyzing treatment options using Jake AI..."):
            try:
                model_ai = genai.GenerativeModel("gemini-1.5-flash-latest")
                chat = model_ai.start_chat(history=[])
                prompt = (
                    f"{health_summary}\n\n"
                    "Based on the patient's profile, suggest:\n"
                    "- Likely cause of risk\n"
                    "- Recommended treatment or lifestyle changes\n"
                    "- Specialist referrals\n"
                    "- Rationale for the recommendation"
                )
                ai_response = chat.send_message(prompt)
                st.markdown("### 🧠 Jake's Auto Analysis")
                st.success("✅ AI-driven treatment suggestions:")
                st.markdown(ai_response.text)

                # PDF Export
                pdf_file = generate_pdf_report(health_summary, ai_response.text)
                with open(pdf_file, "rb") as f:
                    st.download_button("📄 Download Full PDF Report", f, file_name="health_report.pdf", mime="application/pdf")

                st.download_button("📥 Download Treatment Plan", ai_response.text, "treatment_plan.txt")
            except Exception as e:
                st.error(f"Jake AI Error: {e}")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Risk Level", ["Low", "Medium", "High"][prediction])
            st.metric("Prediction Confidence", confidence)
            st.markdown(recommendation)

            # Feature Importance
            try:
                classifier_step = next(step for step in model.named_steps if hasattr(model.named_steps[step], "coef_"))
                classifier = model.named_steps[classifier_step]
                importance = classifier.coef_[0]
            except Exception:
                importance = [0, 0, 0]

            fig_imp = px.bar(
                x=["Frequency", "Monetary", "Time"],
                y=importance,
                labels={"x": "Features", "y": "Importance"},
                title="Feature Contribution to Risk"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # SHAP Explainability (fixed)
            st.markdown("#### 🧠 Model Explainability (SHAP)")
            with st.expander("Show SHAP values"):
                try:
                    preprocessed_input = model[:-1].transform(input_df)
                    explainer = shap.LinearExplainer(classifier, preprocessed_input)
                    shap_values = explainer(preprocessed_input)
                    st_shap(shap.plots.waterfall(shap_values[0]))
                except Exception as e:
                    st.warning(f"SHAP could not be generated: {str(e)}")

        with col2:
            st.markdown("#### 📊 Your Health Snapshot")
            patient_df = pd.DataFrame({
                "Metric": ["Visit Frequency", "Spending", "Time Since Last Visit"],
                "Value": [frequency, monetary, time]
            })
            avg = [df["Frequency"].mean(), df["Monetary"].mean(), df["Time"].mean()]

            fig_patient = go.Figure()
            fig_patient.add_trace(go.Bar(x=patient_df["Metric"], y=patient_df["Value"], name="You"))
            fig_patient.add_trace(go.Scatter(x=patient_df["Metric"], y=avg, mode="lines+markers", name="Population Avg"))
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

    with col2:
        st.markdown("#### Risk Class Distribution")
        fig_class = px.pie(df, names="Class", title="Risk Class Distribution", hole=0.3)
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("#### Missing Data Overview")
        st.dataframe(df.isnull().sum().to_frame("Missing Count"))

# === Tab 3: Model Insights ===
with tab3:
    st.subheader("📈 Model Performance & Insights")
    st.markdown("#### Feature Correlation Heatmap")
    fig_corr = px.imshow(df.corr(numeric_only=True), title="Feature Correlation", color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### Feature Distributions")
    feat = st.selectbox("Choose a feature:", ["Frequency", "Monetary", "Time"])
    fig_feat = px.box(df, x="Class", y=feat, color="Class", title=f"{feat} by Risk Class")
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("#### Pairwise Feature Plot")
    st.image("images/pairplot.png", caption="Pairwise Feature Analysis")

# === Tab 4: AI Chat Assistant ===
with tab4:
    st.subheader("🤖 AI Chat Assistant")
    st.markdown("Ask personalized health questions and get dynamic responses from Jake AI.")

    user_input = st.text_area("💬 Enter your question:")
    if st.button("🚀 Ask AI"):
        with st.spinner("Jake is thinking..."):
            try:
                chat_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                chat = chat_model.start_chat(history=[])
                context = st.session_state.get("health_summary", "")
                prompt = f"{context}\n\nPatient's Question: {user_input}"
                response = chat.send_message(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI error: {e}")

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
