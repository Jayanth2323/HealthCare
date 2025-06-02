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
from modules.utils import st_shap
from modules.utils import generate_pdf_report, clean_text
from sklearn.metrics import classification_report
import tempfile

# === Streamlit Setup ===
st.set_page_config(page_title="AI Healthcare Advisor", layout="wide", page_icon="🧠")

# === Load API Key ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=api_key)

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
if model is None or df.empty:
    st.stop()

# === Sidebar ===
st.sidebar.header("📝 Patient Profile")
frequency = st.sidebar.slider("🗕️ Visit Frequency (visits/year)", 0, 50, 5)
monetary = st.sidebar.slider("💸 Annual Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("⏳ Time Since Last Visit (months)", 0, 60, 12)

# === Main Interface ===
st.title("🧠 AI-Driven Personalized Healthcare Advisor")
st.markdown("Empowering health decisions through machine intelligence.")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧬 Recommendation Engine",
    "📈 Data Intelligence",
    "🔍 Model Insights",
    "🤖 AI Chat Assistant",
    "ℹ️ About",
])

# === Tab 1: Recommendation Engine ===
with tab1:
    st.subheader("Your Personalized Health Recommendation")

    st.markdown("#### 📁 Upload Patient Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV for batch processing:", type=["csv"])
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df["Prediction"] = model.predict(uploaded_df)
        st.dataframe(uploaded_df)
        st.download_button("📥 Download Predictions", uploaded_df.to_csv(index=False), "batch_predictions.csv")

    if st.sidebar.button("💡 Generate Recommendation"):
        input_df = pd.DataFrame({"Frequency": [frequency], "Monetary": [monetary], "Time": [time]})
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = f"{probs[prediction]*100:.2f}%"

        def generate_recommendation(pred_label):
            return {
                0: "✅ **Low Risk**\nMaintain your current healthy lifestyle. Annual check-ups recommended.",
                1: "⚠️ **Medium Risk**\nIncrease physical activity, monitor diet. Schedule a medical consultation.",
                2: "🚨 **High Risk**\nImmediate medical attention advised. Begin treatment under supervision.",
            }.get(pred_label, "❓ No recommendation available.")

        recommendation = generate_recommendation(prediction)
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

        with st.spinner("🔬 Analyzing treatment options using Jake AI..."):
            try:
                model_ai = genai.GenerativeModel("gemini-1.5-flash-latest")
                chat = model_ai.start_chat(history=[])
                prompt = f"{health_summary}\n\nBased on the patient's profile, suggest:\n- Likely cause of risk\n- Recommended treatment or lifestyle changes\n- Specialist referrals\n- Rationale for the recommendation"
                ai_response = chat.send_message(prompt)
                st.markdown("### 🧠 Jake's Auto Analysis")
                st.success("✅ AI-driven treatment suggestions:")
                st.markdown(ai_response.text)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    pdf_path = generate_pdf_report(health_summary, ai_response.text, path=tmp_pdf.name)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📄 Download Full PDF Report", f, file_name="healthcare_ai_report.pdf", mime="application/pdf")

                st.download_button("📥 Download Treatment Plan", ai_response.text, "treatment_plan.txt")
            except Exception as e:
                st.error(f"Jake AI Error: {e}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Risk Level", ["Low", "Medium", "High"][prediction])
            st.metric("Confidence Score", confidence)
            st.markdown(recommendation)

            st.markdown("#### 🔍 Feature Impact")
            st.markdown("#### 🧠 Model Explainability (SHAP)")
            with st.expander("Show SHAP values"):
                try:
                    classifier_step = next(step for step in model.named_steps if hasattr(model.named_steps[step], "coef_"))
                    classifier = model.named_steps[classifier_step]
                    importance = classifier.coef_[0]
                except Exception:
                    importance = [0, 0, 0]

            fig_imp = px.bar(x=["Frequency", "Monetary", "Time"], y=importance, labels={"x": "Features", "y": "Importance"}, title="Feature Contribution to Risk")
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Your Health Snapshot")
            patient_df = pd.DataFrame({"Metric": ["Visit Frequency", "Spending", "Time Since Last Visit"], "Value": [frequency, monetary, time]})
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
        fig_class = px.pie(df, names="Class", title="Class Breakdown", hole=0.3)
        st.plotly_chart(fig_class, use_container_width=True)
        st.markdown("#### Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing Count"))

# === Tab 3: Model Insights ===
with tab3:
    st.subheader("📈 Model Performance & Insights")
    st.markdown("#### Feature Correlation Heatmap")
    fig_corr = px.imshow(df.corr(numeric_only=True), title="Correlations", color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### Feature Distribution by Class")
    selected_feat = st.selectbox("Select Feature:", ["Frequency", "Monetary", "Time"])
    fig_dist = px.box(df, x="Class", y=selected_feat, color="Class", title=f"{selected_feat} by Risk Class")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### Pairwise Feature Plot")
    st.image("images/pairplot.png", caption="Pairplot of Patient Data")

# === Tab 4: Chat with AI ===
with tab4:
    st.subheader("🤖 AI Chat Assistant")
    st.markdown("Ask personalized health questions — Jake AI will help you make better decisions.")

    user_input = st.text_area("💬 Enter your question:")
    if st.button("🚀 Ask AI"):
        with st.spinner("Jake is generating a response..."):
            try:
                chat_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                chat = chat_model.start_chat(history=[])
                health_context = st.session_state.get("health_summary", "")
                prompt = f"{health_context}\n\nBased on the above profile, answer:\n{user_input}"
                response = chat.send_message(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI error: {e}")

# === Tab 5: About ===
with tab5:
    st.subheader("ℹ️ About")
    st.markdown("""
    This is a smart, AI-powered healthcare advisor built with:
    - 🤖 ML: Logistic Regression Model
    - 📊 DA: Streamlit + Plotly visualizations
    - 🧠 AI: Jake conversational agent
    - 💡 Future-Ready: SHAP, AutoML, Batch Prediction, Treatment Intelligence

    **Developer:** Jayanth (Full Stack Developer & AI Enthusiast)  
    🔗 [GitHub Repository](https://github.com/Jayanth2323/HealthCare)
    """)

# === Tab 6: Contact ===
with tab6:
    st.subheader("📞 Contact")
    st.markdown("👉 [Email](mailto:Jayanthchennoju@gmail.com) me for any inquiries or feedback.")