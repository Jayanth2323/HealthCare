import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from dotenv import load_dotenv

# === Load API Key Securely ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "❌ Gemini API key not found. Please set GOOGLE_API_KEY in your .env file."
    )
    st.stop()
genai.configure(api_key=api_key)

# === Streamlit Config ===
st.set_page_config(page_title="AI Healthcare Advisor", layout="wide", page_icon="🧠")


# === Load Model and Data ===
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/logistic_regression_pipeline.pkl")
    except FileNotFoundError:
        st.error(
            "❌ Model file not found. Please check 'models/logistic_regression_pipeline.pkl'."
        )
        return None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/cleaned_blood_data.csv")
    except FileNotFoundError:
        st.error("❌ Data file not found. Please check 'data/cleaned_blood_data.csv'.")
        return pd.DataFrame()


model = load_model()
df = load_data()

if df.empty:
    st.warning("⚠️ No data available to display. Please verify your dataset.")
    st.stop()


# === Recommendation Logic ===
def generate_recommendation(prediction_label):
    recommendations = {
        0: "✅ **Low Risk**\nMaintain your current healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ **Medium Risk**\nIncrease physical activity, monitor diet. Schedule a medical consultation.",
        2: "🚨 **High Risk**\nImmediate medical attention advised. Begin treatment under supervision.",
    }
    return recommendations.get(prediction_label, "❓ No recommendation available.")


# === Sidebar Inputs ===
st.sidebar.header("📝 Patient Profile")
frequency = st.sidebar.slider("📅 Visit Frequency (visits/year)", 0, 50, 5)
monetary = st.sidebar.slider("💸 Annual Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("⏳ Time Since Last Visit (months)", 0, 60, 12)

# === Main UI ===
st.title("🧠 AI-Driven Personalized Healthcare Advisor")
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
    if st.sidebar.button("💡 Generate Recommendation"):
        input_data = pd.DataFrame(
            {"Frequency": [frequency], "Monetary": [monetary], "Time": [time]}
        )
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        confidence = f"{probs[prediction]*100:.2f}%"
        recommendation = generate_recommendation(prediction)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Predicted Risk Level", ["Low", "Medium", "High"][prediction])
            st.metric("Prediction Confidence", confidence)
            st.markdown(recommendation)

            st.markdown("#### 🔍 Feature Impact")
            features = ["Frequency", "Monetary", "Time"]
            try:
                classifier_step = next(
                    step
                    for step in model.named_steps
                    if hasattr(model.named_steps[step], "coef_")
                )
                importance = model.named_steps[classifier_step].coef_[0]
            except Exception:
                importance = [0, 0, 0]

            fig_imp = px.bar(
                x=features,
                y=importance,
                labels={"x": "Features", "y": "Weight"},
                title="Feature Contribution to Risk",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            report = f"""
            🔍 **AI Healthcare Report**
            --------------------------
            Risk Level: {['Low', 'Medium', 'High'][prediction]}
            Confidence: {confidence}
            Recommendation: {recommendation}

            Metrics:
            - Visit Frequency: {frequency}
            - Spending: ${monetary}
            - Time Since Last Visit: {time} months
            """
            st.download_button(
                "📥 Download Report", report, file_name="health_report.txt"
            )

        with col2:
            st.markdown("#### 🧾 Your Health Snapshot")
            patient_data = pd.DataFrame(
                {
                    "Metric": ["Visit Frequency", "Spending", "Time Since Last Visit"],
                    "Value": [frequency, monetary, time],
                }
            )

            avg_values = [
                df["Frequency"].mean(),
                df["Monetary"].mean(),
                df["Time"].mean(),
            ]

            fig_patient = go.Figure()
            fig_patient.add_trace(
                go.Bar(
                    x=patient_data["Metric"],
                    y=patient_data["Value"],
                    name="You",
                    text=patient_data["Value"],
                    textposition="auto",
                )
            )
            fig_patient.add_trace(
                go.Scatter(
                    x=patient_data["Metric"],
                    y=avg_values,
                    mode="markers+lines",
                    name="Population Avg",
                    marker=dict(size=10, color="red"),
                )
            )
            fig_patient.update_layout(
                title="Your Metrics vs Population Averages", barmode="group"
            )
            st.plotly_chart(fig_patient, use_container_width=True)
    else:
        st.info("👈 Set parameters in the sidebar and click 'Generate Recommendation'")

# === Tab 2: Data Intelligence ===
with tab2:
    st.subheader("📂 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sample Data")
        st.dataframe(df.head())
        st.markdown("#### Summary Stats")
        st.dataframe(df.describe())

    with col2:
        st.markdown("#### Class Breakdown")
        fig_class = px.pie(df, names="Class", title="Risk Class Distribution", hole=0.3)
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("#### Missing Data")
        st.dataframe(df.isnull().sum().to_frame(name="Missing Values"))

# === Tab 3: Model Insights ===
with tab3:
    st.subheader("📈 Model Performance & Interpretability")

    st.markdown("#### Correlation Heatmap")
    fig_corr = px.imshow(
        df.corr(numeric_only=True),
        title="Feature Correlations",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### Feature Distributions")
    feature = st.selectbox(
        "Select a feature to visualize:", ["Frequency", "Monetary", "Time"]
    )
    fig_dist = px.box(
        df,
        x="Class",
        y=feature,
        color="Class",
        title=f"{feature} Distribution by Risk Class",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### Pairwise Feature Relationships")
    st.image("images/pairplot.png", caption="Pairplot of Feature Interactions")

# === Tab 4: AI Chat Assistant ===
with tab4:
    st.subheader("🤖 AI Chat Assistant")
    st.markdown(
        "🧠 Ask your health-related queries and get answers powered by Gemini AI."
    )

    model_list = ["gemini-pro", "gemini-1.5-flash-latest"]
    selected_model = st.selectbox("Choose a Gemini model:", model_list)
    user_input = st.text_area("💬 Enter your health question:")

    if st.button("🚀 Ask AI"):
        with st.spinner("Thinking..."):
            try:
                model = genai.GenerativeModel(model_name=selected_model)
                chat = model.start_chat(history=[])
                response = chat.send_message(user_input)
                st.success("✅ Response received:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error fetching AI response: {str(e)}")

# === Tab 5: About ===
with tab5:
    st.subheader("ℹ️ About This Application")
    st.markdown(
        """
    This AI-powered dashboard provides real-time personalized healthcare recommendations based on user inputs.

    **Technologies Used:**
    - Logistic Regression Model
    - Streamlit for UI
    - Plotly for Visualization
    - Google Gemini AI for conversational assistance

    **Key Features:**
    - Confidence-based prediction
    - Feature impact explanations
    - Visual health metrics comparison
    - Downloadable recommendation reports
    - Conversational AI for health inquiries

    **Developed by:** Jayanth | Full Stack Developer & AI Enthusiast  
    🔗 [GitHub Repository](https://github.com/Jayanth2323/HealthCare)
    """
    )
