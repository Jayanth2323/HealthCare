import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Personalized Healthcare Advisor", layout="wide", page_icon="🧠"
)


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
        0: "🟢 Low Risk: Maintain current lifestyle. Annual check-ups recommended.",
        1: "🟡 Medium Risk: Increase physical activity, monitor diet, and consult a physician.",
        2: "🔴 High Risk: Immediate medical attention advised. Begin treatment under supervision.",
    }
    return recommendations.get(prediction_label, "⚠️ No recommendation available.")


# === Sidebar Inputs ===
st.sidebar.header("🔍 Input Your Health Parameters")
frequency = st.sidebar.slider("📅 Visit Frequency (visits/year)", 0, 50, 5)
monetary = st.sidebar.slider("💵 Healthcare Spending ($/year)", 0, 10000, 500)
time = st.sidebar.slider("⏳ Time Since Last Visit (months)", 0, 60, 12)

# === Main UI ===
st.title("🧠 Personalized-Powered Healthcare Recommendation System")

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(
    ["🤖 Recommendation", "📊 Data Overview", "📈 Model Analysis", "ℹ️ About"]
)

# === Tab 1: Recommendation ===
with tab1:
    st.header("📌 Personalized Health Insight")
    if st.sidebar.button("🧬 Generate Recommendation"):
        input_data = pd.DataFrame(
            {"Frequency": [frequency], "Monetary": [monetary], "Time": [time]}
        )

        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        confidence = f"{probs[prediction]*100:.2f}%"
        recommendation = generate_recommendation(prediction)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(
                f"📉 Predicted Risk Level: **{['Low', 'Medium', 'High'][prediction]}**"
            )
            st.metric("🔎 Confidence", confidence)
            st.success(f"✅ Recommended Action:\n{recommendation}")

            st.markdown("### 📌 Feature Importance")
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
                labels={"x": "Features", "y": "Importance"},
                title="🚦 Feature Influence on Prediction",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            report = f"""
            Health Risk Level: {['Low', 'Medium', 'High'][prediction]}
            Confidence: {confidence}
            Recommendation: {recommendation}

            Input Metrics:
            - Frequency: {frequency}
            - Spending: ${monetary}
            - Time Since Last Visit: {time} months
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            st.download_button(
                "📥 Download Personalized Report",
                report,
                file_name="health_ai_report.txt",
            )

        with col2:
            st.subheader("📋 Your Health Snapshot")
            patient_data = pd.DataFrame(
                {
                    "Metric": [
                        "Visit Frequency",
                        "Healthcare Spending",
                        "Time Since Last Visit",
                    ],
                    "Value": [frequency, monetary, time],
                }
            )
            avg_values = [
                df["Frequency"].mean(),
                df["Monetary"].mean(),
                df["Time"].mean(),
            ]
            fig_patient = px.bar(
                patient_data,
                x="Metric",
                y="Value",
                text="Value",
                title="🔬 Your Metrics vs. Population Averages",
            )
            fig_patient.add_scatter(
                x=patient_data["Metric"],
                y=avg_values,
                mode="markers",
                name="Population Avg",
                marker=dict(color="red", size=12),
            )
            st.plotly_chart(fig_patient, use_container_width=True)
    else:
        st.info(
            "👈 Adjust sidebar parameters and click 'Generate Recommendation' to begin."
        )

# === Tab 2: Data Overview ===
with tab2:
    st.header("📂 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Dataset Preview")
        st.dataframe(df.head())

        st.markdown("#### 📊 Statistical Summary")
        st.dataframe(df.describe())

    with col2:
        st.markdown("#### 🧮 Risk Class Distribution")
        fig_class = px.pie(
            df, names="Class", title="🧬 Risk Class Composition", hole=0.3
        )
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("#### ⚠️ Missing Values")
        st.dataframe(df.isnull().sum().to_frame(name="Missing Values"))

# === Tab 3: Model Analysis ===
with tab3:
    st.header("📈 Model Insight & Validation")

    st.markdown("#### 🔗 Feature Correlation")
    fig_corr = px.imshow(
        df.corr(numeric_only=True),
        title="🔍 Correlation Heatmap",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### 📤 Distributions by Class")
    feature = st.selectbox("Choose Feature to View:", ["Frequency", "Monetary", "Time"])
    fig_dist = px.box(
        df,
        x="Class",
        y=feature,
        color="Class",
        title=f"📊 {feature} by Health Risk Level",
        labels={"Class": "Risk Level"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### 🧬 Feature Relationships")
    st.image("images/pairplot.png", caption="Pairwise Feature Distribution")

# === Tab 4: About ===
with tab4:
    st.header("ℹ️ About This Personalized HealthCare Dashboard")
    st.markdown(
        """
    ### 🧠 Personalized Healthcare Advisor
    This system leverages logistic regression to assess individual health risk based on historical clinical and financial usage.

    **Features**:
    - Real-time prediction
    - Confidence scoring
    - Population comparisons
    - Downloadable health reports
    - AI-style intuitive UI

    **Data Source:** Blood test results & healthcare visit metrics

    ---
    Developed by **Jayanth** 
    Full Stack Developer | AI & ML Enthusiast 
    🚀 [GitHub](https://github.com/Jayanth2323/HealthCare)
    """
    )
