import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# ==== CONFIG ====
st.set_page_config(
    page_title="AI Healthcare Recommender", layout="wide", page_icon="🧬"
)

st.markdown(
    """
    <style>
        .reportview-container {
            background: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
        }
        .metric-label {
            font-weight: bold;
            font-size: 1.2em;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ==== LOAD MODEL AND DATA ====
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/logistic_regression_pipeline.pkl")
    except FileNotFoundError:
        st.error(
            "Model file not found. Please check 'models/logistic_regression_pipeline.pkl'."
        )
        return None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/cleaned_blood_data.csv")
    except FileNotFoundError:
        st.error("Data file not found. Please check 'data/cleaned_blood_data.csv'.")
        return pd.DataFrame()


model = load_model()
df = load_data()

if df.empty:
    st.stop()


# ==== AI LOGIC ====
def generate_ai_recommendation(pred):
    persona = [
        (
            "🟢 Low",
            "You are in great shape! Maintain your routine and monitor annually.",
        ),
        (
            "🟡 Medium",
            "Caution! Adjust lifestyle and consult with a healthcare provider.",
        ),
        (
            "🔴 High",
            "Urgent attention required. Schedule a medical evaluation immediately.",
        ),
    ]
    return persona[pred]


# ==== SIDEBAR ====
st.sidebar.header("🧪 Patient Parameters")
frequency = st.sidebar.slider("Annual Visit Frequency", 0, 50, 5)
monetary = st.sidebar.slider("Annual Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("Months Since Last Visit", 0, 60, 12)

# ==== MAIN ====
st.title("🤖 AI-Powered Personalized Healthcare Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(
    ["AI Recommendation", "Data Explorer", "Model Insights", "About"]
)

# ==== TAB 1: RECOMMENDATION ====
with tab1:
    st.header("🧠 AI Health Assistant")
    if st.sidebar.button("Analyze Health Profile"):
        input_df = pd.DataFrame(
            {"Frequency": [frequency], "Monetary": [monetary], "Time": [time]}
        )
        prediction = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = f"{probs[prediction]*100:.2f}%"

        label, recommendation = generate_ai_recommendation(prediction)

        st.success(f"### {label} Risk\n{recommendation}")
        st.metric("Prediction Confidence", confidence)

        # Feature Importance
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
            x=["Frequency", "Monetary", "Time"],
            y=importance,
            labels={"x": "Feature", "y": "Importance"},
            title="🔍 Feature Importance",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Download Report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report = f"""Health AI Report\nDate: {timestamp}\n\nRisk Level: {label}\nConfidence: {confidence}\n\nRecommendation:\n{recommendation}\n\nInputs:\n- Frequency: {frequency}\n- Spending: ${monetary}\n- Time Since Last Visit: {time} months"""
        st.download_button(
            "📥 Download AI Health Report",
            report,
            file_name=f"Health_Report_{timestamp}.txt",
        )

        # Profile Comparison
        avg_vals = [df[col].mean() for col in ["Frequency", "Monetary", "Time"]]
        fig_profile = px.bar(
            pd.DataFrame(
                {
                    "Metric": ["Frequency", "Monetary", "Time"],
                    "You": [frequency, monetary, time],
                    "Population Avg": avg_vals,
                }
            ),
            x="Metric",
            y=["You", "Population Avg"],
            barmode="group",
            title="📊 Profile vs Population",
        )
        st.plotly_chart(fig_profile, use_container_width=True)
    else:
        st.info("👈 Adjust parameters and click 'Analyze Health Profile'")

# ==== TAB 2: DATA OVERVIEW ====
with tab2:
    st.header("📂 Data Overview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")
    st.subheader("📌 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader("📊 Risk Distribution")
    st.plotly_chart(
        px.pie(df, names="Class", hole=0.3, title="Risk Class Proportions"),
        use_container_width=True,
    )

# ==== TAB 3: MODEL INSIGHTS ====
with tab3:
    st.header("📈 Model Diagnostic")
    st.subheader("🔗 Feature Correlations")
    st.plotly_chart(
        px.imshow(
            df.corr(numeric_only=True),
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
        ),
        use_container_width=True,
    )

    st.subheader("📌 Distribution by Class")
    feature = st.selectbox("Select Feature", ["Frequency", "Monetary", "Time"])
    st.plotly_chart(
        px.box(
            df, x="Class", y=feature, color="Class", title=f"{feature} by Risk Class"
        ),
        use_container_width=True,
    )

# ==== TAB 4: ABOUT ====
with tab4:
    st.header("ℹ️ About")
    st.markdown(
        """
    ### AI-Powered Personalized Healthcare
    This dashboard delivers:
    - Real-time health risk predictions
    - AI-driven action recommendations
    - Data transparency & visual analytics

    **Model:** Logistic Regression
    **Features:** Frequency, Spending, Time since last visit

    **Developed by:** Jayanth  
    [GitHub](https://github.com/Jayanth2323/HealthCare)
    """
    )
