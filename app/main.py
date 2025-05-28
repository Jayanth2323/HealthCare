import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import datetime

st.set_page_config(page_title="AI Healthcare Assistant", layout="wide", page_icon="🧠")


# === Load Model and Data ===
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/logistic_regression_pipeline.pkl")
    except FileNotFoundError:
        st.error("Model file not found at 'models/logistic_regression_pipeline.pkl'.")
        return None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/cleaned_blood_data.csv")
    except FileNotFoundError:
        st.error("Data file not found at 'data/cleaned_blood_data.csv'.")
        return pd.DataFrame()


model = load_model()
df = load_data()

if df.empty:
    st.stop()


# === Recommendation Logic ===
def generate_recommendation(pred):
    return {
        0: "🟢 Low Risk: Maintain a healthy lifestyle. Regular check-ups recommended.",
        1: "🟠 Medium Risk: Increase physical activity. Consult with a healthcare provider.",
        2: "🔴 High Risk: Immediate medical consultation recommended. Initiate treatment as advised.",
    }.get(pred, "No recommendation available.")


# === Sidebar Inputs ===
st.sidebar.title("🧾 Patient Profile")
frequency = st.sidebar.slider("Visit Frequency (per year)", 0, 50, 5)
monetary = st.sidebar.slider("Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("Time Since Last Visit (months)", 0, 60, 12)

st.sidebar.markdown("---")
st.sidebar.write("### Current Date & Time")
st.sidebar.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# === Main Title ===
st.title("🧠 AI-Powered Personalized Healthcare Assistant")
st.markdown("### Experience smarter, data-driven health guidance.")

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(
    ["🩺 Recommendation", "📊 Data Overview", "📈 Model Insights", "ℹ️ About"]
)

# === Tab 1: Recommendation ===
with tab1:
    st.header("💡 Smart Health Recommendation")

    if st.sidebar.button("🚀 Generate Recommendation"):
        input_df = pd.DataFrame(
            {"Frequency": [frequency], "Monetary": [monetary], "Time": [time]}
        )

        prediction = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        confidence = f"{probas[prediction] * 100:.2f}%"

        recommendation = generate_recommendation(prediction)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(
                "Predicted Risk Level",
                ["Low", "Medium", "High"][prediction],
                delta=confidence,
            )
            st.success(recommendation)

            st.markdown("---")
            st.markdown("#### 🔍 Feature Impact")
            try:
                step = next(
                    s
                    for s in model.named_steps
                    if hasattr(model.named_steps[s], "coef_")
                )
                weights = model.named_steps[step].coef_[0]
            except Exception:
                weights = [0, 0, 0]

            fig_imp = px.bar(
                x=["Frequency", "Monetary", "Time"],
                y=weights,
                labels={"x": "Feature", "y": "Importance"},
                title="📌 Feature Contribution to Prediction",
                color=["Frequency", "Monetary", "Time"],
                text=weights,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            report = f"""
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔍 Risk Level: {["Low", "Medium", "High"][prediction]}
📊 Confidence: {confidence}
🩺 Recommendation: {recommendation}

📌 Input Metrics:
- Frequency: {frequency} visits/year
- Spending: ${monetary}
- Time Since Last Visit: {time} months
"""
            st.download_button(
                "📥 Download Report", report, file_name="health_ai_report.txt"
            )

        with col2:
            st.markdown("#### 📈 Health Profile vs Population Average")
            user_metrics = pd.DataFrame(
                {
                    "Metric": ["Frequency", "Monetary", "Time"],
                    "User": [frequency, monetary, time],
                    "Average": [
                        df["Frequency"].mean(),
                        df["Monetary"].mean(),
                        df["Time"].mean(),
                    ],
                }
            )

            fig_profile = px.bar(
                user_metrics.melt(
                    id_vars="Metric", var_name="Category", value_name="Value"
                ),
                x="Metric",
                y="Value",
                color="Category",
                barmode="group",
                title="User vs Average Health Metrics",
            )
            st.plotly_chart(fig_profile, use_container_width=True)
    else:
        st.info(
            "👈 Use the sidebar to input patient details and click 'Generate Recommendation'."
        )

# === Tab 2: Data Overview ===
with tab2:
    st.header("📊 Dataset Snapshot")
    st.subheader("Data Sample")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Class Distribution")
    st.plotly_chart(
        px.pie(df, names="Class", hole=0.4, title="Risk Category Distribution")
    )

    st.subheader("Missing Value Analysis")
    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

# === Tab 3: Model Analysis ===
with tab3:
    st.header("📈 Model Insights")

    st.subheader("Feature Correlation")
    st.plotly_chart(
        px.imshow(
            df.corr(numeric_only=True),
            color_continuous_scale="RdBu",
            title="Feature Correlation Heatmap",
        )
    )

    st.subheader("Feature Distributions")
    feature = st.selectbox(
        "Choose Feature to Explore", ["Frequency", "Monetary", "Time"]
    )
    st.plotly_chart(
        px.box(
            df, x="Class", y=feature, color="Class", title=f"{feature} by Risk Class"
        )
    )

# === Tab 4: About ===
with tab4:
    st.header("ℹ️ About This AI Assistant")
    st.markdown(
        """
    ### AI-Powered Personalized Healthcare Dashboard
    This interactive tool provides:
    - Personalized health risk assessments
    - AI-generated lifestyle recommendations
    - Visual comparisons with population averages

    **Technology Stack:**
    - Streamlit
    - Plotly
    - Logistic Regression

    **Developer:** Jayanth  
    **GitHub:** [HealthCare Project](https://github.com/Jayanth2323/HealthCare)
    """
    )
