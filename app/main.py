import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Healthcare Recommendation System",
    layout="wide",
    page_icon="🩺"
)


# === Load Model and Data ===
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/logistic_regression_pipeline.pkl")
    except FileNotFoundError:
        st.error(
            """
            ❌ Model file not found.
            Please check 'models/logistic_regression_pipeline.pkl'."""
        )
        return None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/cleaned_blood_data.csv")
    except FileNotFoundError:
        st.error("""
                ❌ Data file not found.
                Please check 'data/cleaned_blood_data.csv'.""")
        return pd.DataFrame()


model = load_model()
df = load_data()

if df.empty:
    st.warning("⚠️ No data available to display. Please verify your dataset.")
    st.stop()


# === Recommendation Logic ===
def generate_recommendation(prediction_label):
    recommendations = {
        0: """Low Risk: Maintain current lifestyle.
        Regular annual check-ups recommended.""",
        1: """Medium Risk: Increase physical activity and monitor diet.
        Schedule a medical consultation.""",
        2: """High Risk: Immediate medical attention advised.
        Begin treatment under supervision.""",
    }
    return recommendations.get(
        prediction_label, "No recommendation available.")


# === UI Config ===


# === Sidebar Inputs ===
st.sidebar.title("Patient Input Parameters")
frequency = st.sidebar.slider("Visit Frequency (visits per year)", 0, 50, 5)
monetary = st.sidebar.slider("Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("Time Since Last Visit (months)", 0, 60, 12)

# === Main Dashboard ===
st.title("🩺 Personalized Healthcare Recommendation Dashboard")

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(
    ["Recommendation", "Data Overview", "Model Analysis", "About"]
)

# === Tab 1: Recommendation ===
with tab1:
    st.header("Personalized Health Recommendation")

    if st.sidebar.button("Get Recommendation"):
        input_data = pd.DataFrame(
            {
                "Frequency": [frequency],
                "Monetary": [monetary],
                "Time": [time],
            }
        )

        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        confidence = f"{probs[prediction]*100:.2f}%"
        recommendation = generate_recommendation(prediction)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                ### 🧠 Predicted Health Risk Level: **{
                    ['Low', 'Medium', 'High'][prediction]}**"""
            )
            st.metric("📊 Prediction Confidence", confidence)
            st.success(f"✅ Recommended Action: {recommendation}")

            # Feature Importance
            st.markdown("### Feature Importance")
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
                title="Feature Importance for Risk Prediction",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # Downloadable Report
            report = f"""Health Risk Level: {
                ['Low', 'Medium', 'High'],
                [prediction]
                }
                \nConfidence: {confidence}
                \nRecommendation: {recommendation}
                \n\nMetrics:\n- Frequency: {frequency}
                \n- Spending: ${monetary}
                \n- Time Since Last Visit: {time} months"""
            st.download_button(
                "📥 Download Report", report, file_name="health_report.txt"
            )

        with col2:
            st.markdown("### Your Health Profile")
            patient_data = {
                "Metric": [
                    "Visit Frequency",
                    "Healthcare Spending",
                    "Time Since Last Visit",
                ],
                "Value": [frequency, monetary, time],
            }
            fig_patient = px.bar(
                pd.DataFrame(patient_data),
                x="Metric",
                y="Value",
                title="Your Health Metrics Compared to Averages",
                text="Value",
            )
            avg_values = [
                df["Frequency"].mean(),
                df["Monetary"].mean(),
                df["Time"].mean(),
            ]
            fig_patient.add_scatter(
                x=patient_data["Metric"],
                y=avg_values,
                mode="markers",
                name="Population Average",
                marker=dict(color="red", size=12),
            )
            st.plotly_chart(fig_patient, use_container_width=True)
    else:
        st.info("""
                👈 Adjust the parameters in the sidebar and
                click 'Get Recommendation'.
                """)


# === Tab 2: Data Overview ===
with tab2:
    st.header("Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dataset Sample")
        st.dataframe(df.head())

        st.markdown("### Basic Statistics")
        st.dataframe(df.describe())

    with col2:
        st.markdown("### Class Distribution")
        fig_class = px.pie(
            df, names="Class", title="Distribution of Risk Classes", hole=0.3
        )
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("### Missing Values")
        st.dataframe(df.isnull().sum().to_frame(name="Missing Values"))

# === Tab 3: Model Analysis ===
with tab3:
    st.header("Model Analysis")

    st.markdown("### Feature Correlation Matrix")
    fig_corr = px.imshow(
        df.corr(numeric_only=True),
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Feature Distributions by Risk Class")
    feature = st.selectbox(
        "Select feature to visualize:", ["Frequency", "Monetary", "Time"]
    )
    fig_dist = px.box(
        df,
        x="Class",
        y=feature,
        color="Class",
        title=f"{feature} Distribution by Risk Class",
        labels={"Class": "Risk Level"},
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Feature Relationships")
    st.image(
        "images/pairplot.png", caption="Pairplot of Features by Risk Class")

# === Tab 4: About ===
with tab4:
    st.header("About This Dashboard")
    st.markdown(
        """
    ### Personalized Healthcare Recommendation System
    This dashboard provides:
    - Personalized health risk assessments
    - Data-driven recommendations
    - Interactive visualizations of health metrics

    **How it works:**
    1. Adjust your health parameters in the sidebar
    2. Click 'Get Recommendation'
    3. View your personalized risk assessment and recommended actions

    **Model Details:**
    - Algorithm: Logistic Regression
    - Features: Visit frequency, healthcare spending, time since last visit
    - Target: Risk classification (Low, Medium, High)

    **Data Source:** Blood test results and healthcare utilization metrics
    """
    )
    st.markdown("---")
    st.markdown(
        """
        Developed by **Jayanth**, Full Stack Developer |
        Data Scientist | 
        Machine Learning Enthusiast | 
        Python Enthusiast
        🚀 [GitHub](https://github.com/Jayanth2323/HealthCare)"""
    )
