import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# Load model and data
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/logistic_regression_pipeline.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/cleaned_blood_data.csv")
    except FileNotFoundError:
        st.error("Data file not found. Please check the path.")
        return pd.DataFrame()  # Return an empty DataFrame


model = load_model()
df = load_data()


# Recommendation logic
def generate_recommendation(prediction_label):
    recommendations = {
        0: """Low Risk: Maintain current lifestyle.
        Regular annual check-ups recommended.""",
        1: """Medium Risk: Increase physical activity and monitor diet.
        Schedule a medical consultation.""",
        2: """High Risk: Immediate medical attention advised.
        Begin treatment under supervision.""",
    }
    return recommendations.get(prediction_label, "No recommendation available.")


# Dashboard layout
st.set_page_config(
    page_title="Healthcare Recommendation System", layout="wide", page_icon="🩺"
)

# Sidebar for user inputs
st.sidebar.title("Patient Input Parameters")
frequency = st.sidebar.slider("Visit Frequency (visits per year)", 0, 50, 5)
monetary = st.sidebar.slider("Healthcare Spending ($)", 0, 10000, 500)
time = st.sidebar.slider("Time Since Last Visit (months)", 0, 60, 12)

# Main dashboard
st.title("🩺 Personalized Healthcare Recommendation Dashboard")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Recommendation", "Data Overview", "Model Analysis", "About"]
)

with tab1:
    st.header("Personalized Health Recommendation")

    if st.sidebar.button("Get Recommendation"):
        if frequency < 0 or monetary < 0 or time < 0:
            st.error("Please enter non-negative values for all parameters.")
        else:
            input_data = pd.DataFrame(
                {
                    "Frequency": [frequency],
                    "Monetary": [monetary],
                    "Time": [time],
                }
            )

            prediction = model.predict(input_data)[0]
            recommendation = generate_recommendation(prediction)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""### 🧠 Predicted Health Risk Level: **{
                        ['Low', 'Medium', 'High'][prediction]}**"""
                )
                st.success(f"✅ Recommended Action: {recommendation}")

                # Display feature importance
                st.markdown("### Feature Importance")
                features = ["Frequency", "Monetary", "Time"]
                importance = model.named_steps["classifier"].coef_[0]
                fig_imp = px.bar(
                    x=features,
                    y=importance,
                    labels={"x": "Features", "y": "Importance"},
                    title="Feature Importance for Risk Prediction",
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            with col2:
                # Visualize patient's position
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
        st.info(
            """👈 Adjust the parameters in the sidebar and
            click 'Get Recommendation'"""
        )

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

with tab3:
    st.header("Model Analysis")

    # Correlation matrix
    st.markdown("### Feature Correlation Matrix")
    fig_corr = px.imshow(
        df.corr(numeric_only=True),
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Feature distributions
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

    # Pairplot
    st.markdown("### Feature Relationships")
    st.image("images/pairplot.png", caption="Pairplot of Features by Risk Class")


with tab4:
    st.header("About This Dashboard")
    st.markdown(
        """
        ### Personalized Healthcare Recommendation System
        This dashboard provides:
        - Personalized health risk assessments
        - Data-driven recommendations
        - Interactive visualization of health metrics

        **How it works:**
        1. Adjust your health parameters in the sidebar
        2. Click 'Get Recommendation'
        3. View your personalized risk assessment and recommendations

        **Model Details:**
        - Algorithm: Logistic Regression
        - Features: Visit frequency, healthcare spending, time since last visit
        - Target: Risk classification (Low, Medium, High)

        **Data Source:** Blood test results and healthcare utilization metrics
        """
    )

    st.markdown("---")
    st.markdown("Developed by [Jayanth] | [UnifiedMentor]")
